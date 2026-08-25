#pragma once

// Shared data contract between the three execution lanes (seven, unicorn,
// hardware). Every lane maps its own register/memory representation onto
// this and nothing else, so the comparator never needs to know lane
// internals.

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace sf {

// Fixed virtual-address layout every lane maps identically. Chosen far from
// typical image/heap/stack ranges so a fresh Windows process is very
// unlikely to already have something there.
inline constexpr std::uint64_t kCodeBase   = 0x0000'1000'0000'0000ull;
inline constexpr std::uint64_t kDataBase   = 0x0000'1000'0000'1000ull;
inline constexpr std::uint64_t kStackBase  = 0x0000'1000'0000'2000ull;  // one page
inline constexpr std::uint64_t kStackTop   = kStackBase + 0x0F00ull;    // headroom for pushes
inline constexpr std::size_t   kPageSize   = 0x1000;
inline constexpr std::size_t   kDataWindow = 0x100;  // bytes of scratch data compared post-step

// x86 general register numbering: 0=RAX,1=RCX,2=RDX,3=RBX,4=RSP,5=RBP,
// 6=RSI,7=RDI,8..15=R8..R15. Matches seven's CpuState.gpr[] convention.
struct RegState {
  std::array<std::uint64_t, 16> gpr{};
  std::uint64_t rip = 0;
  std::uint64_t rflags = 0x202;  // reserved bit1=1, IF=1
  // XMM0..XMM15, low/high 64 bits of each. Legacy-SSE/VEX.128 scope only —
  // AVX2 YMM upper halves and AVX-512 ZMM/mask state are out of scope (see
  // Instruction Generator notes for why).
  std::array<std::uint64_t, 16> xmm_lo{};
  std::array<std::uint64_t, 16> xmm_hi{};
  std::array<std::uint8_t, kDataWindow> data_after{};  // only meaningful post-step, on LaneOutcome::after
};

// Status flags only — the ones ALU/shift/control-flow instructions define.
// IF/RF/TF/IOPL/reserved bits are fixed by the harness and excluded here so
// scheduling/debugging artifacts never show up as false mismatches.
inline constexpr std::uint64_t kCompareFlagsMask =
    0x0001u /*CF*/ | 0x0004u /*PF*/ | 0x0010u /*AF*/ | 0x0040u /*ZF*/ |
    0x0080u /*SF*/ | 0x0400u /*DF*/ | 0x0800u /*OF*/;

struct TestCase {
  std::string text;                    // disassembly, for reporting
  std::vector<std::uint8_t> bytes;      // encoded instruction
  RegState initial;
  bool touches_memory = false;
  // Some instructions leave certain flags architecturally undefined
  // (BSF/BSR define only ZF; BT/BTS/BTR/BTC define only CF; multi-bit
  // shifts/rotates leave OF undefined). Comparing undefined bits as if they
  // were a spec would report two equally-correct engines as diverging, so
  // the generator narrows this per instruction instead of the comparator
  // guessing after the fact.
  std::uint64_t flags_mask = kCompareFlagsMask;
  // Pre-fill for the data scratch page (all three lanes write this before
  // executing) so memory-source operands don't always read zero — a
  // constant-zero source is a real edge case worth occasional coverage
  // (e.g. BSF/BSR-of-zero has its own undefined-destination behavior) but
  // shouldn't be the *only* value ever exercised.
  std::array<std::uint8_t, kDataWindow> data_seed{};
  // Bit i set means GPR i is meaningful to compare across lanes. BSF/BSR
  // leave the destination register's value fully undefined by spec when the
  // source operand is zero -- confirmed via the --probe-bsr-zero ground
  // truth check that real hardware leaves it completely untouched while
  // Unicorn incorrectly zero-extends it (seven's interpreter and seven-jit's
  // callout bridge both correctly match hardware here). The generator clears
  // the destination's bit when it detects this exact case so the comparator
  // doesn't report a known Unicorn quirk as a seven/seven-jit finding.
  std::uint32_t gpr_compare_mask = 0xFFFFu;
};

// Normalized fault vocabulary every lane maps its own exception model onto.
enum class Stop {
  ok,       // instruction retired normally (including a taken branch)
  ud,       // invalid/unsupported opcode
  gp,       // general protection / privileged instruction in ring 3
  pf,       // page fault / access violation
  de,       // divide error
  bp,       // breakpoint (unexpected int3 mid-instruction stream)
  halted,   // HLT actually halted (vs. faulting) — kept distinct from `ok`
            // because whether HLT faults in ring 3 is itself a thing worth
            // comparing, not something to paper over.
  other,    // anything else — see detail
};

[[nodiscard]] inline const char* stop_name(Stop s) noexcept {
  switch (s) {
    case Stop::ok: return "ok";
    case Stop::ud: return "ud";
    case Stop::gp: return "gp";
    case Stop::pf: return "pf";
    case Stop::de: return "de";
    case Stop::bp: return "bp";
    case Stop::halted: return "halted";
    default: return "other";
  }
}

struct LaneOutcome {
  bool setup_ok = true;     // false = harness-internal failure, not a CPU result
  std::string setup_error;
  Stop stop = Stop::ok;
  std::string detail;       // free-form, only for Stop::other / diagnostics
  RegState after{};
};

}  // namespace sf
