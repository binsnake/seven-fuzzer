#include "lanes/seven_lane.hpp"

#include <array>
#include <cstdint>
#include <cstring>

#include "seven/compat.hpp"

namespace sf {

namespace {

// seven::CpuState::vectors[i].value is a math::wide_integer type (128/256/512
// bits depending on build config), not a plain uint64 pair. Packed/unpacked
// one byte at a time via shift+mask, the same idiom seven_core's own SIMD
// handlers use internally (see simd_shuffle.cpp's load_elements/store_elements)
// -- avoids relying on any wide-integer API beyond what's already proven to
// compile in this codebase.
[[nodiscard]] seven::SimdUint pack_xmm(std::uint64_t lo, std::uint64_t hi) {
  std::array<std::uint8_t, 16> bytes{};
  std::memcpy(bytes.data(), &lo, 8);
  std::memcpy(bytes.data() + 8, &hi, 8);
  seven::SimdUint out(0);
  for (std::size_t i = 0; i < 16; ++i) {
    out |= (seven::SimdUint(bytes[i]) << static_cast<int>(i * 8));
  }
  return out;
}

void unpack_xmm(const seven::SimdUint& value, std::uint64_t& lo, std::uint64_t& hi) {
  std::array<std::uint8_t, 16> bytes{};
  for (std::size_t i = 0; i < 16; ++i) {
    bytes[i] = static_cast<std::uint8_t>((value >> static_cast<int>(i * 8)) & 0xFFu);
  }
  std::memcpy(&lo, bytes.data(), 8);
  std::memcpy(&hi, bytes.data() + 8, 8);
}

// seven holds the x87 stack in PHYSICAL registers R0..R7 and resolves ST(i) through TOP; sf's
// X87State is stack-relative (see its comment). These two rotate between the conventions. TOP has
// to be established first, via set_x87_status_word, because that is what the rotation is relative
// to.
void apply_x87(seven::CpuState& state, const X87State& x) {
  state.set_x87_control_word(x.control_word);
  state.set_x87_status_word(x.status_word);
  const int top = static_cast<int>(state.get_x87_top());
  for (int i = 0; i < 8; ++i) {
    const auto phys = static_cast<std::size_t>((top + i) & 0x7);
    state.x87_stack[phys].val.signif = x.signif[static_cast<std::size_t>(i)];
    state.x87_stack[phys].val.signExp = x.signexp[static_cast<std::size_t>(i)];
  }
  for (int phys = 0; phys < 8; ++phys) {
    state.x87_tags[static_cast<std::size_t>(phys)] = static_cast<std::uint8_t>((x.tag_word >> (2 * phys)) & 0x3u);
  }
}

void read_x87(const seven::CpuState& state, X87State& x) {
  x.control_word = state.get_x87_control_word();
  x.status_word = state.get_x87_status_word();
  const int top = static_cast<int>(state.get_x87_top());
  for (int i = 0; i < 8; ++i) {
    const auto phys = static_cast<std::size_t>((top + i) & 0x7);
    x.signif[static_cast<std::size_t>(i)] = state.x87_stack[phys].val.signif;
    x.signexp[static_cast<std::size_t>(i)] = state.x87_stack[phys].val.signExp;
  }
  std::uint16_t tw = 0;
  for (int phys = 0; phys < 8; ++phys) {
    tw |= static_cast<std::uint16_t>(static_cast<std::uint16_t>(state.x87_tags[static_cast<std::size_t>(phys)] & 0x3u)
                                      << (2 * phys));
  }
  x.tag_word = tw;
}

[[nodiscard]] Stop map_stop_reason(seven::StopReason r) noexcept {
  using SR = seven::StopReason;
  switch (r) {
    case SR::none:
      return Stop::ok;
    case SR::halted:
      return Stop::halted;
    case SR::invalid_opcode:
    case SR::unsupported_instruction:
    case SR::decode_error:
      return Stop::ud;
    case SR::general_protection:
      return Stop::gp;
    case SR::page_fault:
      return Stop::pf;
    case SR::divide_error:
      return Stop::de;
    default:
      return Stop::other;
  }
}

}  // namespace

LaneOutcome run_seven(const TestCase& tc) {
  LaneOutcome out;

  seven::StandaloneMachine m;
  m.state.mode = seven::ExecutionMode::long64;
  m.state.rip = kCodeBase;
  m.state.rflags = tc.initial.rflags;
  // CPL comes from the CS selector's low two bits, and the hardware lane is a real ring-3 Windows
  // thread. Leaving this zero told seven it was ring 0 while the oracle it is compared against was
  // ring 3, so every privileged instruction disagreed by construction and the whole class had to be
  // kept out of the corpus. 0x33 is the usual long-mode user CS.
  m.state.sreg[1] = 0x33;
  m.state.gpr = tc.initial.gpr;
  for (int i = 0; i < 16; ++i) {
    m.state.vectors[static_cast<std::size_t>(i)].value =
        pack_xmm(tc.initial.xmm_lo[static_cast<std::size_t>(i)], tc.initial.xmm_hi[static_cast<std::size_t>(i)]);
  }
  apply_x87(m.state, tc.initial.x87);

  // All three scratch pages RWX in every lane so permission mismatches never
  // masquerade as execution-semantics differences.
  m.memory.map(kCodeBase, kPageSize);
  m.memory.map(kDataBase, kPageSize);
  m.memory.map(kStackBase, kPageSize);

  // Seed [RSP] with a valid, mapped, executable address so a generated RET
  // lands somewhere real instead of address 0 — see gen_jcc's comment for
  // why unmapped branch destinations are avoided rather than compared.
  {
    const std::uint64_t retaddr = kCodeBase;
    (void)m.memory.write(kStackTop, &retaddr, sizeof(retaddr));
  }
  (void)m.memory.write(kDataBase, tc.data_seed.data(), tc.data_seed.size());

  if (!tc.bytes.empty() &&
      !m.memory.write(kCodeBase, tc.bytes.data(), tc.bytes.size())) {
    out.setup_ok = false;
    out.setup_error = "seven: failed to write code bytes";
    return out;
  }

  const seven::ExecutionResult res = m.executor.step(m.state, m.memory);

  out.stop = map_stop_reason(res.reason);
  if (out.stop == Stop::other) {
    out.detail = "seven StopReason=" + std::to_string(static_cast<int>(res.reason));
  }
  out.after.gpr = m.state.gpr;
  out.after.rip = m.state.rip;
  out.after.rflags = m.state.rflags;
  for (int i = 0; i < 16; ++i) {
    unpack_xmm(m.state.vectors[static_cast<std::size_t>(i)].value, out.after.xmm_lo[static_cast<std::size_t>(i)],
               out.after.xmm_hi[static_cast<std::size_t>(i)]);
  }
  read_x87(m.state, out.after.x87);
  out.captures_x87 = true;
  (void)m.memory.read(kDataBase, out.after.data_after.data(), kDataWindow);

  return out;
}

}  // namespace sf
