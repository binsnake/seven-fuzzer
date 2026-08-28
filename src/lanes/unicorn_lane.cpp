#include "lanes/unicorn_lane.hpp"

#include <array>
#include <cstring>
#include <string>

#include <unicorn/unicorn.h>

namespace sf {

namespace {

constexpr std::array<int, 16> kGprRegIds = {
    UC_X86_REG_RAX, UC_X86_REG_RCX, UC_X86_REG_RDX, UC_X86_REG_RBX,
    UC_X86_REG_RSP, UC_X86_REG_RBP, UC_X86_REG_RSI, UC_X86_REG_RDI,
    UC_X86_REG_R8,  UC_X86_REG_R9,  UC_X86_REG_R10, UC_X86_REG_R11,
    UC_X86_REG_R12, UC_X86_REG_R13, UC_X86_REG_R14, UC_X86_REG_R15,
};

constexpr std::array<int, 16> kXmmRegIds = {
    UC_X86_REG_XMM0,  UC_X86_REG_XMM1,  UC_X86_REG_XMM2,  UC_X86_REG_XMM3,
    UC_X86_REG_XMM4,  UC_X86_REG_XMM5,  UC_X86_REG_XMM6,  UC_X86_REG_XMM7,
    UC_X86_REG_XMM8,  UC_X86_REG_XMM9,  UC_X86_REG_XMM10, UC_X86_REG_XMM11,
    UC_X86_REG_XMM12, UC_X86_REG_XMM13, UC_X86_REG_XMM14, UC_X86_REG_XMM15,
};

[[nodiscard]] Stop map_uc_err(uc_err e) noexcept {
  switch (e) {
    case UC_ERR_OK:
      return Stop::ok;
    case UC_ERR_INSN_INVALID:
      return Stop::ud;
    case UC_ERR_READ_UNMAPPED:
    case UC_ERR_WRITE_UNMAPPED:
    case UC_ERR_FETCH_UNMAPPED:
    case UC_ERR_READ_PROT:
    case UC_ERR_WRITE_PROT:
    case UC_ERR_FETCH_PROT:
      return Stop::pf;
    default:
      return Stop::other;
  }
}

// uc_emu_start only ever returns UC_ERR_EXCEPTION for a genuine CPU-raised exception (divide
// error, #UD via a decodable-but-architecturally-illegal encoding, #GP, an internal #PF distinct
// from the already-handled unmapped/protected-memory cases above) -- it carries no detail about
// WHICH exception fired. That detail only exists inside QEMU's own exception dispatch
// (cpu_handle_exception in accel/tcg/cpu-exec.c), which hands the exception vector to any
// registered UC_HOOK_INTR callback before falling back to UC_ERR_EXCEPTION -- without a hook
// registered, every one of those distinct exceptions collapses into the same generic "other"
// bucket. Found via a real seven-fuzzer run: seven's interpreter correctly reports stop=de for an
// IDIV overflow/divide-by-zero, but the unicorn lane reported stop=other, a false-positive
// divergence -- not a real behavioral difference, just this lane failing to decode its own
// engine's exception report. Vector numbers match QEMU's target/i386/cpu.h EXCP00_DIVZ=0/
// EXCP06_ILLOP=6/EXCP0D_GPF=13/EXCP0E_PAGE=14, which are the real x86 architectural exception
// vectors, not unicorn-specific values.
// Merely registering a UC_HOOK_INTR callback changes unicorn's own behavior on an unhandled CPU
// exception: cpu_handle_exception (accel/tcg/cpu-exec.c) treats the exception as "caught" the
// moment ANY UC_HOOK_INTR hook exists, regardless of what the hook does, and resumes emulation
// instead of stopping with UC_ERR_EXCEPTION -- confirmed empirically (a first version of this hook
// that only recorded intno turned every "stop=de" case into a false "stop=ok", strictly worse than
// the undifferentiated "other" this replaces). uc_emu_stop() forces the run to actually halt right
// here, matching what would have happened with no hook registered at all, just with intno now
// captured on the way out.
void hook_intr(uc_engine* uc, std::uint32_t intno, void* user_data) {
  *static_cast<std::uint32_t*>(user_data) = intno;
  uc_emu_stop(uc);
}

[[nodiscard]] Stop map_intno(std::uint32_t intno) noexcept {
  switch (intno) {
    case 0:
      return Stop::de;
    case 6:
      return Stop::ud;
    case 13:
      return Stop::gp;
    case 14:
      return Stop::pf;
    default:
      return Stop::other;
  }
}

}  // namespace

LaneOutcome run_unicorn(const TestCase& tc) {
  LaneOutcome out;
  uc_engine* uc = nullptr;

  auto fail = [&](const char* what, uc_err e) -> LaneOutcome {
    out.setup_ok = false;
    out.setup_error = std::string("unicorn: ") + what + " failed: " + uc_strerror(e);
    if (uc != nullptr) uc_close(uc);
    return out;
  };

  uc_err err = uc_open(UC_ARCH_X86, UC_MODE_64, &uc);
  if (err != UC_ERR_OK) return fail("uc_open", err);

  if ((err = uc_mem_map(uc, kCodeBase, kPageSize, UC_PROT_ALL)) != UC_ERR_OK) return fail("map code", err);
  if ((err = uc_mem_map(uc, kDataBase, kPageSize, UC_PROT_ALL)) != UC_ERR_OK) return fail("map data", err);
  if ((err = uc_mem_map(uc, kStackBase, kPageSize, UC_PROT_ALL)) != UC_ERR_OK) return fail("map stack", err);

  {
    const std::uint64_t retaddr = kCodeBase;
    uc_mem_write(uc, kStackTop, &retaddr, sizeof(retaddr));  // see gen_jcc's comment
  }
  uc_mem_write(uc, kDataBase, tc.data_seed.data(), tc.data_seed.size());

  if (!tc.bytes.empty()) {
    err = uc_mem_write(uc, kCodeBase, tc.bytes.data(), tc.bytes.size());
    if (err != UC_ERR_OK) return fail("write code", err);
  }

  for (int i = 0; i < 16; ++i) {
    std::uint64_t v = tc.initial.gpr[static_cast<std::size_t>(i)];
    if ((err = uc_reg_write(uc, kGprRegIds[static_cast<std::size_t>(i)], &v)) != UC_ERR_OK) {
      return fail("gpr write", err);
    }
  }
  {
    std::uint64_t rip = kCodeBase;
    if ((err = uc_reg_write(uc, UC_X86_REG_RIP, &rip)) != UC_ERR_OK) return fail("rip write", err);
    std::uint64_t fl = tc.initial.rflags;
    if ((err = uc_reg_write(uc, UC_X86_REG_EFLAGS, &fl)) != UC_ERR_OK) return fail("eflags write", err);
  }
  for (int i = 0; i < 16; ++i) {
    std::uint64_t v[2] = {tc.initial.xmm_lo[static_cast<std::size_t>(i)], tc.initial.xmm_hi[static_cast<std::size_t>(i)]};
    if ((err = uc_reg_write(uc, kXmmRegIds[static_cast<std::size_t>(i)], v)) != UC_ERR_OK) {
      return fail("xmm write", err);
    }
  }

  // Write-only: this lane sets the x87 state so an x87 test case's GPR and memory results (FNSTSW
  // AX, FSTP m64, ...) stay comparable, but LaneOutcome::captures_x87 stays false and the register
  // file is never read back. See that field's comment. FPSW goes first because Unicorn derives its
  // TOP from it, and UC_X86_REG_ST0..ST7 are stack-relative (UC_X86_REG_FP0..FP7 are the physical
  // ones), so the order below matters. FPTAG takes the full architectural word, same form X87State
  // carries.
  {
    std::uint16_t sw = tc.initial.x87.status_word;
    if ((err = uc_reg_write(uc, UC_X86_REG_FPSW, &sw)) != UC_ERR_OK) return fail("fpsw write", err);
    std::uint16_t cw = tc.initial.x87.control_word;
    if ((err = uc_reg_write(uc, UC_X86_REG_FPCW, &cw)) != UC_ERR_OK) return fail("fpcw write", err);
    for (int i = 0; i < 8; ++i) {
      std::uint8_t raw[10] = {};
      std::memcpy(raw, &tc.initial.x87.signif[static_cast<std::size_t>(i)], 8);
      std::memcpy(raw + 8, &tc.initial.x87.signexp[static_cast<std::size_t>(i)], 2);
      if ((err = uc_reg_write(uc, UC_X86_REG_ST0 + i, raw)) != UC_ERR_OK) return fail("st write", err);
    }
    std::uint16_t tw = tc.initial.x87.tag_word;
    if ((err = uc_reg_write(uc, UC_X86_REG_FPTAG, &tw)) != UC_ERR_OK) return fail("fptag write", err);
  }

  // See hook_intr's comment: without this, every CPU-raised exception (divide error, #GP, ...)
  // collapses into the same undifferentiated UC_ERR_EXCEPTION/Stop::other bucket, producing
  // false-positive divergences against seven's interpreter reporting the correct specific stop
  // reason for the exact same fault.
  std::uint32_t intno = 0xFFFFFFFFu;
  uc_hook intr_hook{};
  uc_hook_add(uc, &intr_hook, UC_HOOK_INTR, reinterpret_cast<void*>(&hook_intr), &intno, 1, 0);

  // count=1 stops after exactly one instruction; `until` is unreachable so
  // count is what actually bounds it (standard unicorn single-step idiom).
  err = uc_emu_start(uc, kCodeBase, 0xFFFFFFFFFFFFFFFFull, 0, 1);
  // intno being set at all is proof positive a CPU exception fired during this step (hook_intr's
  // uc_emu_stop() forces uc_emu_start to return before the count=1 budget would otherwise report
  // UC_ERR_OK) -- that takes priority over err itself, which the hook's mere presence already
  // steers away from UC_ERR_EXCEPTION.
  if (intno != 0xFFFFFFFFu) {
    out.stop = map_intno(intno);
  } else {
    out.stop = map_uc_err(err);
  }
  if (out.stop == Stop::other) {
    out.detail = std::string("unicorn uc_err=") + uc_strerror(err);
  }

  for (int i = 0; i < 16; ++i) {
    std::uint64_t v = 0;
    (void)uc_reg_read(uc, kGprRegIds[static_cast<std::size_t>(i)], &v);
    out.after.gpr[static_cast<std::size_t>(i)] = v;
  }
  {
    std::uint64_t rip = 0, fl = 0;
    (void)uc_reg_read(uc, UC_X86_REG_RIP, &rip);
    (void)uc_reg_read(uc, UC_X86_REG_EFLAGS, &fl);
    out.after.rip = rip;
    out.after.rflags = fl;
  }
  for (int i = 0; i < 16; ++i) {
    std::uint64_t v[2] = {0, 0};
    (void)uc_reg_read(uc, kXmmRegIds[static_cast<std::size_t>(i)], v);
    out.after.xmm_lo[static_cast<std::size_t>(i)] = v[0];
    out.after.xmm_hi[static_cast<std::size_t>(i)] = v[1];
  }
  (void)uc_mem_read(uc, kDataBase, out.after.data_after.data(), kDataWindow);

  uc_close(uc);
  return out;
}

}  // namespace sf
