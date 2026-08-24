#include "lanes/unicorn_lane.hpp"

#include <array>
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

  // count=1 stops after exactly one instruction; `until` is unreachable so
  // count is what actually bounds it (standard unicorn single-step idiom).
  err = uc_emu_start(uc, kCodeBase, 0xFFFFFFFFFFFFFFFFull, 0, 1);
  out.stop = map_uc_err(err);
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
