#include "common/host_caps.hpp"

#include <array>
#include <cstdint>

#if defined(_MSC_VER)
#include <immintrin.h>
#include <intrin.h>
#endif

namespace fuzz {
namespace {

struct CpuidRegs {
  std::uint32_t eax = 0, ebx = 0, ecx = 0, edx = 0;
};

[[nodiscard]] CpuidRegs cpuid(std::uint32_t leaf, std::uint32_t subleaf) {
  CpuidRegs r{};
#if defined(_MSC_VER)
  std::array<int, 4> regs{};
  __cpuidex(regs.data(), static_cast<int>(leaf), static_cast<int>(subleaf));
  r.eax = static_cast<std::uint32_t>(regs[0]);
  r.ebx = static_cast<std::uint32_t>(regs[1]);
  r.ecx = static_cast<std::uint32_t>(regs[2]);
  r.edx = static_cast<std::uint32_t>(regs[3]);
#else
  (void)leaf;
  (void)subleaf;
#endif
  return r;
}

[[nodiscard]] bool bit(std::uint32_t value, int index) {
  return ((value >> index) & 1u) != 0;
}

[[nodiscard]] HostCaps probe() {
  HostCaps caps{};
#if defined(_MSC_VER)
  const CpuidRegs leaf0 = cpuid(0, 0);
  if (leaf0.eax < 1) return caps;

  const CpuidRegs leaf1 = cpuid(1, 0);
  const bool osxsave = bit(leaf1.ecx, 27);
  const bool cpu_avx = bit(leaf1.ecx, 28);

  // Without OSXSAVE the XGETBV below is itself a #UD, so nothing wider than SSE can be trusted.
  if (!osxsave) return caps;

  const std::uint64_t xcr0 = _xgetbv(0);
  const bool ymm_state = (xcr0 & 0x6u) == 0x6u;          // SSE + YMM_Hi128
  const bool zmm_state = (xcr0 & 0xE6u) == 0xE6u;        // + opmask, ZMM_Hi256, Hi16_ZMM

  caps.avx = cpu_avx && ymm_state;

  if (leaf0.eax >= 7) {
    const CpuidRegs leaf7 = cpuid(7, 0);
    caps.bmi1 = bit(leaf7.ebx, 3);
    caps.bmi2 = bit(leaf7.ebx, 8);
    caps.avx2 = caps.avx && bit(leaf7.ebx, 5);
    if (zmm_state) {
      caps.avx512f = bit(leaf7.ebx, 16);
      caps.avx512dq = caps.avx512f && bit(leaf7.ebx, 17);
      caps.avx512bw = caps.avx512f && bit(leaf7.ebx, 30);
      caps.avx512vl = caps.avx512f && bit(leaf7.ebx, 31);
    }
  }
#endif
  return caps;
}

}  // namespace

std::string HostCaps::summary() const {
  std::string out;
  const auto add = [&out](const char* name, bool present) {
    if (!present) return;
    if (!out.empty()) out += ' ';
    out += name;
  };
  add("avx", avx);
  add("avx2", avx2);
  add("bmi1", bmi1);
  add("bmi2", bmi2);
  add("avx512f", avx512f);
  add("avx512vl", avx512vl);
  add("avx512bw", avx512bw);
  add("avx512dq", avx512dq);
  if (out.empty()) out = "sse only";
  return out;
}

const HostCaps& host_caps() {
  static const HostCaps caps = probe();
  return caps;
}

}  // namespace fuzz
