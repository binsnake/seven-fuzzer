#pragma once

#include <string>

namespace fuzz {

// What the CPU this fuzzer is running on can actually execute.
//
// Every lane except the hardware one is an emulator, and seven in particular is compiled with
// SEVEN_ENABLE_AVX512=1 unconditionally, so it will happily execute an EVEX instruction on a host
// whose silicon has no AVX-512 at all. The hardware lane would then #UD on every single case and
// the run would fill up with thousands of identical, uninteresting divergences that say nothing
// about seven's correctness. The generator asks these questions before emitting an encoding so the
// corpus stays within what the oracle can actually adjudicate.
//
// A feature is only reported present when both the CPU implements it and the OS has enabled the
// register state it needs (XCR0 via XGETBV). Windows does enable AVX and AVX-512 state, but a
// hypervisor or an old kernel can hide either, and executing VEX with YMM state disabled faults
// exactly the same way an unsupported CPU does.
struct HostCaps {
  bool avx = false;      // VEX encoding usable, YMM state enabled
  bool avx2 = false;     // integer VEX.256
  bool bmi1 = false;
  bool bmi2 = false;
  bool avx512f = false;  // EVEX usable, opmask + ZMM state enabled
  bool avx512vl = false; // EVEX at 128/256-bit widths
  bool avx512bw = false; // byte/word element EVEX
  bool avx512dq = false;

  [[nodiscard]] std::string summary() const;
};

// Probed once on first call.
[[nodiscard]] const HostCaps& host_caps();

}  // namespace fuzz
