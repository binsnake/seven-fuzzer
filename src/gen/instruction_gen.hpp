#pragma once

// Generates random-but-valid x86-64 instructions by building
// iced_x86::Instruction objects through the same encoder seven_core's own
// decoder round-trips against, so a mismatch downstream is a real execution-
// semantics difference, never a generator encoding bug.
//
// v1 scope, deliberately: GPR ALU/shift/data-movement/control-flow, plus
// legacy-SSE (non-VEX/EVEX) XMM shuffle/logic/pack/shift/fp instructions.
// Excluded for now (see project notes for why): SYSCALL/SYSENTER/INT-n
// (hardware-lane safety — see lanes/hardware_lane.hpp), privileged
// instructions, x87/MMX, and VEX/EVEX-encoded SIMD (no host AVX/AVX-512
// capability detection yet — see the comment above gen_simd_shuffle). All
// memory operands are [rdi+disp8] with a small bounded displacement, and
// RDI/RSP are pinned to the harness's fixed scratch addresses rather than
// randomized, so every generated instruction's memory/stack access stays
// inside the three mapped pages every lane maps identically.

#include <cstdint>
#include <random>

#include "common/types.hpp"

namespace sf {

class InstructionGenerator {
 public:
  explicit InstructionGenerator(std::uint64_t seed) : rng_(seed) {}

  [[nodiscard]] TestCase next();

 private:
  std::mt19937_64 rng_;
};

}  // namespace sf
