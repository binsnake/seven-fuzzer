#include "lanes/seven_jit_lane.hpp"

#include <array>
#include <cstdint>
#include <cstring>

#include "seven/compat.hpp"
#include "seven_jit/jit_executor.hpp"

namespace sf {

namespace {

// Same packing idiom as seven_lane.cpp -- duplicated rather than shared, matching every other lane
// in this project mapping its own representation onto sf::RegState independently (see
// common/types.hpp's top comment).
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

[[nodiscard]] Stop map_stop_reason(seven::StopReason r) noexcept {
  using SR = seven::StopReason;
  switch (r) {
    case SR::none:
      // JitExecutor::run()'s contract differs from Executor::step()'s here: run() reports
      // execution_limit whenever it retires the full requested budget with no other stop
      // condition, rather than `none`. run_seven_jit always asks for exactly 1 instruction, so
      // execution_limit is precisely "that one instruction executed fine" -- the JIT lane's
      // equivalent of step()'s `none`, not a genuine "other" stop.
    case SR::execution_limit:
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

LaneOutcome run_seven_jit(const TestCase& tc, seven_jit::JitExecutor& jit_executor, seven::Memory& memory) {
  LaneOutcome out;

  seven::CpuState state{};
  state.mode = seven::ExecutionMode::long64;
  state.rip = kCodeBase;
  state.rflags = tc.initial.rflags;
  state.gpr = tc.initial.gpr;
  for (int i = 0; i < 16; ++i) {
    state.vectors[static_cast<std::size_t>(i)].value =
        pack_xmm(tc.initial.xmm_lo[static_cast<std::size_t>(i)], tc.initial.xmm_hi[static_cast<std::size_t>(i)]);
  }

  // unmap() before every map() -- see seven_jit_lane.hpp's comment: memory is reused across many
  // TestCases (paired with the also-reused jit_executor), so this is what makes every call start
  // from genuinely fresh, zeroed pages with freshly-bumped code epochs, matching what a brand new
  // Memory object would look like, rather than leaking either stale byte content or a stale-but-
  // epoch-coincidentally-fresh-looking JIT cache entry from an earlier, unrelated TestCase.
  memory.unmap(kCodeBase, kPageSize);
  memory.unmap(kDataBase, kPageSize);
  memory.unmap(kStackBase, kPageSize);
  memory.map(kCodeBase, kPageSize);
  memory.map(kDataBase, kPageSize);
  memory.map(kStackBase, kPageSize);

  {
    const std::uint64_t retaddr = kCodeBase;
    (void)memory.write(kStackTop, &retaddr, sizeof(retaddr));
  }
  (void)memory.write(kDataBase, tc.data_seed.data(), tc.data_seed.size());

  if (!tc.bytes.empty() && !memory.write(kCodeBase, tc.bytes.data(), tc.bytes.size())) {
    out.setup_ok = false;
    out.setup_error = "seven-jit: failed to write code bytes";
    return out;
  }

  const seven::ExecutionResult res = jit_executor.run(state, memory, 1);

  out.stop = map_stop_reason(res.reason);
  if (out.stop == Stop::other) {
    out.detail = "seven-jit StopReason=" + std::to_string(static_cast<int>(res.reason));
  }
  out.after.gpr = state.gpr;
  out.after.rip = state.rip;
  out.after.rflags = state.rflags;
  for (int i = 0; i < 16; ++i) {
    unpack_xmm(state.vectors[static_cast<std::size_t>(i)].value, out.after.xmm_lo[static_cast<std::size_t>(i)],
               out.after.xmm_hi[static_cast<std::size_t>(i)]);
  }
  (void)memory.read(kDataBase, out.after.data_after.data(), kDataWindow);

  return out;
}

}  // namespace sf
