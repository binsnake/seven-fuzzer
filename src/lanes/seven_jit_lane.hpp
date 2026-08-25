#pragma once

#include "common/types.hpp"

namespace seven_jit {
class JitExecutor;
}  // namespace seven_jit

namespace seven {
class Memory;
}  // namespace seven

namespace sf {

// Runs one TestCase through seven-jit's JitExecutor instead of seven_core's plain interpreter --
// the same differential contract as run_seven, just a different engine on the "seven" side. Every
// TestCase is a single instruction, so this always calls jit_executor.run(state, memory, 1); the
// max_instructions=1 cap is what keeps this safe even when JitExecutor decides to compile a
// multi-instruction block starting at kCodeBase (e.g. it decoded past the one real instruction into
// whatever garbage bytes happen to follow it on the page) -- JitExecutor::run() falls back to a
// single interpreter step whenever a compiled block would cover more instructions than the caller
// asked for, rather than ever overrunning the requested count.
//
// jit_executor AND memory are both caller-owned and meant to be reused together across many calls
// (one pair per worker thread, same lifetime as that worker's InstructionGenerator/HardwareSession)
// -- constructing a fresh JitExecutor per TestCase would mean reinitializing asmjit's JIT runtime on
// every single fuzzer iteration, far too expensive for a tight fuzzing loop.
//
// The two must be reused TOGETHER, not independently: JitExecutor's cache staleness check
// (cache_entry_stale) compares a cached entry's recorded page epoch against memory.page_code_epoch()
// on THAT SAME Memory object. Pairing a persistent jit_executor with a freshly-constructed Memory
// every call would silently defeat that check -- a brand new Memory's page epochs start over from
// scratch, so a stale cache entry from a completely different, earlier TestCase's instruction could
// read back as "not stale" purely by epoch-value coincidence and get replayed against the wrong
// bytes. run_seven_jit re-maps (via unmap()+map(), both of which unconditionally bump the affected
// pages' code_epoch) kCodeBase/kDataBase/kStackBase at the start of every call specifically so this
// can never happen: every call's compiled/cached state is provably fresh relative to the SAME memory
// object's own monotonic epoch counter, and every page's prior contents are actually cleared too
// (matching what a brand new Memory object would look like), not just epoch-invalidated.
[[nodiscard]] LaneOutcome run_seven_jit(const TestCase& tc, seven_jit::JitExecutor& jit_executor,
                                         seven::Memory& memory);

}  // namespace sf
