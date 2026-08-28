# seven-fuzzer

A differential ISA-level fuzzer for [seven-emulator](https://github.com/binsnake/seven-emulator). It
generates random-but-valid x86-64 instructions and single-steps each one through three independent
execution engines -- `seven_core` itself, [Unicorn](https://github.com/unicorn-engine/unicorn) (built
from [sogen](https://github.com/momo5502/sogen)'s vendored submodule, so it matches what sogen's
default backend actually runs), and real hardware (a sandboxed debuggee single-stepped via the trap
flag -- genuine CPU execution, not a third simulator) -- then diffs the resulting registers, flags, and
memory. Real hardware is the ground truth; agreement with Unicorn alone is not the bar.

## Why

Every correctness gap between `seven_core` and real silicon is a place where code that runs correctly
on hardware could behave differently under emulation. The fuzzer's job is to find those gaps
systematically, at the single-instruction level, with real hardware as the oracle rather than another
piece of software.

## Architecture

Three lanes behind one shared contract (`src/common/types.hpp`): a `TestCase` in (encoded instruction
bytes, initial register/XMM/flags/memory state), a `LaneOutcome` out (resulting state plus a
normalized stop reason -- `ok`, `ud`, `gp`, `pf`, `de`, `bp`, `halted`, `other` -- that every lane maps
its own fault vocabulary onto).

- `src/lanes/seven_lane.*` -- runs the instruction through a standalone `seven_core::Executor`.
- `src/lanes/unicorn_lane.*` -- runs it through Unicorn in `UC_MODE_64`.
- `src/lanes/hardware_lane.*` -- runs it on real hardware: a sandboxed, restricted-token, job-object-
  confined victim process, single-stepped via the trap flag and Windows' debug-event API.
- `src/gen/instruction_gen.*` -- generates the test cases via `iced_x86::InstructionFactory` +
  `Encoder`, the same encoder/decoder `seven_core` links, so a mismatch downstream is never a
  generator-encoding artifact.
- `src/report/report.*` -- writes a full repro (bytes, initial state, all three lanes' outcomes, and
  which fields diverged) for every case where the lanes disagree.

## Scope

GPR ALU / shift / rotate / data-movement / control-flow / bit-manipulation / MUL-IMUL-DIV-IDIV, plus
legacy-SSE (non-VEX/EVEX) XMM instructions -- shuffle/permute, bitwise logic, pack/unpack, integer
shift, and FP horizontal/compare ops. Not yet covered: x87/MMX, VEX/EVEX-encoded SIMD (no host
AVX/AVX-512 capability detection yet), and privileged/ring-0 instructions (implemented in the
generator but not wired in -- every one of them unconditionally executes rather than faulting in ring 3
in both `seven_core` and Unicorn, which is a real architectural gap rather than scattered bugs, and not
something to fix by guessing).

All generated memory operands are `[rdi+disp8]` with a small bounded displacement; RDI/RSP are pinned
to the harness's fixed scratch addresses rather than randomized, so every generated instruction's
memory/stack access stays inside pages every lane maps identically.

## Building

Requires a sibling checkout of `seven-emulator` and `sogen` (CMake, Ninja, MSVC on Windows -- the
hardware lane is Windows-only, it uses the Win32 debug-event API):

```
seven-fuzzer/
seven-emulator/
sogen/
```

If your layout differs, override the paths at configure time:

```
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DSEVEN_ROOT=<path> -DSOGEN_ROOT=<path>
cmake --build build -j
```

## Running

```
seven_fuzzer.exe [iterations] [seed] [--threads N] [--no-hw] [--no-unicorn] [--no-seven] [--verbose-from N]
```

Divergences are written to `findings/`; cases where only Unicorn disagrees with hardware (not a
`seven_core` bug, but real data on Unicorn's own fidelity gap) are written separately to
`unicorn_outliers/`. Neither directory is checked in -- they're regenerated per run.

## Status

Multithreaded (~6.4x speedup at 14 threads on a 16-core box -- the hardware lane's real debug-event
round trips are the actual bottleneck; `seven_core`/Unicorn calls are microseconds each). Several real
`seven_core` bugs found and fixed this way so far, each hardware-confirmed before being merged upstream
-- see [seven-emulator](https://github.com/binsnake/seven-emulator)'s commit history for the fixes this
project has produced.
