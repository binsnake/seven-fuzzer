#pragma once

// Real-hardware lane: single-steps generated code on the actual CPU inside a
// disposable, sandboxed victim process that this session debugs (WinAPI
// debug-event loop, trap-flag single-stepping — the same technique any
// debugger uses; nothing here is a simulator).
//
// Safety model (see project notes for the full rationale):
//   - Victim runs under a restricted, low-integrity token derived from this
//     process's own token via CreateRestrictedToken (no elevation needed).
//   - Victim is confined to a job object: killed if this process dies,
//     capped active-process count of 1, capped memory, UI restrictions.
//   - The generator (gen/instruction_gen.*) never emits SYSCALL/SYSENTER/
//     INT-n(n!=3) — see its header comment for why that's a hard exclusion
//     rather than something this lane could sandbox its way around.
//   - Any debug event other than the expected single-step trap (a real
//     fault, an unexpected process exit, a timeout) causes the victim to be
//     killed and respawned fresh before the next test case runs.

#include <memory>

#include "common/types.hpp"

namespace sf {

class HardwareSession {
 public:
  HardwareSession();
  ~HardwareSession();
  HardwareSession(const HardwareSession&) = delete;
  HardwareSession& operator=(const HardwareSession&) = delete;

  // victim_exe_path: full path to seven_fuzz_victim.exe (built alongside
  // this binary). Returns false with a message on unrecoverable setup
  // failure (e.g. can't create the restricted token or spawn the process).
  [[nodiscard]] bool start(const std::wstring& victim_exe_path, std::string& error);

  // Runs exactly one instruction on real hardware from tc.initial state.
  // Transparently respawns the victim if the previous test case left it in
  // a state that isn't safe to keep reusing.
  [[nodiscard]] LaneOutcome run_one(const TestCase& tc);

  void shutdown();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sf
