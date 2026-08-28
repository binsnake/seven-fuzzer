// Differential fuzzer: generates random x86-64 instructions and single-steps
// each one through three independent execution engines — seven_core, Unicorn
// (built from sogen's vendored submodule), and real hardware (a sandboxed
// debuggee this process single-steps via the trap flag) — then diffs the
// resulting architectural state. See project notes for the full design and
// the safety rationale behind the hardware lane.
//
// Multithreaded: each worker owns an independent InstructionGenerator (seed
// mixed per worker), seven/unicorn calls are pure per-call computation with
// no shared state, and each worker gets its own HardwareSession -- a fully
// self-contained sandboxed victim process (own job object, own restricted
// token, own address space), so there's nothing to share or contend on
// between workers there either. The hardware lane's real debug-event round
// trips per instruction are the actual bottleneck; this is what makes
// parallelizing it worthwhile.
//
// Usage: seven_fuzzer.exe [iterations] [seed] [--no-hw] [--jit] [--threads N] [--verbose-from N]
//                          [--watchdog-seconds N] [--sweep-seeds N]
//
// --jit runs the "seven" side through seven-jit's JitExecutor instead of seven_core's plain
// interpreter (see lanes/seven_jit_lane.hpp) -- everything else about the run (TestCase generation,
// the Unicorn/hardware oracles, the comparator, reporting) is unchanged. This is a single check on
// which engine is under test, not a fourth parallel lane: run once with --jit, once without, to
// verify both independently against the exact same two oracles.
//
// --sweep-seeds N: instead of one run, relaunches this same exe as N fresh CHILD PROCESSES, one per
// derived seed, each with the other flags passed through unchanged (see run_seed_sweep). A fresh
// process per seed matters specifically for --jit: that lane can crash or hang from what looks like
// shared-heap corruption (see the project notes), and a corrupted heap doesn't reset between
// iterations within one process the way a fresh child does. This is how to get real fuzzing coverage
// out of --jit today despite that -- a single long --jit run on one seed will very likely die within
// its first few hundred iterations and never get anywhere near real depth.

// NOMINMAX: seven_jit/jit_executor.hpp transitively pulls in seven_core's float80.hpp, which
// defines std::numeric_limits<Float80>::min()/max() -- windows.h's own min/max macros silently
// mangle those into broken syntax if left defined, so this has to come before windows.h itself, not
// just before the seven_jit include below.
#define NOMINMAX
#include <windows.h>

#include <dbghelp.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <eh.h>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "common/host_caps.hpp"
#include "common/types.hpp"
#include "gen/instruction_gen.hpp"
#include "lanes/hardware_lane.hpp"
#include "lanes/seven_jit_lane.hpp"
#include "lanes/seven_lane.hpp"
#include "lanes/unicorn_lane.hpp"
#include "report/report.hpp"
#include "seven_jit/jit_executor.hpp"

#pragma comment(lib, "dbghelp.lib")

using namespace sf;

namespace {

// Last-resort, process-wide: fires only if an exception makes it all the
// way up without being caught (e.g. on a thread/in a context the per-worker
// try/catch in run_worker doesn't cover). Writes a real minidump before the
// process dies so a crash can actually be diagnosed with a debugger instead
// of just a bare exit code.
LONG WINAPI write_crash_dump(EXCEPTION_POINTERS* ep) {
  wchar_t path[MAX_PATH];
  const DWORD n = GetModuleFileNameW(nullptr, path, MAX_PATH);
  std::wstring dir(path, n);
  const auto pos = dir.find_last_of(L"\\/");
  if (pos != std::wstring::npos) dir.resize(pos);
  const std::wstring dump_path = dir + L"\\crash.dmp";

  HANDLE file = CreateFileW(dump_path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file != INVALID_HANDLE_VALUE) {
    MINIDUMP_EXCEPTION_INFORMATION mei{};
    mei.ThreadId = GetCurrentThreadId();
    mei.ExceptionPointers = ep;
    mei.ClientPointers = FALSE;
    // This has been observed to fail silently (0-byte dump, no diagnostic) when the crash is severe
    // enough that the process's own heap/CRT state is too corrupted for MiniDumpWriteDump to walk --
    // logging the failure here at least tells the next person that happened, instead of leaving a
    // mysteriously-empty dump file with no explanation.
    const BOOL wrote = MiniDumpWriteDump(
        GetCurrentProcess(), GetCurrentProcessId(), file,
        static_cast<MINIDUMP_TYPE>(MiniDumpWithFullMemory | MiniDumpWithHandleData | MiniDumpWithThreadInfo), &mei,
        nullptr, nullptr);
    const DWORD dump_err = wrote ? 0 : GetLastError();
    CloseHandle(file);
    if (wrote) {
      std::fwprintf(stderr, L"[fatal] unhandled exception 0x%08lX at %p -- wrote %s\n",
                    ep->ExceptionRecord->ExceptionCode, ep->ExceptionRecord->ExceptionAddress, dump_path.c_str());
    } else {
      std::fwprintf(stderr,
                    L"[fatal] unhandled exception 0x%08lX at %p -- MiniDumpWriteDump FAILED (err=%lu), %s is "
                    L"empty/unusable -- process state was likely too corrupted to dump\n",
                    ep->ExceptionRecord->ExceptionCode, ep->ExceptionRecord->ExceptionAddress, dump_err,
                    dump_path.c_str());
    }
  }
  return EXCEPTION_EXECUTE_HANDLER;  // terminate after dumping
}

// Called from the watchdog thread when a worker stops making progress -- unlike
// write_crash_dump above, there's no exception to hang the dump off of, so this
// snapshots every thread's current stack as-is. That's exactly what's needed for
// the known single-threaded --jit hang (see the --jit thread-clamp comment in
// main()): it lets the actual stuck call stack be inspected after the fact,
// instead of needing cdb attached live at the moment it happens.
void write_hang_dump() {
  wchar_t path[MAX_PATH];
  const DWORD n = GetModuleFileNameW(nullptr, path, MAX_PATH);
  std::wstring dir(path, n);
  const auto pos = dir.find_last_of(L"\\/");
  if (pos != std::wstring::npos) dir.resize(pos);
  const std::wstring dump_path = dir + L"\\watchdog_hang.dmp";

  HANDLE file = CreateFileW(dump_path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file != INVALID_HANDLE_VALUE) {
    const BOOL wrote = MiniDumpWriteDump(
        GetCurrentProcess(), GetCurrentProcessId(), file,
        static_cast<MINIDUMP_TYPE>(MiniDumpWithFullMemory | MiniDumpWithHandleData | MiniDumpWithThreadInfo),
        nullptr, nullptr, nullptr);
    const DWORD dump_err = wrote ? 0 : GetLastError();
    CloseHandle(file);
    if (wrote) {
      std::fwprintf(stderr, L"[watchdog] wrote %s\n", dump_path.c_str());
    } else {
      std::fwprintf(stderr, L"[watchdog] MiniDumpWriteDump FAILED (err=%lu), %s is empty/unusable\n", dump_err,
                    dump_path.c_str());
    }
  }
}

std::string exe_dir() {
  char buf[MAX_PATH];
  const DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
  std::string p(buf, n);
  const auto pos = p.find_last_of("\\/");
  return pos == std::string::npos ? "." : p.substr(0, pos);
}

// Diagnostic-only bisection switches, set once before any worker thread
// starts, only ever read afterward -- not a race.
bool g_no_unicorn = false;
bool g_no_seven = false;

std::wstring exe_dir_w() {
  wchar_t buf[MAX_PATH];
  const DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
  std::wstring p(buf, n);
  const auto pos = p.find_last_of(L"\\/");
  return pos == std::wstring::npos ? L"." : p.substr(0, pos);
}

std::uint64_t mix_seed(std::uint64_t base, int worker_id) {
  // splitmix64-style spread so nearby worker ids don't produce correlated
  // mt19937_64 streams.
  std::uint64_t z = base + static_cast<std::uint64_t>(worker_id + 1) * 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

void diff_pair(const char* label, const LaneOutcome& a, const LaneOutcome& b, std::uint64_t flags_mask,
               std::uint32_t gpr_compare_mask, std::vector<Divergence>& out) {
  if (!a.setup_ok || !b.setup_ok) return;  // harness hiccup, not a finding
  Divergence d;
  d.pairing = label;
  char buf[160];

  if (a.stop != b.stop) {
    std::snprintf(buf, sizeof(buf), "stop: %s vs %s", stop_name(a.stop), stop_name(b.stop));
    d.lines.emplace_back(buf);
  }
  if (a.stop == Stop::ok && b.stop == Stop::ok) {
    static const char* kNames[16] = {"rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
                                      "r8",  "r9",  "r10", "r11", "r12", "r13", "r14", "r15"};
    for (int i = 0; i < 16; ++i) {
      if (((gpr_compare_mask >> i) & 1u) == 0) continue;  // e.g. BSF/BSR-zero-source, see TestCase::gpr_compare_mask
      if (a.after.gpr[static_cast<std::size_t>(i)] != b.after.gpr[static_cast<std::size_t>(i)]) {
        std::snprintf(buf, sizeof(buf), "%s: %016llX vs %016llX", kNames[i],
                      static_cast<unsigned long long>(a.after.gpr[static_cast<std::size_t>(i)]),
                      static_cast<unsigned long long>(b.after.gpr[static_cast<std::size_t>(i)]));
        d.lines.emplace_back(buf);
      }
    }
    if (a.after.rip != b.after.rip) {
      std::snprintf(buf, sizeof(buf), "rip: %016llX vs %016llX", static_cast<unsigned long long>(a.after.rip),
                    static_cast<unsigned long long>(b.after.rip));
      d.lines.emplace_back(buf);
    }
    const std::uint64_t fa = a.after.rflags & flags_mask;
    const std::uint64_t fb = b.after.rflags & flags_mask;
    if (fa != fb) {
      std::snprintf(buf, sizeof(buf), "flags: %03llX vs %03llX", static_cast<unsigned long long>(fa),
                    static_cast<unsigned long long>(fb));
      d.lines.emplace_back(buf);
    }
    if (a.after.data_after != b.after.data_after) d.lines.emplace_back("data window differs");
    if (a.captures_x87 && b.captures_x87) {
      const X87State& xa = a.after.x87;
      const X87State& xb = b.after.x87;
      if (xa.control_word != xb.control_word) {
        std::snprintf(buf, sizeof(buf), "x87 cw: %04X vs %04X", xa.control_word, xb.control_word);
        d.lines.emplace_back(buf);
      }
      const std::uint16_t sa = xa.status_word & kX87StatusMask;
      const std::uint16_t sb = xb.status_word & kX87StatusMask;
      if (sa != sb) {
        // The differing bits are spelled out because triaging an x87 status word by eye is
        // otherwise a bit-picking exercise every single time -- a run where only bit 5 (PE) ever
        // differs is a very different conclusion from one where TOP does.
        std::snprintf(buf, sizeof(buf), "x87 sw: %04X vs %04X (differing bits %04X)", sa, sb,
                      static_cast<unsigned>(sa ^ sb));
        d.lines.emplace_back(buf);
      }
      if (xa.tag_word != xb.tag_word) {
        std::snprintf(buf, sizeof(buf), "x87 tw: %04X vs %04X", xa.tag_word, xb.tag_word);
        d.lines.emplace_back(buf);
      }
      for (int i = 0; i < 8; ++i) {
        // An empty x87 register's contents are architecturally undefined -- a pop marks the slot
        // free without specifying what is left behind in it -- so only compare slots both engines
        // agree are live. Same principle as gpr_compare_mask for BSF/BSR's zero-source destination.
        if (x87_tag_of_st(xa, i) == 0x3 || x87_tag_of_st(xb, i) == 0x3) continue;
        const auto ii = static_cast<std::size_t>(i);
        if (xa.signexp[ii] != xb.signexp[ii] || xa.signif[ii] != xb.signif[ii]) {
          std::snprintf(buf, sizeof(buf), "st%d: %04X:%016llX vs %04X:%016llX", i, xa.signexp[ii],
                        static_cast<unsigned long long>(xa.signif[ii]), xb.signexp[ii],
                        static_cast<unsigned long long>(xb.signif[ii]));
          d.lines.emplace_back(buf);
        }
      }
    }
  }
  if (!d.lines.empty()) out.push_back(std::move(d));
}

// Converts structured exceptions (access violations, etc.) into catchable
// C++ exceptions -- see the /EHa note in CMakeLists.txt. _set_se_translator
// is per-thread in the MSVC CRT, so each worker installs its own.
struct SehException {
  unsigned int code;
  void* address;
};

void seh_translator(unsigned int code, EXCEPTION_POINTERS* ep) {
  void* addr = (ep != nullptr && ep->ExceptionRecord != nullptr) ? ep->ExceptionRecord->ExceptionAddress : nullptr;
  throw SehException{code, addr};
}

// How many dump files each of findings/ and unicorn_outliers/ may hold before the rest are counted
// but not written. A run against a genuinely broken area produces divergences by the thousand -- the
// x87 families alone found ~2,300 in 30k iterations -- and one text file each filled a 954GB disk to
// zero, which takes the whole machine down, not just the run. The counts printed at the end stay
// exact either way, and a few hundred dumps is already far more than anyone reads before fixing the
// first one and re-running.
constexpr int kMaxDumpFilesPerDir = 250;

struct SharedCounters {
  std::atomic<int> finding_count{0};
  std::atomic<int> unicorn_outlier_saved{0};  // index counter for the saved-outlier files, separate from the count below
  std::atomic<std::uint64_t> unicorn_outlier_count{0};
  std::atomic<std::uint64_t> hw_harness_errors{0};
  // Candidates the generator built and then threw away because the encoder refused them. Worth
  // surfacing: next() just retries until something encodes, so a family whose instructions are all
  // being rejected looks exactly like one that is working. That hid a bug where nearly every
  // immediate form was rejected and those codes never reached any lane at all.
  std::atomic<std::uint64_t> generator_discards{0};
  std::atomic<std::uint64_t> total_done{0};
  std::mutex io_mutex;
  // Bisection (seven-alone and unicorn-alone each survive sustained heavy
  // concurrent load indefinitely; seven+unicorn together crash within
  // seconds every time) points to a cross-library interaction -- likely
  // something in the shared CRT/heap rather than project-specific static
  // state (already audited clean for seven_core/iced_x86). Root cause not
  // pinned down further; both calls are pure in-memory computation on the
  // order of microseconds, while the hardware lane's real debug-event round
  // trips are milliseconds and the actual reason to parallelize at all, so
  // serializing just these two sidesteps the bug at negligible cost instead
  // of chasing it deeper.
  std::mutex compute_mutex;
};

// One entry per worker, updated from that worker's own thread every TestCase and read from the
// watchdog thread -- see run_watchdog below. local_i plus the worker's id is enough to reproduce
// deterministically (InstructionGenerator is fully seeded from mix_seed(seed, worker_id)), so the
// diagnostic doesn't need to carry the TestCase text itself and can stay lock-free in the hot loop.
struct WorkerHeartbeat {
  std::atomic<std::uint64_t> last_tick_ms{0};
  std::atomic<std::uint64_t> local_i{0};
  std::atomic<bool> finished{false};
};

// Last-resort safety net for the known single-threaded --jit hang (Unicorn's TCG-vs-asmjit
// allocator interaction, see the --jit thread-clamp comment in main()) and, more generally, for
// any future hang this fuzzer wasn't specifically designed around: an unattended overnight run
// should always terminate on its own rather than sit stuck forever with no one watching. Polls
// each worker's heartbeat; a worker that hasn't started a new TestCase within the timeout gets a
// live minidump (captures the actually-stuck call stack, no need to catch it under a debugger) and
// the whole process is torn down immediately -- there's no way to safely cancel just the one stuck
// thread out from under vendored Unicorn/asmjit state, and a hard kill with a clear exit code beats
// an unattended run hanging indefinitely.
void run_watchdog(std::vector<WorkerHeartbeat>& heartbeats, unsigned int watchdog_seconds, std::mutex& io_mutex) {
  if (watchdog_seconds == 0) return;  // --watchdog-seconds 0 disables it
  const std::uint64_t timeout_ms = static_cast<std::uint64_t>(watchdog_seconds) * 1000ull;
  for (;;) {
    std::this_thread::sleep_for(std::chrono::seconds(2));
    const std::uint64_t now = GetTickCount64();
    bool all_finished = true;
    for (std::size_t w = 0; w < heartbeats.size(); ++w) {
      if (heartbeats[w].finished.load(std::memory_order_relaxed)) continue;
      all_finished = false;
      const std::uint64_t last = heartbeats[w].last_tick_ms.load(std::memory_order_relaxed);
      if (now - last > timeout_ms) {
        {
          std::lock_guard<std::mutex> lock(io_mutex);
          std::printf(
              "[watchdog] worker %zu appears hung: no progress for over %u seconds, stuck at local "
              "iteration %llu. Writing a live minidump and terminating so an unattended run doesn't "
              "hang forever -- rerun with --threads 1 and the same seed to reach this worker's "
              "iteration range and reproduce.\n",
              w, watchdog_seconds, static_cast<unsigned long long>(heartbeats[w].local_i.load(std::memory_order_relaxed)));
          std::fflush(stdout);
        }
        write_hang_dump();
        TerminateProcess(GetCurrentProcess(), 2);
      }
    }
    if (all_finished) return;
  }
}

void run_worker(int worker_id, std::uint64_t iterations, std::uint64_t seed, bool use_hw, bool use_jit,
                 std::uint64_t total_iterations, std::uint64_t verbose_from, const std::string& findings_dir,
                 const std::string& outliers_dir, SharedCounters& counters, WorkerHeartbeat& heartbeat) {
  heartbeat.last_tick_ms.store(GetTickCount64(), std::memory_order_relaxed);
  _set_se_translator(seh_translator);
  InstructionGenerator gen(mix_seed(seed, worker_id));
  HardwareSession hw;
  // Reused together across every TestCase this worker runs -- see seven_jit_lane.hpp's comment on
  // why a fresh JitExecutor per call would be far too expensive, AND why memory has to be reused in
  // lockstep with it rather than freshly constructed per call (the cache staleness check is
  // Memory-instance-relative).
  seven_jit::JitExecutor jit_executor;
  seven::Memory jit_memory;
  const char* seven_label = use_jit ? "seven-jit" : "seven";
  const std::string seven_vs_unicorn_label = std::string(seven_label) + "-vs-unicorn";
  const std::string seven_vs_hw_label = std::string(seven_label) + "-vs-hw";
  bool worker_use_hw = use_hw;
  if (worker_use_hw) {
    const std::wstring victim_path = exe_dir_w() + L"\\seven_fuzz_victim.exe";
    std::string err;
    if (!hw.start(victim_path, err)) {
      std::lock_guard<std::mutex> lock(counters.io_mutex);
      std::printf("[worker %d] hardware lane unavailable, continuing without it: %s\n", worker_id, err.c_str());
      std::fflush(stdout);
      worker_use_hw = false;
    } else {
      std::lock_guard<std::mutex> lock(counters.io_mutex);
      std::printf("[worker %d] hardware session started\n", worker_id);
      std::fflush(stdout);
    }
  }

  // JitExecutor never releases an individual compiled function's executable memory (see
  // RuntimeKeepalive in seven-jit's block_compiler.cpp for why that path is disabled), and this
  // lane forces a fresh recompile on literally every TestCase -- left unchecked, that's an
  // unbounded leak over a long run. Periodically dropping the whole cache and JitRuntime at once
  // (JitExecutor::recycle(), safe to call here since nothing is executing between TestCases) frees
  // everything compiled so far in one shot without needing per-function release() at all. every
  // TestCase re-maps memory with a fresh epoch anyway, so there is no cross-call cache warmth this
  // lane could lose by recycling -- the only cost is occasionally reconstructing the JitRuntime.
  constexpr std::uint64_t kJitRecycleInterval = 2000;

  for (std::uint64_t local_i = 0; local_i < iterations; ++local_i) {
    heartbeat.last_tick_ms.store(GetTickCount64(), std::memory_order_relaxed);
    heartbeat.local_i.store(local_i, std::memory_order_relaxed);
    if (use_jit && local_i != 0 && local_i % kJitRecycleInterval == 0) {
      jit_executor.recycle();
    }
    const TestCase tc = gen.next();
    if (local_i >= verbose_from) {
      std::lock_guard<std::mutex> lock(counters.io_mutex);
      std::printf("[worker %d][%llu] running (%s)\n", worker_id, static_cast<unsigned long long>(local_i),
                  tc.text.c_str());
      std::fflush(stdout);
    }

    try {
    LaneOutcome seven_out, unicorn_out;
    {
      std::lock_guard<std::mutex> compute_lock(counters.compute_mutex);
      // Unicorn goes first when --jit is active: uc_open()'s first uc_mem_map() lazily triggers
      // Unicorn's own one-time-per-engine TCG prologue codegen, which needs to claim its own
      // reachable executable-memory region. Letting asmjit's JitRuntime (run_seven_jit, below)
      // allocate first reliably corrupts that codegen on this machine -- see seven_jit_lane.hpp's
      // sibling comment and the memory notes on this investigation. Unicorn is opened/closed fresh
      // every single call (unlike jit_executor, which is reused across the whole worker), so this
      // ordering has to hold on every iteration, not just the first.
      unicorn_out = g_no_unicorn ? LaneOutcome{} : run_unicorn(tc);
      seven_out = g_no_seven ? LaneOutcome{} : (use_jit ? run_seven_jit(tc, jit_executor, jit_memory) : run_seven(tc));
    }
    LaneOutcome hw_out;
    bool hw_ok_this_round = false;
    if (worker_use_hw) {
      hw_out = hw.run_one(tc);
      hw_ok_this_round = hw_out.setup_ok;
      if (!hw_ok_this_round) {
        counters.hw_harness_errors.fetch_add(1, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(counters.io_mutex);
        std::printf("[worker %d] hw harness error: %s\n", worker_id, hw_out.setup_error.c_str());
        std::fflush(stdout);
      }
    }

    std::vector<Divergence> seven_vs_unicorn, seven_vs_hw, unicorn_vs_hw;
    diff_pair(seven_vs_unicorn_label.c_str(), seven_out, unicorn_out, tc.flags_mask, tc.gpr_compare_mask,
              seven_vs_unicorn);
    if (hw_ok_this_round) {
      diff_pair(seven_vs_hw_label.c_str(), seven_out, hw_out, tc.flags_mask, tc.gpr_compare_mask, seven_vs_hw);
      diff_pair("unicorn-vs-hw", unicorn_out, hw_out, tc.flags_mask, tc.gpr_compare_mask, unicorn_vs_hw);
    }

    // When hardware is available and seven matches it exactly, a
    // seven-vs-unicorn/unicorn-vs-hw disagreement is Unicorn's own
    // limitation, not a seven bug (see Findings: the residual Unicorn-only
    // outlier pattern) -- so it doesn't get triaged as a "finding" needing
    // investigation. Still saved (separate directory, own index) rather
    // than only counted: this is real, useful data on Unicorn's actual
    // divergence-from-hardware surface, not something to throw away just
    // because it isn't a seven bug.
    const bool unicorn_only_outlier =
        hw_ok_this_round && seven_vs_hw.empty() && (!seven_vs_unicorn.empty() || !unicorn_vs_hw.empty());
    if (unicorn_only_outlier) {
      counters.unicorn_outlier_count.fetch_add(1, std::memory_order_relaxed);
      std::vector<Divergence> outlier_divergences = seven_vs_unicorn;
      outlier_divergences.insert(outlier_divergences.end(), unicorn_vs_hw.begin(), unicorn_vs_hw.end());
      const int oidx = counters.unicorn_outlier_saved.fetch_add(1, std::memory_order_relaxed);
      if (oidx < kMaxDumpFilesPerDir) {
        std::lock_guard<std::mutex> lock(counters.io_mutex);
        (void)write_finding(outliers_dir, oidx, tc, seven_label, seven_out, unicorn_out, hw_out, hw_ok_this_round,
                             outlier_divergences);
      }
    } else {
      std::vector<Divergence> divergences = seven_vs_unicorn;
      divergences.insert(divergences.end(), seven_vs_hw.begin(), seven_vs_hw.end());
      divergences.insert(divergences.end(), unicorn_vs_hw.begin(), unicorn_vs_hw.end());
      if (!divergences.empty()) {
        const int idx = counters.finding_count.fetch_add(1, std::memory_order_relaxed);
        // std::filesystem::create_directories (inside write_finding) targets
        // the same shared findings_dir from every thread -- serialize it
        // rather than rely on it being safe for concurrent callers, which
        // isn't guaranteed by the standard even if a given implementation
        // usually gets away with it.
        std::lock_guard<std::mutex> lock(counters.io_mutex);
        if (idx < kMaxDumpFilesPerDir) {
          const std::string path = write_finding(findings_dir, idx, tc, seven_label, seven_out, unicorn_out, hw_out,
                                                  hw_ok_this_round, divergences);
          std::printf("[worker %d] DIVERGENCE (%s) -> %s\n", worker_id, tc.text.c_str(), path.c_str());
        } else if (idx == kMaxDumpFilesPerDir) {
          std::printf("[worker %d] %d dumps written, still counting but no longer writing them\n", worker_id,
                      kMaxDumpFilesPerDir);
        }
        std::fflush(stdout);
      }
    }

    const std::uint64_t done = counters.total_done.fetch_add(1, std::memory_order_relaxed) + 1;
    if (done % 5000 == 0) {
      std::lock_guard<std::mutex> lock(counters.io_mutex);
      std::printf("progress: %llu/%llu, findings=%d, unicorn_only_outliers=%llu, hw_harness_errors=%llu\n",
                  static_cast<unsigned long long>(done), static_cast<unsigned long long>(total_iterations),
                  counters.finding_count.load(std::memory_order_relaxed),
                  static_cast<unsigned long long>(counters.unicorn_outlier_count.load(std::memory_order_relaxed)),
                  static_cast<unsigned long long>(counters.hw_harness_errors.load(std::memory_order_relaxed)));
      std::fflush(stdout);
    }
    } catch (const SehException& e) {
      // Diagnostic only, not a recovery -- state past this point (this
      // worker's HardwareSession in particular) may be corrupted, so stop
      // this worker rather than risk continuing on bad state. Other workers
      // are unaffected and keep going.
      std::lock_guard<std::mutex> lock(counters.io_mutex);
      std::printf("[worker %d][local %llu] CRASH: structured exception 0x%08X at %p while running (%s)\n", worker_id,
                  static_cast<unsigned long long>(local_i), e.code, e.address, tc.text.c_str());
      std::fflush(stdout);
      heartbeat.finished.store(true, std::memory_order_relaxed);
      return;
    } catch (const std::exception& e) {
      std::lock_guard<std::mutex> lock(counters.io_mutex);
      std::printf("[worker %d][local %llu] CRASH: C++ exception (%s) while running (%s)\n", worker_id,
                  static_cast<unsigned long long>(local_i), e.what(), tc.text.c_str());
      std::fflush(stdout);
      heartbeat.finished.store(true, std::memory_order_relaxed);
      return;
    } catch (...) {
      std::lock_guard<std::mutex> lock(counters.io_mutex);
      std::printf("[worker %d][local %llu] CRASH: unknown exception while running (%s)\n", worker_id,
                  static_cast<unsigned long long>(local_i), tc.text.c_str());
      std::fflush(stdout);
      heartbeat.finished.store(true, std::memory_order_relaxed);
      return;
    }
  }
  counters.generator_discards.fetch_add(gen.discarded_attempts(), std::memory_order_relaxed);
  heartbeat.finished.store(true, std::memory_order_relaxed);
}

// Runs `sweep_seeds` fresh CHILD PROCESSES of this same exe, one per derived seed, moving each
// child's findings/outliers/dumps into a per-seed subdirectory before the next child starts (each
// process invocation resets its own finding-index counters, so without this a later child would
// clobber an earlier one's output). A crash, hang, or watchdog-kill on one seed just moves on to the
// next -- this is the whole point: --jit's known shared-heap-corruption crash/hang (see the project
// notes) doesn't reset within one process, so a single long run dies almost immediately and never
// gets real depth, while a fresh process per seed sidesteps that entirely and still gets full
// coverage over many seeds. Same shape as the throwaway sweep script used to first measure this
// tonight, now a permanent, supported feature instead of an ad hoc tool.
int run_seed_sweep(std::uint64_t base_seed, std::uint64_t iterations_per_seed, unsigned int sweep_seeds,
                    bool use_hw, bool use_jit, unsigned int watchdog_seconds, unsigned int threads,
                    bool threads_explicit) {
  namespace fs = std::filesystem;

  wchar_t exe_path_buf[MAX_PATH];
  const DWORD exe_path_len = GetModuleFileNameW(nullptr, exe_path_buf, MAX_PATH);
  const std::wstring exe_path(exe_path_buf, exe_path_len);
  const fs::path exe_dir_path = fs::path(exe_path).parent_path();
  const fs::path sweep_dir = exe_dir_path / "sweep_results";
  fs::create_directories(sweep_dir);

  std::printf(
      "seven-fuzzer: sweeping %u seeds x %llu iterations each (derived from base seed 0x%llX), engine=%s "
      "hw=%s watchdog=%us -- results under %ls\n",
      sweep_seeds, static_cast<unsigned long long>(iterations_per_seed), static_cast<unsigned long long>(base_seed),
      use_jit ? "seven-jit" : "seven", use_hw ? "on" : "off", watchdog_seconds, sweep_dir.c_str());
  std::fflush(stdout);

  unsigned int clean = 0, crashed = 0, watchdog_killed = 0, supervisor_killed = 0;
  const DWORD supervisor_timeout_ms =
      (watchdog_seconds + 60) * 1000 + static_cast<DWORD>(std::min<std::uint64_t>(iterations_per_seed, 600000));

  for (unsigned int i = 0; i < sweep_seeds; ++i) {
    const std::uint64_t child_seed = mix_seed(base_seed, static_cast<int>(i));
    wchar_t seed_hex[32];
    swprintf(seed_hex, 32, L"0x%llX", static_cast<unsigned long long>(child_seed));

    std::wstring cmdline = L"\"" + exe_path + L"\" " + std::to_wstring(iterations_per_seed) + L" " + seed_hex;
    if (!use_hw) cmdline += L" --no-hw";
    if (use_jit) cmdline += L" --jit";
    cmdline += L" --watchdog-seconds " + std::to_wstring(watchdog_seconds);
    if (threads_explicit) cmdline += L" --threads " + std::to_wstring(threads);

    STARTUPINFOW si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    const BOOL created = CreateProcessW(exe_path.c_str(), cmdline.data(), nullptr, nullptr, FALSE, 0, nullptr,
                                        exe_dir_path.wstring().c_str(), &si, &pi);
    if (!created) {
      std::printf("[sweep %u/%u] seed=0x%llX FAILED TO LAUNCH (err=%lu)\n", i + 1, sweep_seeds,
                  static_cast<unsigned long long>(child_seed), GetLastError());
      std::fflush(stdout);
      continue;
    }

    const DWORD wait_result = WaitForSingleObject(pi.hProcess, supervisor_timeout_ms);
    const char* status;
    if (wait_result == WAIT_TIMEOUT) {
      // The child's own watchdog should always catch this first -- reaching this means even that
      // didn't fire, which itself would be worth investigating, not just a normal outcome to expect.
      TerminateProcess(pi.hProcess, 3);
      status = "SUPERVISOR TIMEOUT (child's own watchdog never fired)";
      ++supervisor_killed;
    } else {
      DWORD exit_code = 0;
      GetExitCodeProcess(pi.hProcess, &exit_code);
      if (exit_code == 2) {
        status = "watchdog-killed";
        ++watchdog_killed;
      } else if (exit_code != 0) {
        status = "crashed";
        ++crashed;
      } else {
        status = "clean";
        ++clean;
      }
    }
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

    // Move this child's output aside before the next child starts and resets the counters findings/
    // unicorn_outliers are numbered from. child_seed (and therefore the destination path below) is
    // deterministic from base_seed + index, so re-running the same base seed reaches this exact same
    // destination -- std::filesystem::rename throws on Windows if the destination already exists, and
    // this whole function had no try/catch around it, so a second sweep over an already-swept base
    // seed reliably crashed the supervisor right here. Pre-clearing the destination and using the
    // error_code overloads makes a repeat sweep just overwrite stale results instead.
    const std::wstring seed_tag = seed_hex;
    for (const auto& [src_name, dst_prefix] :
        {std::pair<const wchar_t*, const wchar_t*>{L"findings", L"findings_"},
         std::pair<const wchar_t*, const wchar_t*>{L"unicorn_outliers", L"outliers_"}}) {
      const fs::path src = exe_dir_path / src_name;
      std::error_code ec;
      if (fs::exists(src, ec) && !fs::is_empty(src, ec)) {
        const fs::path dst = sweep_dir / (dst_prefix + seed_tag);
        fs::remove_all(dst, ec);
        fs::rename(src, dst, ec);
        if (ec) {
          std::fwprintf(stderr, L"[sweep] warning: failed to archive %ls: %hs\n", src.c_str(), ec.message().c_str());
        }
      }
      fs::remove_all(src, ec);
    }
    for (const wchar_t* dmp_name : {L"crash.dmp", L"watchdog_hang.dmp"}) {
      const fs::path src = exe_dir_path / dmp_name;
      std::error_code ec;
      if (fs::exists(src, ec)) {
        const std::wstring stem = fs::path(dmp_name).stem().wstring();
        const fs::path dst = sweep_dir / (stem + L"_" + seed_tag + L".dmp");
        fs::remove_all(dst, ec);
        fs::rename(src, dst, ec);
        if (ec) {
          std::fwprintf(stderr, L"[sweep] warning: failed to archive %ls: %hs\n", src.c_str(), ec.message().c_str());
        }
      }
    }

    std::printf("[sweep %u/%u] seed=0x%llX %s\n", i + 1, sweep_seeds, static_cast<unsigned long long>(child_seed),
                status);
    std::fflush(stdout);
  }

  std::printf(
      "sweep done: %u seeds -- clean=%u crashed=%u watchdog_killed=%u supervisor_killed=%u -- results under "
      "%ls\n",
      sweep_seeds, clean, crashed, watchdog_killed, supervisor_killed, sweep_dir.c_str());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  SetUnhandledExceptionFilter(write_crash_dump);
  std::uint64_t iterations = 200000;
  std::uint64_t seed = 0xC0FFEEULL;
  bool use_hw = true;
  bool use_jit = false;
  std::uint64_t verbose_from = UINT64_MAX;
  unsigned int threads = 0;  // 0 = auto
  bool threads_explicit = false;
  unsigned int watchdog_seconds = 60;  // 0 disables; see run_watchdog
  unsigned int sweep_seeds = 0;  // 0 = normal single run; see run_seed_sweep

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--no-hw") {
      use_hw = false;
    } else if (a == "--watchdog-seconds" && i + 1 < argc) {
      watchdog_seconds = static_cast<unsigned int>(std::strtoul(argv[++i], nullptr, 10));
    } else if (a == "--sweep-seeds" && i + 1 < argc) {
      sweep_seeds = static_cast<unsigned int>(std::strtoul(argv[++i], nullptr, 10));
    } else if (a == "--jit") {
      // Runs the "seven" side of every comparison through seven-jit's JitExecutor instead of
      // seven_core's plain interpreter -- same TestCase, same Unicorn/hardware oracles, just a
      // different engine under test. Off by default so a bare run still verifies the interpreter,
      // which is what most of this fuzzer's existing findings/methodology assumes.
      use_jit = true;
    } else if (a == "--no-unicorn") {
      g_no_unicorn = true;
    } else if (a == "--no-seven") {
      g_no_seven = true;
    } else if (a == "--verbose-from" && i + 1 < argc) {
      verbose_from = std::strtoull(argv[++i], nullptr, 10);
    } else if (a == "--threads" && i + 1 < argc) {
      threads = static_cast<unsigned int>(std::strtoul(argv[++i], nullptr, 10));
      threads_explicit = true;
    } else if (i == 1) {
      iterations = std::strtoull(a.c_str(), nullptr, 10);
    } else if (i == 2) {
      seed = std::strtoull(a.c_str(), nullptr, 0);
    }
  }

  if (sweep_seeds > 0) {
    return run_seed_sweep(seed, iterations, sweep_seeds, use_hw, use_jit, watchdog_seconds, threads,
                          threads_explicit);
  }

  if (argc > 1 && std::string(argv[1]) == "--probe-bsr-zero") {
    // One hand-crafted test case, ground-truth check for "does real hardware
    // leave the destination fully untouched when BSR's source is 0, or does
    // it apply the usual 32-bit-write zero-extension" -- see Findings #4.
    TestCase tc;
    tc.bytes = {0x0F, 0xBD, 0x17};  // BSR EDX, [RDI]
    tc.text = "0F BD 17 (BSR EDX,[RDI])";
    tc.touches_memory = true;
    tc.flags_mask = 0x40;  // ZF only
    for (int i = 0; i < 16; ++i) tc.initial.gpr[static_cast<std::size_t>(i)] = 0x1122334455667788ull;
    tc.initial.gpr[4] = kStackTop;
    tc.initial.gpr[7] = kDataBase;
    tc.initial.rflags = 0x202;
    // data_seed defaults to all-zero already -- [RDI] reads 0.

    const LaneOutcome seven_out = run_seven(tc);
    const LaneOutcome unicorn_out = run_unicorn(tc);
    HardwareSession probe_hw;
    std::string herr;
    if (!probe_hw.start(exe_dir_w() + L"\\seven_fuzz_victim.exe", herr)) {
      std::printf("hardware unavailable: %s\n", herr.c_str());
      return 1;
    }
    const LaneOutcome hw_out = probe_hw.run_one(tc);

    std::printf("initial RDX = %016llX\n", static_cast<unsigned long long>(tc.initial.gpr[2]));
    std::printf("seven:   stop=%s RDX=%016llX\n", stop_name(seven_out.stop),
                static_cast<unsigned long long>(seven_out.after.gpr[2]));
    std::printf("unicorn: stop=%s RDX=%016llX\n", stop_name(unicorn_out.stop),
                static_cast<unsigned long long>(unicorn_out.after.gpr[2]));
    std::printf("hw:      stop=%s%s RDX=%016llX\n", stop_name(hw_out.stop),
                hw_out.setup_ok ? "" : (" (" + hw_out.setup_error + ")").c_str(),
                static_cast<unsigned long long>(hw_out.after.gpr[2]));
    return 0;
  }

  if (threads == 0) {
    const unsigned int hw_conc = std::thread::hardware_concurrency();
    // Debug-event round trips are kernel-transition-bound, not CPU-bound, so
    // some oversubscription is fine -- but leave real headroom for the OS
    // and whatever else is running on the box.
    threads = hw_conc == 0 ? 4 : (hw_conc > 4 ? hw_conc - 2 : hw_conc);
  }
  if (threads > iterations) threads = static_cast<unsigned int>(iterations == 0 ? 1 : iterations);
  if (threads == 0) threads = 1;

  // --jit is clamped to one thread until seven-jit's compiled code carries real Windows SEH
  // unwind info (asmjit never calls RtlAddFunctionTable for the executable memory it hands out).
  // A hardware fault while a compiled block is actually on the call stack -- routine under random
  // fuzzing -- has undefined unwind behavior without that, and multi-threaded --jit runs reproduce
  // a hard, unrecoverable crash far faster than single-threaded ones do (observed within a couple
  // dozen iterations per worker). See block_compiler.cpp's RuntimeKeepalive comment for the full
  // investigation. Remove this clamp once that's fixed.
  //
  // Single-threaded --jit is NOT fully immune either, and the failure mode there is worse than a
  // clean catch: Unicorn opens/closes a fresh engine every single TestCase (see run_unicorn), and
  // its own TCG prologue codegen (tcg_out8/tcg_target_qemu_prologue, ordinary compiled code, not
  // JIT-generated) can crash writing into its own code buffer after enough alloc/free cycles have
  // interleaved with asmjit's own allocator -- confirmed via cdb, and confirmed to need --jit
  // specifically (the exact same seeds run cleanly through 20000+ iterations without it). That
  // particular crash IS caught cleanly by the SEH translator below (worker prints CRASH and stops).
  // But the same interaction has also been observed to hang indefinitely instead -- no crash
  // printed, no forward progress -- on seeds where the clean-crash variant does NOT reproduce,
  // which rules out a fixed per-run iteration threshold as a safe bound. Root cause not fully
  // pinned down (unclear whether this is the same missing-unwind-info hazard as above manifesting
  // differently, or a separate address-space-exhaustion retry loop inside Unicorn's own allocator);
  // not something to guess-fix without room to verify.
  //
  // Checked and ruled out as the mechanism here: this vendored Unicorn's uc_close() DOES correctly
  // call may_remove_handler() (via release_common -> free_code_gen_buffer, wired through
  // uc->release = x86_release in target/i386/unicorn.c) before freeing the engine, so the
  // Windows-only lazy-commit VEH it registers per uc_open() is properly torn down every call --
  // there is no leaked/stale vectored handler accumulating across the fresh-engine-per-TestCase
  // cycle run_unicorn does. An earlier pass through this file assumed otherwise and prototyped
  // reusing one Unicorn engine across calls to avoid the (nonexistent) leak; that reuse design was
  // reverted after it reintroduced real correctness bugs of its own (stale translation-block cache
  // and an exception-hook/engine-reuse interaction both producing false divergences) chasing a
  // problem that wasn't actually there. Left as an open question for whoever picks this back up.
  //
  // Unattended runs no longer rely solely on an external wrapper for this: main() below spawns a
  // watchdog thread (run_watchdog) that terminates the process and writes a live minidump if a
  // worker stops making progress, so a hang here is now bounded and postmortem-debuggable without
  // needing to catch it live under cdb. Still worth wrapping in an external wall-clock timeout too
  // (e.g. `timeout` on Linux, a job object or scheduled-task deadline on Windows) as a second layer.
  if (use_jit && threads > 1) {
    std::printf(
        "seven-fuzzer: --jit is not yet safe multi-threaded (missing SEH unwind info for JIT "
        "code) -- clamping threads %u -> 1\n",
        threads);
    threads = 1;
  }

  std::printf("seven-fuzzer: iterations=%llu seed=0x%llX hw=%s engine=%s threads=%u\n",
              static_cast<unsigned long long>(iterations), static_cast<unsigned long long>(seed),
              use_hw ? "on" : "off", use_jit ? "seven-jit" : "seven", threads);
  // The VEX/EVEX families are only generated on a host whose silicon can adjudicate them, so two
  // runs of the same seed on different machines can cover different instruction sets. Printing it
  // means a log says which one it was rather than leaving it to be guessed later.
  std::printf("seven-fuzzer: host simd = %s\n", fuzz::host_caps().summary().c_str());
  std::fflush(stdout);

  if (watchdog_seconds != 0) {
    std::printf("seven-fuzzer: watchdog armed, %u second timeout (--watchdog-seconds 0 to disable)\n",
                watchdog_seconds);
  }
  std::fflush(stdout);

  const std::string findings_dir = exe_dir() + "/findings";
  const std::string outliers_dir = exe_dir() + "/unicorn_outliers";
  SharedCounters counters;
  std::vector<WorkerHeartbeat> heartbeats(threads);

  std::thread watchdog_thread(run_watchdog, std::ref(heartbeats), watchdog_seconds, std::ref(counters.io_mutex));

  std::vector<std::thread> pool;
  pool.reserve(threads);
  const std::uint64_t base_share = iterations / threads;
  const std::uint64_t remainder = iterations % threads;
  for (unsigned int w = 0; w < threads; ++w) {
    const std::uint64_t share = base_share + (w < remainder ? 1 : 0);
    pool.emplace_back(run_worker, static_cast<int>(w), share, seed, use_hw, use_jit, iterations, verbose_from,
                       findings_dir, outliers_dir, std::ref(counters), std::ref(heartbeats[w]));
  }
  for (auto& t : pool) t.join();
  watchdog_thread.join();

  std::printf("done. findings=%d unicorn_only_outliers=%llu hw_harness_errors=%llu generator_discards=%llu\n",
              counters.finding_count.load(), static_cast<unsigned long long>(counters.unicorn_outlier_count.load()),
              static_cast<unsigned long long>(counters.hw_harness_errors.load()),
              static_cast<unsigned long long>(counters.generator_discards.load()));
  return 0;
}
