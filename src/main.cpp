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
//
// --jit runs the "seven" side through seven-jit's JitExecutor instead of seven_core's plain
// interpreter (see lanes/seven_jit_lane.hpp) -- everything else about the run (TestCase generation,
// the Unicorn/hardware oracles, the comparator, reporting) is unchanged. This is a single check on
// which engine is under test, not a fourth parallel lane: run once with --jit, once without, to
// verify both independently against the exact same two oracles.

// NOMINMAX: seven_jit/jit_executor.hpp transitively pulls in seven_core's float80.hpp, which
// defines std::numeric_limits<Float80>::min()/max() -- windows.h's own min/max macros silently
// mangle those into broken syntax if left defined, so this has to come before windows.h itself, not
// just before the seven_jit include below.
#define NOMINMAX
#include <windows.h>

#include <dbghelp.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <eh.h>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

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
    MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(), file,
                       static_cast<MINIDUMP_TYPE>(MiniDumpWithFullMemory | MiniDumpWithHandleData | MiniDumpWithThreadInfo),
                       &mei, nullptr, nullptr);
    CloseHandle(file);
    std::fwprintf(stderr, L"[fatal] unhandled exception 0x%08lX at %p -- wrote %s\n",
                  ep->ExceptionRecord->ExceptionCode, ep->ExceptionRecord->ExceptionAddress, dump_path.c_str());
  }
  return EXCEPTION_EXECUTE_HANDLER;  // terminate after dumping
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
               std::vector<Divergence>& out) {
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

struct SharedCounters {
  std::atomic<int> finding_count{0};
  std::atomic<int> unicorn_outlier_saved{0};  // index counter for the saved-outlier files, separate from the count below
  std::atomic<std::uint64_t> unicorn_outlier_count{0};
  std::atomic<std::uint64_t> hw_harness_errors{0};
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

void run_worker(int worker_id, std::uint64_t iterations, std::uint64_t seed, bool use_hw, bool use_jit,
                 std::uint64_t total_iterations, std::uint64_t verbose_from, const std::string& findings_dir,
                 const std::string& outliers_dir, SharedCounters& counters) {
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

  for (std::uint64_t local_i = 0; local_i < iterations; ++local_i) {
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
    diff_pair(seven_vs_unicorn_label.c_str(), seven_out, unicorn_out, tc.flags_mask, seven_vs_unicorn);
    if (hw_ok_this_round) {
      diff_pair(seven_vs_hw_label.c_str(), seven_out, hw_out, tc.flags_mask, seven_vs_hw);
      diff_pair("unicorn-vs-hw", unicorn_out, hw_out, tc.flags_mask, unicorn_vs_hw);
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
      std::lock_guard<std::mutex> lock(counters.io_mutex);
      (void)write_finding(outliers_dir, oidx, tc, seven_label, seven_out, unicorn_out, hw_out, hw_ok_this_round,
                           outlier_divergences);
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
        const std::string path = write_finding(findings_dir, idx, tc, seven_label, seven_out, unicorn_out, hw_out,
                                                hw_ok_this_round, divergences);
        std::printf("[worker %d] DIVERGENCE (%s) -> %s\n", worker_id, tc.text.c_str(), path.c_str());
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
      return;
    } catch (const std::exception& e) {
      std::lock_guard<std::mutex> lock(counters.io_mutex);
      std::printf("[worker %d][local %llu] CRASH: C++ exception (%s) while running (%s)\n", worker_id,
                  static_cast<unsigned long long>(local_i), e.what(), tc.text.c_str());
      std::fflush(stdout);
      return;
    } catch (...) {
      std::lock_guard<std::mutex> lock(counters.io_mutex);
      std::printf("[worker %d][local %llu] CRASH: unknown exception while running (%s)\n", worker_id,
                  static_cast<unsigned long long>(local_i), tc.text.c_str());
      std::fflush(stdout);
      return;
    }
  }
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

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--no-hw") {
      use_hw = false;
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
    } else if (i == 1) {
      iterations = std::strtoull(a.c_str(), nullptr, 10);
    } else if (i == 2) {
      seed = std::strtoull(a.c_str(), nullptr, 0);
    }
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
  // fuzzing, and normally caught cleanly by this process's own SEH translator -- has undefined
  // unwind behavior without that, and multi-threaded --jit runs reproduce a hard, unrecoverable
  // crash far faster than single-threaded ones do (observed within a couple dozen iterations per
  // worker vs. a clean 20000+ single-threaded). See block_compiler.cpp's RuntimeKeepalive comment
  // for the full investigation. Remove this clamp once that's fixed.
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
  std::fflush(stdout);

  const std::string findings_dir = exe_dir() + "/findings";
  const std::string outliers_dir = exe_dir() + "/unicorn_outliers";
  SharedCounters counters;

  std::vector<std::thread> pool;
  pool.reserve(threads);
  const std::uint64_t base_share = iterations / threads;
  const std::uint64_t remainder = iterations % threads;
  for (unsigned int w = 0; w < threads; ++w) {
    const std::uint64_t share = base_share + (w < remainder ? 1 : 0);
    pool.emplace_back(run_worker, static_cast<int>(w), share, seed, use_hw, use_jit, iterations, verbose_from,
                       findings_dir, outliers_dir, std::ref(counters));
  }
  for (auto& t : pool) t.join();

  std::printf("done. findings=%d unicorn_only_outliers=%llu hw_harness_errors=%llu\n",
              counters.finding_count.load(), static_cast<unsigned long long>(counters.unicorn_outlier_count.load()),
              static_cast<unsigned long long>(counters.hw_harness_errors.load()));
  return 0;
}
