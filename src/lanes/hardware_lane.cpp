#include "lanes/hardware_lane.hpp"

#include <windows.h>
#include <sddl.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <string>

#pragma comment(lib, "advapi32.lib")

namespace sf {

namespace {

constexpr DWORD kWaitTimeoutMs = 10000;
constexpr std::uint64_t kRespawnEvery = 2000;  // periodic hygiene respawn

void apply_context_to_gpr(CONTEXT& ctx, const RegState& r) {
  ctx.Rax = r.gpr[0];
  ctx.Rcx = r.gpr[1];
  ctx.Rdx = r.gpr[2];
  ctx.Rbx = r.gpr[3];
  ctx.Rsp = r.gpr[4];
  ctx.Rbp = r.gpr[5];
  ctx.Rsi = r.gpr[6];
  ctx.Rdi = r.gpr[7];
  ctx.R8 = r.gpr[8];
  ctx.R9 = r.gpr[9];
  ctx.R10 = r.gpr[10];
  ctx.R11 = r.gpr[11];
  ctx.R12 = r.gpr[12];
  ctx.R13 = r.gpr[13];
  ctx.R14 = r.gpr[14];
  ctx.R15 = r.gpr[15];
}

void apply_context_to_xmm(CONTEXT& ctx, const RegState& r) {
  for (int i = 0; i < 16; ++i) {
    ctx.FltSave.XmmRegisters[i].Low = r.xmm_lo[static_cast<std::size_t>(i)];
    ctx.FltSave.XmmRegisters[i].High = static_cast<LONGLONG>(r.xmm_hi[static_cast<std::size_t>(i)]);
  }
}

// FltSave is the raw FXSAVE legacy area, so FloatRegisters[i] is ST(i) (stack-relative) and
// TagWord is the abridged 8-bit form indexed by PHYSICAL register. Both were confirmed on this
// host rather than taken from the manual: FNINIT/FLDZ/FLD1 leaves TOP=6, and the resulting image
// had 1.0 in FloatRegisters[0] with TagWord 0x00C0 (bits 6 and 7, the two physical registers in
// use). See X87State's comment for which of the two forms the comparison is done in.
void apply_context_to_x87(CONTEXT& ctx, const RegState& r) {
  ctx.FltSave.ControlWord = r.x87.control_word;
  ctx.FltSave.StatusWord = r.x87.status_word;
  std::uint8_t abridged = 0;
  for (int phys = 0; phys < 8; ++phys) {
    if (((r.x87.tag_word >> (2 * phys)) & 0x3u) != 0x3u) abridged |= static_cast<std::uint8_t>(1u << phys);
  }
  ctx.FltSave.TagWord = abridged;
  for (int i = 0; i < 8; ++i) {
    auto* slot = reinterpret_cast<std::uint8_t*>(&ctx.FltSave.FloatRegisters[i]);
    std::memset(slot, 0, sizeof(M128A));
    const std::uint64_t signif = r.x87.signif[static_cast<std::size_t>(i)];
    const std::uint16_t signexp = r.x87.signexp[static_cast<std::size_t>(i)];
    std::memcpy(slot, &signif, sizeof(signif));
    std::memcpy(slot + 8, &signexp, sizeof(signexp));
  }
}

void read_x87_from_context(const CONTEXT& ctx, RegState& r) {
  r.x87.control_word = ctx.FltSave.ControlWord;
  r.x87.status_word = ctx.FltSave.StatusWord;
  for (int i = 0; i < 8; ++i) {
    const auto* slot = reinterpret_cast<const std::uint8_t*>(&ctx.FltSave.FloatRegisters[i]);
    std::memcpy(&r.x87.signif[static_cast<std::size_t>(i)], slot, 8);
    std::memcpy(&r.x87.signexp[static_cast<std::size_t>(i)], slot + 8, 2);
  }
  // Expand the abridged byte into the architectural two-bits-per-register word, which needs the
  // register contents as well as the in-use bit. Note the byte is physical-order while the
  // registers just read above are stack-relative, so this walks TOP the other way.
  const auto top = static_cast<int>((ctx.FltSave.StatusWord >> 11) & 0x7u);
  std::uint16_t full = 0;
  for (int phys = 0; phys < 8; ++phys) {
    std::uint8_t tag = 0x3;
    if (((ctx.FltSave.TagWord >> phys) & 1u) != 0) {
      const int st = (phys - top) & 0x7;
      tag = x87_classify_tag(r.x87.signexp[static_cast<std::size_t>(st)], r.x87.signif[static_cast<std::size_t>(st)]);
    }
    full |= static_cast<std::uint16_t>(static_cast<std::uint16_t>(tag) << (2 * phys));
  }
  r.x87.tag_word = full;
}

void read_xmm_from_context(const CONTEXT& ctx, RegState& r) {
  for (int i = 0; i < 16; ++i) {
    r.xmm_lo[static_cast<std::size_t>(i)] = ctx.FltSave.XmmRegisters[i].Low;
    r.xmm_hi[static_cast<std::size_t>(i)] = static_cast<std::uint64_t>(ctx.FltSave.XmmRegisters[i].High);
  }
}

void read_gpr_from_context(const CONTEXT& ctx, RegState& r) {
  r.gpr[0] = ctx.Rax;
  r.gpr[1] = ctx.Rcx;
  r.gpr[2] = ctx.Rdx;
  r.gpr[3] = ctx.Rbx;
  r.gpr[4] = ctx.Rsp;
  r.gpr[5] = ctx.Rbp;
  r.gpr[6] = ctx.Rsi;
  r.gpr[7] = ctx.Rdi;
  r.gpr[8] = ctx.R8;
  r.gpr[9] = ctx.R9;
  r.gpr[10] = ctx.R10;
  r.gpr[11] = ctx.R11;
  r.gpr[12] = ctx.R12;
  r.gpr[13] = ctx.R13;
  r.gpr[14] = ctx.R14;
  r.gpr[15] = ctx.R15;
  r.rip = ctx.Rip;
  r.rflags = ctx.EFlags;
}

// Windows collapses a genuine x86 #GP into the SAME STATUS_ACCESS_VIOLATION (0xC0000005) code it
// uses for a real #PF -- confirmed empirically via a standalone VectoredExceptionHandler probe
// (a deliberately misaligned legacy MOVAPS memory operand, which the SDM guarantees #GP(0) for):
// the resulting exception record showed ExceptionInformation[0] == 0 and
// ExceptionInformation[1] == ULONG_PTR(-1) -- there's no CR2 equivalent for #GP, so the kernel has
// no real faulting address to report and leaves that slot as an all-ones sentinel instead. A
// genuine #PF always carries a real faulting virtual address there. This was previously
// misclassified as Stop::pf, which is what made the legacy-SSE-alignment #GP fix elsewhere in this
// repo look like it hadn't changed anything against the hardware lane -- it had, this was just
// mislabeled. Reclassify only that exact, unambiguous sentinel shape; anything else (including a
// real #PF that happens to have NumberParameters>=2 with a real address) keeps its normal pf
// classification.
[[nodiscard]] Stop map_exception(const EXCEPTION_RECORD& rec) noexcept {
  switch (rec.ExceptionCode) {
    case EXCEPTION_SINGLE_STEP:
      return Stop::ok;
    case EXCEPTION_ACCESS_VIOLATION: {
      const bool gp_shaped = rec.NumberParameters >= 2 && rec.ExceptionInformation[1] == static_cast<ULONG_PTR>(-1);
      return gp_shaped ? Stop::gp : Stop::pf;
    }
    case EXCEPTION_ARRAY_BOUNDS_EXCEEDED:
    case EXCEPTION_IN_PAGE_ERROR:
      return Stop::pf;
    case EXCEPTION_ILLEGAL_INSTRUCTION:
      return Stop::ud;
    case EXCEPTION_PRIV_INSTRUCTION:
      return Stop::gp;
    case EXCEPTION_INT_DIVIDE_BY_ZERO:
    // Both map to x86's single #DE fault -- Windows' trap handler reports
    // divisor==0 as DIVIDE_BY_ZERO and a nonzero divisor whose quotient
    // doesn't fit as INT_OVERFLOW, but architecturally they're the same
    // exception DIV/IDIV can raise.
    case EXCEPTION_INT_OVERFLOW:
      return Stop::de;
    case EXCEPTION_BREAKPOINT:
      return Stop::bp;
    default:
      return Stop::other;
  }
}

}  // namespace

struct HardwareSession::Impl {
  HANDLE job = nullptr;
  HANDLE process = nullptr;
  HANDLE thread = nullptr;
  DWORD pid = 0;
  std::wstring victim_path;

  bool ready = false;
  DEBUG_EVENT pending_event{};
  bool pending_valid = false;
  std::uint64_t steps_since_respawn = 0;

  ~Impl() { teardown(); }

  // Drains debug events for a dying/dead victim so the OS can actually
  // finish tearing it down (a debuggee is not fully gone until its debugger
  // has continued it through to EXIT_PROCESS_DEBUG_EVENT). Bounded so a
  // stuck victim can never hang the fuzzer — the job object's
  // kill-on-close is the backstop if this loop gives up early.
  void drain_to_exit() {
    if (process == nullptr) return;
    TerminateProcess(process, 1);
    for (int i = 0; i < 50; ++i) {
      DEBUG_EVENT de{};
      if (!WaitForDebugEvent(&de, 200)) break;
      const bool exited = de.dwDebugEventCode == EXIT_PROCESS_DEBUG_EVENT;
      ContinueDebugEvent(de.dwProcessId, de.dwThreadId, DBG_CONTINUE);
      if (exited) break;
    }
  }

  void teardown() {
    if (process != nullptr) drain_to_exit();
    if (thread != nullptr) { CloseHandle(thread); thread = nullptr; }
    if (process != nullptr) { CloseHandle(process); process = nullptr; }
    if (job != nullptr) { CloseHandle(job); job = nullptr; }  // kill-on-close backstop
    ready = false;
    pending_valid = false;
    pid = 0;
  }

  [[nodiscard]] bool build_restricted_token(HANDLE& out_token, std::string& error) {
    HANDLE self_token = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(),
                           TOKEN_DUPLICATE | TOKEN_QUERY | TOKEN_ASSIGN_PRIMARY | TOKEN_ADJUST_DEFAULT,
                           &self_token)) {
      error = "OpenProcessToken failed";
      return false;
    }
    HANDLE dup_token = nullptr;
    const BOOL dup_ok = DuplicateTokenEx(self_token, TOKEN_ALL_ACCESS, nullptr, SecurityImpersonation,
                                         TokenPrimary, &dup_token);
    CloseHandle(self_token);
    if (!dup_ok) {
      error = "DuplicateTokenEx failed";
      return false;
    }

    HANDLE restricted_token = nullptr;
    const BOOL restrict_ok = CreateRestrictedToken(dup_token, DISABLE_MAX_PRIVILEGE, 0, nullptr, 0, nullptr, 0,
                                                    nullptr, &restricted_token);
    CloseHandle(dup_token);
    if (!restrict_ok) {
      error = "CreateRestrictedToken failed";
      return false;
    }

    PSID low_sid = nullptr;
    if (!ConvertStringSidToSidW(L"S-1-16-4096", &low_sid)) {  // Low mandatory integrity level
      CloseHandle(restricted_token);
      error = "ConvertStringSidToSidW failed";
      return false;
    }
    TOKEN_MANDATORY_LABEL label{};
    label.Label.Attributes = SE_GROUP_INTEGRITY;
    label.Label.Sid = low_sid;
    const BOOL set_ok = SetTokenInformation(restricted_token, TokenIntegrityLevel, &label,
                                             sizeof(TOKEN_MANDATORY_LABEL) + GetLengthSid(low_sid));
    LocalFree(low_sid);
    if (!set_ok) {
      CloseHandle(restricted_token);
      error = "SetTokenInformation(TokenIntegrityLevel) failed";
      return false;
    }

    out_token = restricted_token;
    return true;
  }

  [[nodiscard]] bool spawn_fresh(std::string& error) {
    teardown();

    HANDLE token = nullptr;
    if (!build_restricted_token(token, error)) return false;

    job = CreateJobObjectW(nullptr, nullptr);
    if (job == nullptr) {
      CloseHandle(token);
      error = "CreateJobObjectW failed";
      return false;
    }
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION jeli{};
    jeli.BasicLimitInformation.LimitFlags =
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_ACTIVE_PROCESS | JOB_OBJECT_LIMIT_PROCESS_MEMORY;
    jeli.BasicLimitInformation.ActiveProcessLimit = 1;
    jeli.ProcessMemoryLimit = 64ull * 1024 * 1024;
    SetInformationJobObject(job, JobObjectExtendedLimitInformation, &jeli, sizeof(jeli));
    JOBOBJECT_BASIC_UI_RESTRICTIONS ui{};
    ui.UIRestrictionsClass = JOB_OBJECT_UILIMIT_HANDLES | JOB_OBJECT_UILIMIT_DESKTOP |
                              JOB_OBJECT_UILIMIT_EXITWINDOWS | JOB_OBJECT_UILIMIT_DISPLAYSETTINGS |
                              JOB_OBJECT_UILIMIT_GLOBALATOMS | JOB_OBJECT_UILIMIT_SYSTEMPARAMETERS |
                              JOB_OBJECT_UILIMIT_WRITECLIPBOARD;
    SetInformationJobObject(job, JobObjectBasicUIRestrictions, &ui, sizeof(ui));

    HANDLE nul_handle = CreateFileW(L"NUL", GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                     nullptr, OPEN_EXISTING, 0, nullptr);
    STARTUPINFOW si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;
    si.hStdInput = nul_handle;
    si.hStdOutput = nul_handle;
    si.hStdError = nul_handle;

    // Minimal environment: only SystemRoot, so the loader has what it needs
    // without handing the sandboxed victim the real process's environment.
    wchar_t system_root[MAX_PATH] = L"";
    GetEnvironmentVariableW(L"SystemRoot", system_root, MAX_PATH);
    std::wstring env_block = L"SystemRoot=";
    env_block += system_root;
    env_block.push_back(L'\0');
    env_block.push_back(L'\0');

    wchar_t temp_dir[MAX_PATH] = L"";
    GetTempPathW(MAX_PATH, temp_dir);

    std::wstring cmdline = L"\"" + victim_path + L"\"";
    PROCESS_INFORMATION pi{};
    const BOOL created = CreateProcessAsUserW(
        token, victim_path.c_str(), cmdline.data(), nullptr, nullptr, TRUE,
        DEBUG_PROCESS | CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT, env_block.data(), temp_dir, &si, &pi);
    CloseHandle(token);
    if (nul_handle != nullptr && nul_handle != INVALID_HANDLE_VALUE) CloseHandle(nul_handle);

    if (!created) {
      error = "CreateProcessAsUserW failed (err=" + std::to_string(GetLastError()) + ")";
      CloseHandle(job);
      job = nullptr;
      return false;
    }

    pid = pi.dwProcessId;
    AssignProcessToJobObject(job, pi.hProcess);
    // pi.hProcess/pi.hThread are only used for the job-object assignment
    // above; the actual context/memory operations use the handles the
    // debug-event stream hands back (see CREATE_PROCESS_DEBUG_EVENT below),
    // which were observed to NOT always be the same underlying thread
    // identity as PROCESS_INFORMATION's — see the comment there. Close
    // these now rather than leak them.
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);

    // Pump debug events until the loader's automatic initial breakpoint —
    // our sync point. Nothing of the victim's own code (not even its
    // trivial main()) has executed yet when this fires.
    int uninteresting_events = 0;  // see the identical cap in run_one() for why
    constexpr int kMaxUninterestingEvents = 10000;
    for (;;) {
      DEBUG_EVENT de{};
      if (!WaitForDebugEvent(&de, kWaitTimeoutMs)) {
        error = "WaitForDebugEvent timed out waiting for initial breakpoint";
        teardown();
        return false;
      }
      if (de.dwDebugEventCode != EXCEPTION_DEBUG_EVENT && ++uninteresting_events > kMaxUninterestingEvents) {
        error = "too many non-exception debug events without reaching the sync point";
        teardown();
        return false;
      }
      if (de.dwDebugEventCode == LOAD_DLL_DEBUG_EVENT) {
        if (de.u.LoadDll.hFile != nullptr) CloseHandle(de.u.LoadDll.hFile);
        ContinueDebugEvent(de.dwProcessId, de.dwThreadId, DBG_CONTINUE);
        continue;
      }
      if (de.dwDebugEventCode == CREATE_PROCESS_DEBUG_EVENT) {
        if (de.u.CreateProcessInfo.hFile != nullptr) CloseHandle(de.u.CreateProcessInfo.hFile);
        // The debug-event-supplied thread/process handles are the ones
        // guaranteed to match the thread ID the rest of the debug-event
        // stream reports. CreateProcessAsUserW's own PROCESS_INFORMATION
        // handles were observed NOT to match (a different thread ID) on
        // this host, which meant GetThreadContext/SetThreadContext were
        // silently operating on the wrong thread — the real primary thread
        // stayed unredirected and eventually crashed on its own. Use these
        // instead; ownership stays with the system (not closed here).
        if (de.u.CreateProcessInfo.hThread != nullptr) thread = de.u.CreateProcessInfo.hThread;
        if (de.u.CreateProcessInfo.hProcess != nullptr) process = de.u.CreateProcessInfo.hProcess;
        ContinueDebugEvent(de.dwProcessId, de.dwThreadId, DBG_CONTINUE);
        continue;
      }
      if (de.dwDebugEventCode == EXCEPTION_DEBUG_EVENT &&
          de.u.Exception.ExceptionRecord.ExceptionCode == EXCEPTION_BREAKPOINT) {
        pending_event = de;
        pending_valid = true;
        break;
      }
      if (de.dwDebugEventCode == EXIT_PROCESS_DEBUG_EVENT) {
        error = "victim exited before reaching sync point";
        ContinueDebugEvent(de.dwProcessId, de.dwThreadId, DBG_CONTINUE);
        teardown();
        return false;
      }
      ContinueDebugEvent(de.dwProcessId, de.dwThreadId, DBG_CONTINUE);
    }

    // One reservation covering all three pages, not three separate calls:
    // VirtualAllocEx requires MEM_RESERVE addresses to be 64 KB-aligned
    // (allocation granularity), but kDataBase/kStackBase are only 4 KB
    // apart from kCodeBase. kCodeBase itself is 64 KB-aligned, so a single
    // reservation from there covers offsets 0x1000 (kDataBase) and 0x2000
    // (kStackBase) too. RWX everywhere so permission mismatches never
    // masquerade as execution-semantics differences (matches the other two
    // lanes).
    {
      LPVOID got = VirtualAllocEx(process, reinterpret_cast<LPVOID>(kCodeBase), 3 * kPageSize,
                                   MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE);
      if (got == nullptr || reinterpret_cast<std::uint64_t>(got) != kCodeBase) {
        error = "VirtualAllocEx failed for scratch region (err=" + std::to_string(GetLastError()) + ")";
        teardown();
        return false;
      }
    }

    ready = true;
    steps_since_respawn = 0;
    return true;
  }

  [[nodiscard]] bool ensure_ready(std::string& error) {
    if (ready) return true;
    return spawn_fresh(error);
  }

  [[nodiscard]] LaneOutcome run_one(const TestCase& tc) {
    LaneOutcome out;
    std::string err;
    if (!ensure_ready(err)) {
      out.setup_ok = false;
      out.setup_error = "hardware: " + err;
      return out;
    }

    CONTEXT ctx{};
    ctx.ContextFlags = CONTEXT_FULL;
    if (!GetThreadContext(thread, &ctx)) {
      out.setup_ok = false;
      out.setup_error = "hardware: GetThreadContext failed";
      teardown();
      return out;
    }
    apply_context_to_gpr(ctx, tc.initial);
    apply_context_to_xmm(ctx, tc.initial);
    // The victim process is reused across test cases and the FPU is the one piece of state that
    // survives a test case entirely on its own (stack contents, TOP, sticky exception flags), so
    // this is a reset as much as it is a setup: without it an FLD from case N would still be on
    // the stack when case N+1's FADD ran.
    apply_context_to_x87(ctx, tc.initial);
    ctx.Rip = kCodeBase;
    // Force TF (single-step) and reserved bit 1; strip everything the
    // generator doesn't own so a stray bit can't destabilize the victim.
    ctx.EFlags = static_cast<DWORD>((tc.initial.rflags & kCompareFlagsMask) | 0x100u /*TF*/ | 0x2u /*reserved*/);
    if (!SetThreadContext(thread, &ctx)) {
      out.setup_ok = false;
      out.setup_error = "hardware: SetThreadContext failed";
      teardown();
      return out;
    }

    SIZE_T written = 0;
    if (!tc.bytes.empty() &&
        !WriteProcessMemory(process, reinterpret_cast<LPVOID>(kCodeBase), tc.bytes.data(), tc.bytes.size(),
                             &written)) {
      out.setup_ok = false;
      out.setup_error = "hardware: WriteProcessMemory(code) failed";
      teardown();
      return out;
    }
    // Reset data and stack pages every test case — unlike the seven/unicorn
    // lanes, this process is reused across test cases, so leftover writes
    // (e.g. a PUSH from a prior case) would otherwise leak forward.
    std::array<std::uint8_t, kPageSize> zero{};
    WriteProcessMemory(process, reinterpret_cast<LPVOID>(kDataBase), zero.data(), zero.size(), &written);
    WriteProcessMemory(process, reinterpret_cast<LPVOID>(kStackBase), zero.data(), zero.size(), &written);
    WriteProcessMemory(process, reinterpret_cast<LPVOID>(kDataBase), tc.data_seed.data(), tc.data_seed.size(),
                        &written);
    {
      const std::uint64_t retaddr = kCodeBase;  // see gen_jcc's comment
      WriteProcessMemory(process, reinterpret_cast<LPVOID>(kStackTop), &retaddr, sizeof(retaddr), &written);
    }

    if (pending_valid) {
      ContinueDebugEvent(pending_event.dwProcessId, pending_event.dwThreadId, DBG_CONTINUE);
      pending_valid = false;
    }

    bool respawn_after = false;
    // Defensive cap on the "uninteresting event" pump below. Observed once
    // (unreproduced) in a multi-hour run: the process stayed CPU-active
    // (thread state Running, CPU time climbing) but made zero iteration
    // progress for 30+ minutes, never triggering the per-call
    // WaitForDebugEvent timeout below -- consistent with a stream of
    // LOAD_DLL/CREATE_THREAD/etc. events each individually arriving quickly
    // but never reaching the EXCEPTION_DEBUG_EVENT this loop is waiting
    // for. Root cause not identified (didn't reproduce in a clean 241k-case
    // re-run of the same seed), so this converts a silent infinite hang
    // into a recoverable harness error instead of chasing it further.
    int uninteresting_events = 0;
    constexpr int kMaxUninterestingEvents = 10000;
    for (;;) {
      DEBUG_EVENT de{};
      if (!WaitForDebugEvent(&de, kWaitTimeoutMs)) {
        out.setup_ok = false;
        out.setup_error = "hardware: WaitForDebugEvent timeout";
        teardown();
        return out;
      }
      if (de.dwDebugEventCode != EXCEPTION_DEBUG_EVENT) {
        if (++uninteresting_events > kMaxUninterestingEvents) {
          out.setup_ok = false;
          out.setup_error = "hardware: too many non-exception debug events without progress";
          teardown();
          return out;
        }
      }
      switch (de.dwDebugEventCode) {
        case LOAD_DLL_DEBUG_EVENT:
          if (de.u.LoadDll.hFile != nullptr) CloseHandle(de.u.LoadDll.hFile);
          ContinueDebugEvent(de.dwProcessId, de.dwThreadId, DBG_CONTINUE);
          continue;
        case EXIT_THREAD_DEBUG_EVENT:
        case CREATE_THREAD_DEBUG_EVENT:
        case OUTPUT_DEBUG_STRING_EVENT:
        case UNLOAD_DLL_DEBUG_EVENT:
          ContinueDebugEvent(de.dwProcessId, de.dwThreadId, DBG_CONTINUE);
          continue;
        case EXIT_PROCESS_DEBUG_EVENT:
        case RIP_EVENT:
          out.setup_ok = false;
          out.setup_error = "hardware: victim died mid test case";
          ContinueDebugEvent(de.dwProcessId, de.dwThreadId, DBG_CONTINUE);
          teardown();
          return out;
        case EXCEPTION_DEBUG_EVENT: {
          const auto& rec = de.u.Exception.ExceptionRecord;
          const DWORD code = rec.ExceptionCode;
          out.stop = map_exception(rec);
          {
            char buf[96];
            if (code == EXCEPTION_ACCESS_VIOLATION && out.stop == Stop::gp) {
              std::snprintf(buf, sizeof(buf), "gp (access violation, no real fault address) (rip=0x%016llX)",
                            static_cast<unsigned long long>(reinterpret_cast<std::uint64_t>(rec.ExceptionAddress)));
              out.detail = buf;
            } else if (code == EXCEPTION_ACCESS_VIOLATION && rec.NumberParameters >= 2) {
              std::snprintf(buf, sizeof(buf), "av %s at 0x%016llX (rip=0x%016llX)",
                            rec.ExceptionInformation[0] == 8 ? "exec" : rec.ExceptionInformation[0] == 1 ? "write" : "read",
                            static_cast<unsigned long long>(rec.ExceptionInformation[1]),
                            static_cast<unsigned long long>(reinterpret_cast<std::uint64_t>(rec.ExceptionAddress)));
              out.detail = buf;
            } else if (out.stop == Stop::other) {
              std::snprintf(buf, sizeof(buf), "hardware exception 0x%08lX", static_cast<unsigned long>(code));
              out.detail = buf;
            }
          }
          if (out.stop != Stop::ok) respawn_after = true;

          CONTEXT after{};
          after.ContextFlags = CONTEXT_FULL;
          if (GetThreadContext(thread, &after)) {
            read_gpr_from_context(after, out.after);
            read_xmm_from_context(after, out.after);
            read_x87_from_context(after, out.after);
            out.captures_x87 = true;
          }
          SIZE_T got = 0;
          ReadProcessMemory(process, reinterpret_cast<LPCVOID>(kDataBase), out.after.data_after.data(),
                             out.after.data_after.size(), &got);

          pending_event = de;
          pending_valid = true;
          goto classified;
        }
        default:
          ContinueDebugEvent(de.dwProcessId, de.dwThreadId, DBG_CONTINUE);
          continue;
      }
    }
  classified:
    ++steps_since_respawn;
    if (respawn_after || steps_since_respawn >= kRespawnEvery) {
      teardown();
    }
    return out;
  }
};

HardwareSession::HardwareSession() : impl_(std::make_unique<Impl>()) {}
HardwareSession::~HardwareSession() = default;

bool HardwareSession::start(const std::wstring& victim_exe_path, std::string& error) {
  impl_->victim_path = victim_exe_path;
  return impl_->ensure_ready(error);
}

LaneOutcome HardwareSession::run_one(const TestCase& tc) { return impl_->run_one(tc); }

void HardwareSession::shutdown() { impl_->teardown(); }

}  // namespace sf
