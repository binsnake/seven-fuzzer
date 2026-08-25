#include "report/report.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>

namespace sf {

namespace {

void append_outcome(std::string& out, const char* label, const LaneOutcome& o) {
  char buf[256];
  out += std::string("[") + label + "]\n";
  if (!o.setup_ok) {
    out += "  harness error: " + o.setup_error + "\n";
    return;
  }
  std::snprintf(buf, sizeof(buf), "  stop=%s%s%s\n", stop_name(o.stop), o.detail.empty() ? "" : " detail=",
                o.detail.c_str());
  out += buf;
  static const char* kNames[16] = {"rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
                                    "r8",  "r9",  "r10", "r11", "r12", "r13", "r14", "r15"};
  out += "  ";
  for (int i = 0; i < 16; ++i) {
    std::snprintf(buf, sizeof(buf), "%s=%016llX ", kNames[i], static_cast<unsigned long long>(o.after.gpr[i]));
    out += buf;
    if (i == 7) out += "\n  ";
  }
  out += "\n";
  std::snprintf(buf, sizeof(buf), "  rip=%016llX rflags=%016llX (compare-masked=%03llX)\n",
                static_cast<unsigned long long>(o.after.rip), static_cast<unsigned long long>(o.after.rflags),
                static_cast<unsigned long long>(o.after.rflags & kCompareFlagsMask));
  out += buf;
  for (int i = 0; i < 16; ++i) {
    std::snprintf(buf, sizeof(buf), "  xmm%-2d=%016llX%016llX\n", i,
                  static_cast<unsigned long long>(o.after.xmm_hi[static_cast<std::size_t>(i)]),
                  static_cast<unsigned long long>(o.after.xmm_lo[static_cast<std::size_t>(i)]));
    out += buf;
  }
}

}  // namespace

std::string write_finding(const std::string& findings_dir, int index, const TestCase& tc, const char* seven_label,
                           const LaneOutcome& seven, const LaneOutcome& unicorn, const LaneOutcome& hw,
                           bool hw_available, const std::vector<Divergence>& divergences) {
  std::filesystem::create_directories(findings_dir);
  char name[64];
  std::snprintf(name, sizeof(name), "/finding_%06d.txt", index);
  const std::string path = findings_dir + name;

  std::string out;
  out += "bytes: " + tc.text + "\n";
  out += "touches_memory: " + std::string(tc.touches_memory ? "yes" : "no") + "\n";
  out += "initial:\n  ";
  static const char* kNames[16] = {"rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
                                    "r8",  "r9",  "r10", "r11", "r12", "r13", "r14", "r15"};
  char buf[256];
  for (int i = 0; i < 16; ++i) {
    std::snprintf(buf, sizeof(buf), "%s=%016llX ", kNames[i], static_cast<unsigned long long>(tc.initial.gpr[i]));
    out += buf;
    if (i == 7) out += "\n  ";
  }
  out += "\n";
  std::snprintf(buf, sizeof(buf), "  rflags=%016llX\n", static_cast<unsigned long long>(tc.initial.rflags));
  out += buf;
  for (int i = 0; i < 16; ++i) {
    std::snprintf(buf, sizeof(buf), "  xmm%-2d=%016llX%016llX\n", i,
                  static_cast<unsigned long long>(tc.initial.xmm_hi[static_cast<std::size_t>(i)]),
                  static_cast<unsigned long long>(tc.initial.xmm_lo[static_cast<std::size_t>(i)]));
    out += buf;
  }
  out += "\n";

  append_outcome(out, seven_label, seven);
  append_outcome(out, "unicorn", unicorn);
  if (hw_available) {
    append_outcome(out, "hardware", hw);
  } else {
    out += "[hardware]\n  (not compared this round — harness unavailable)\n";
  }

  out += "\ndivergences:\n";
  for (const auto& d : divergences) {
    out += "  " + d.pairing + ":\n";
    for (const auto& line : d.lines) out += "    " + line + "\n";
  }

  std::ofstream f(path, std::ios::binary);
  f << out;
  return path;
}

}  // namespace sf
