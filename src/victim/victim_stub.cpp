// Disposable debuggee for the hardware lane (see lanes/hardware_lane.cpp).
//
// This process never runs any of its own logic in practice: the debugger
// (seven_fuzzer.exe) takes over at the automatic loader breakpoint that
// Windows delivers before main() ever runs, and from then on fully owns
// RIP/RSP/registers/memory for every test case. The loop below only exists
// as a fallback in case something ever resumes this thread without
// redirecting it — it should be unreachable in normal operation.

#include <windows.h>

int main() {
  for (;;) {
    __debugbreak();
  }
}
