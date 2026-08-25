#pragma once

#include <string>
#include <vector>

#include "common/types.hpp"

namespace sf {

struct Divergence {
  std::string pairing;      // "seven-vs-hw", "unicorn-vs-hw", "seven-vs-unicorn"
  std::vector<std::string> lines;
};

// Writes a full repro report (bytes, initial state, all three outcomes, the
// specific diverging fields) to findings/finding_<n>.txt. Returns the path.
// seven_label is "seven" or "seven-jit" depending on which engine produced `seven`, so a report
// triaged later says which one actually ran without the reader having to know how this fuzzer run
// was invoked.
[[nodiscard]] std::string write_finding(const std::string& findings_dir, int index, const TestCase& tc,
                                         const char* seven_label, const LaneOutcome& seven,
                                         const LaneOutcome& unicorn, const LaneOutcome& hw, bool hw_available,
                                         const std::vector<Divergence>& divergences);

}  // namespace sf
