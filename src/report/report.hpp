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
[[nodiscard]] std::string write_finding(const std::string& findings_dir, int index, const TestCase& tc,
                                         const LaneOutcome& seven, const LaneOutcome& unicorn,
                                         const LaneOutcome& hw, bool hw_available,
                                         const std::vector<Divergence>& divergences);

}  // namespace sf
