#pragma once

#include "common/types.hpp"

namespace sf {

// Runs one TestCase through seven_core directly (in-process, no sandboxing
// needed — it's a pure software model, not real code execution).
[[nodiscard]] LaneOutcome run_seven(const TestCase& tc);

}  // namespace sf
