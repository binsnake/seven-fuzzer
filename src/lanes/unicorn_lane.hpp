#pragma once

#include "common/types.hpp"

namespace sf {

// Runs one TestCase through Unicorn (built from sogen's vendored submodule,
// not pip's prebuilt, so this matches what sogen's default backend actually
// runs). In-process; Unicorn never touches real memory or real syscalls.
[[nodiscard]] LaneOutcome run_unicorn(const TestCase& tc);

}  // namespace sf
