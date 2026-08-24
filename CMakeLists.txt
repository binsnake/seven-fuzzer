cmake_minimum_required(VERSION 3.24)

project(seven_fuzzer LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT Embedded)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

if(NOT WIN32)
  message(FATAL_ERROR "seven-fuzzer's hardware lane is Windows-only (WinAPI debug-event single-stepping).")
endif()
if(MSVC)
  add_compile_options(/EHsc)
endif()

include(FetchContent)

# ---------------------------------------------------------------- seven_core
# Reused read-only from the sibling seven-emulator checkout: same source
# list / defines as its own CMakeLists.txt, minus the gtest-dependent test
# targets this project doesn't need.

set(SEVEN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../seven-emulator" CACHE PATH "Path to the seven-emulator checkout")
set(SOGEN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../sogen" CACHE PATH "Path to the sogen checkout")

if(NOT EXISTS "${SEVEN_ROOT}/seven_core/include/seven/compat.hpp")
  message(FATAL_ERROR "SEVEN_ROOT (${SEVEN_ROOT}) doesn't look like a seven-emulator checkout")
endif()
if(NOT EXISTS "${SOGEN_ROOT}/deps/unicorn/CMakeLists.txt")
  message(FATAL_ERROR "SOGEN_ROOT (${SOGEN_ROOT}) doesn't look like a sogen checkout")
endif()

set(SEVEN_VECTOR_BITS 512)
set(SEVEN_MAX_VECTOR_BYTES 64)

FetchContent_Declare(
  softfloat
  GIT_REPOSITORY https://github.com/ucb-bar/berkeley-softfloat-3.git
  GIT_TAG        master
)
FetchContent_GetProperties(softfloat)
if(NOT softfloat_POPULATED)
  FetchContent_Populate(softfloat)
  file(GLOB SOFTFLOAT_SOURCES_ALL
    "${softfloat_SOURCE_DIR}/source/*.c"
    "${softfloat_SOURCE_DIR}/source/8086/*.c"
  )
  list(FILTER SOFTFLOAT_SOURCES_ALL EXCLUDE REGEX "M\\.c$")
  list(FILTER SOFTFLOAT_SOURCES_ALL EXCLUDE REGEX "M_")
  set(_sf_platform_dir "${CMAKE_BINARY_DIR}/softfloat_platform")
  file(MAKE_DIRECTORY "${_sf_platform_dir}")
  if(MSVC)
    file(WRITE "${_sf_platform_dir}/platform.h" "#define LITTLEENDIAN 1\n#define INLINE __forceinline\n")
  else()
    file(WRITE "${_sf_platform_dir}/platform.h"
      "#define LITTLEENDIAN 1\n#define INLINE static inline\n#define SOFTFLOAT_BUILTIN_CLZ 1\n")
  endif()
  add_library(softfloat STATIC ${SOFTFLOAT_SOURCES_ALL})
  target_include_directories(softfloat PUBLIC
    "${softfloat_SOURCE_DIR}/source/include"
    "${softfloat_SOURCE_DIR}/source/8086")
  target_include_directories(softfloat PRIVATE "${_sf_platform_dir}")
  target_compile_definitions(softfloat PUBLIC SOFTFLOAT_FAST_INT64)
  if(MSVC)
    target_compile_options(softfloat PRIVATE /WX- /wd4146 /wd4244 /wd4267)
  endif()
endif()

set(ICED_X86_SOURCES
  ${SEVEN_ROOT}/iced_x86/src/iced_x86/src/block_encoder.cpp
  ${SEVEN_ROOT}/iced_x86/src/iced_x86/src/decoder.cpp
  ${SEVEN_ROOT}/iced_x86/src/iced_x86/src/encoder.cpp
  ${SEVEN_ROOT}/iced_x86/src/iced_x86/src/encoder_handlers.cpp
  ${SEVEN_ROOT}/iced_x86/src/iced_x86/src/encoder_methods.cpp
  ${SEVEN_ROOT}/iced_x86/src/iced_x86/src/encoder_ops.cpp
  ${SEVEN_ROOT}/iced_x86/src/iced_x86/src/fastfmt.cpp
  ${SEVEN_ROOT}/iced_x86/src/iced_x86/src/gas_formatter.cpp
  ${SEVEN_ROOT}/iced_x86/src/iced_x86/src/handlers.cpp
  ${SEVEN_ROOT}/iced_x86/src/iced_x86/src/instruction.cpp
  ${SEVEN_ROOT}/iced_x86/src/iced_x86/src/instruction_create.cpp
  ${SEVEN_ROOT}/iced_x86/src/iced_x86/src/instruction_info.cpp
  ${SEVEN_ROOT}/iced_x86/src/iced_x86/src/intel_formatter.cpp
  ${SEVEN_ROOT}/iced_x86/src/iced_x86/src/masm_formatter.cpp
  ${SEVEN_ROOT}/iced_x86/src/iced_x86/src/memory_size_info.cpp
  ${SEVEN_ROOT}/iced_x86/src/iced_x86/src/mvex_info_data.cpp
  ${SEVEN_ROOT}/iced_x86/src/iced_x86/src/nasm_formatter.cpp
  ${SEVEN_ROOT}/iced_x86/src/iced_x86/src/op_code_info.cpp
  ${SEVEN_ROOT}/iced_x86/src/iced_x86/src/register_info.cpp
  ${SEVEN_ROOT}/iced_x86/src/iced_x86/src/table_deserializer.cpp
  ${SEVEN_ROOT}/iced_x86/src/iced_x86/src/tables.cpp
)
add_library(iced_x86 STATIC ${ICED_X86_SOURCES})
target_include_directories(iced_x86 PUBLIC "${SEVEN_ROOT}/iced_x86/include")
target_compile_definitions(iced_x86 PUBLIC _CRT_SECURE_NO_WARNINGS ICED_X86_CONSTEXPR_HANDLERS=0)
if(MSVC)
  target_compile_options(iced_x86 PRIVATE /WX-)
endif()

file(GLOB_RECURSE SEVEN_CORE_SOURCES CONFIGURE_DEPENDS "${SEVEN_ROOT}/seven_core/src/*.cpp")
add_library(seven_core STATIC ${SEVEN_CORE_SOURCES})
target_include_directories(seven_core PUBLIC "${SEVEN_ROOT}/seven_core/include")
target_include_directories(seven_core PUBLIC "${SEVEN_ROOT}")
target_link_libraries(seven_core PUBLIC iced_x86 softfloat)
target_compile_definitions(seven_core PUBLIC
  SEVEN_SIMD_PROFILE_AVX512=1
  SEVEN_ENABLE_AVX=1
  SEVEN_ENABLE_AVX2=1
  SEVEN_ENABLE_AVX512=1
  SEVEN_VECTOR_BITS=${SEVEN_VECTOR_BITS}
  SEVEN_MAX_VECTOR_BYTES=${SEVEN_MAX_VECTOR_BYTES}
  SOFTFLOAT_FAST_INT64
)
if(MSVC)
  target_compile_options(seven_core PRIVATE /WX-)
endif()

# -------------------------------------------------------------------- unicorn
# Built from sogen's own vendored submodule/commit so the "unicorn" lane
# matches what sogen's default backend actually runs, not an unrelated
# prebuilt version.

set(UNICORN_ARCH "x86" CACHE STRING "" FORCE)
set(UNICORN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(UNICORN_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory("${SOGEN_ROOT}/deps/unicorn" "${CMAKE_BINARY_DIR}/unicorn_build")

# --------------------------------------------------------------- seven_fuzzer

add_executable(seven_fuzz_victim src/victim/victim_stub.cpp)
if(MSVC)
  target_compile_options(seven_fuzz_victim PRIVATE /WX-)
endif()

add_executable(seven_fuzzer
  src/main.cpp
  src/lanes/seven_lane.cpp
  src/lanes/unicorn_lane.cpp
  src/lanes/hardware_lane.cpp
  src/gen/instruction_gen.cpp
  src/report/report.cpp
)
target_include_directories(seven_fuzzer PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(seven_fuzzer PRIVATE seven_core unicorn advapi32)
add_dependencies(seven_fuzzer seven_fuzz_victim)
if(MSVC)
  target_compile_options(seven_fuzzer PRIVATE /WX- /Zc:__cplusplus /permissive-)
  # main.cpp uses _set_se_translator to catch structured exceptions (access
  # violations etc.) from worker threads as C++ exceptions -- that only
  # works under the asynchronous EH model. The rest of the project stays on
  # the default /EHsc (project-wide, see top of this file); /EHa here
  # overrides it for just this one translation unit, which MSVC supports
  # mixed with /EHsc elsewhere in the same binary.
  set_source_files_properties(src/main.cpp PROPERTIES COMPILE_OPTIONS "/EHa")
endif()
