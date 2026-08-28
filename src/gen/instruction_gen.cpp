#include "gen/instruction_gen.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <intrin.h>
#include <optional>
#include <vector>

#include "common/host_caps.hpp"

#include <iced_x86/code.hpp>
#include <iced_x86/encoder.hpp>
#include <iced_x86/instruction.hpp>
#include <iced_x86/instruction_create.hpp>
#include <iced_x86/memory_operand.hpp>
#include <iced_x86/op_kind.hpp>
#include <iced_x86/register.hpp>
#include <iced_x86/rep_prefix_kind.hpp>

namespace sf {

namespace {

using iced_x86::Code;
using iced_x86::Encoder;
using iced_x86::Instruction;
using iced_x86::InstructionFactory;
using iced_x86::MemoryOperand;
using iced_x86::OpKind;
using iced_x86::Register;

enum class Width : int { W8 = 0, W16 = 1, W32 = 2, W64 = 3 };

constexpr std::array<Register, 16> kRegs8 = {
    Register::AL,   Register::CL,   Register::DL,   Register::BL,
    Register::SPL,  Register::BPL,  Register::SIL,  Register::DIL,
    Register::R8_L, Register::R9_L, Register::R10_L, Register::R11_L,
    Register::R12_L, Register::R13_L, Register::R14_L, Register::R15_L,
};
constexpr std::array<Register, 16> kRegs16 = {
    Register::AX,   Register::CX,   Register::DX,   Register::BX,
    Register::SP,   Register::BP,   Register::SI,   Register::DI,
    Register::R8_W, Register::R9_W, Register::R10_W, Register::R11_W,
    Register::R12_W, Register::R13_W, Register::R14_W, Register::R15_W,
};
constexpr std::array<Register, 16> kRegs32 = {
    Register::EAX,  Register::ECX,  Register::EDX,  Register::EBX,
    Register::ESP,  Register::EBP,  Register::ESI,  Register::EDI,
    Register::R8_D, Register::R9_D, Register::R10_D, Register::R11_D,
    Register::R12_D, Register::R13_D, Register::R14_D, Register::R15_D,
};
constexpr std::array<Register, 16> kRegs64 = {
    Register::RAX, Register::RCX, Register::RDX, Register::RBX,
    Register::RSP, Register::RBP, Register::RSI, Register::RDI,
    Register::R8,  Register::R9,  Register::R10, Register::R11,
    Register::R12, Register::R13, Register::R14, Register::R15,
};

[[nodiscard]] Register reg_of(Width w, int idx) {
  switch (w) {
    case Width::W8: return kRegs8[static_cast<std::size_t>(idx)];
    case Width::W16: return kRegs16[static_cast<std::size_t>(idx)];
    case Width::W32: return kRegs32[static_cast<std::size_t>(idx)];
    default: return kRegs64[static_cast<std::size_t>(idx)];
  }
}

[[nodiscard]] int bits_of(Width w) {
  switch (w) {
    case Width::W8: return 8;
    case Width::W16: return 16;
    case Width::W32: return 32;
    default: return 64;
  }
}

struct Ctx {
  std::mt19937_64& rng;
  bool touches_memory = false;
  std::uint64_t flags_mask = kCompareFlagsMask;
  // Lets a family override specific GPRs after the generic per-register
  // randomization (e.g. DIV/IDIV biasing the dividend/divisor toward an
  // in-range case instead of the near-certain #DE a fully random pair
  // would produce). Index is x86 register numbering, same as RegState::gpr.
  std::array<std::optional<std::uint64_t>, 16> force_gpr{};
  // BSF/BSR-specific: recorded so next() can check the ACTUAL runtime source
  // value after registers/data are randomized and clear gpr_compare_mask's
  // destination bit only when that source turns out to be zero. See
  // TestCase::gpr_compare_mask's comment for why.
  std::optional<int> bsf_bsr_dest;
  std::optional<int> bsf_bsr_src_reg;
  std::optional<std::int8_t> bsf_bsr_src_mem_disp;
  int bsf_bsr_width_bytes = 0;
  // Plants a known qword in the scratch page after next() has randomized it. Memory-indirect
  // branches read their target from there, and a random one would land outside the mapped code
  // page -- see the branch-target comment above gen_jcc for why that isn't comparable.
  struct ForcedQword {
    std::size_t offset;
    std::uint64_t value;
  };
  std::optional<ForcedQword> force_data_qword;
  // Same idea, any width up to the 10 bytes an m80 operand needs. The x87 memory forms need it
  // because a uniformly random 4/8/10-byte pattern reads as a NaN or an unnormal nearly every
  // time, and the ordinary finite values are where the rounding paths actually get exercised.
  struct ForcedBytes {
    std::size_t offset;
    std::size_t size;
    std::array<std::uint8_t, 10> value;
  };
  std::optional<ForcedBytes> force_data_bytes;
  // Set by the x87 families. next() then builds a whole starting FPU state -- stack contents, TOP,
  // tags, control word -- instead of leaving the default empty stack that every one of them would
  // otherwise just underflow straight through.
  bool uses_x87 = false;
  int x87_needs = 0;  // ST slots the instruction reads; see next() for how the stack depth is picked
};

[[nodiscard]] int rand_int(std::mt19937_64& rng, int lo, int hi) {
  std::uniform_int_distribution<int> d(lo, hi);
  return d(rng);
}

[[nodiscard]] std::uint64_t random_interesting_u64(std::mt19937_64& rng) {
  static constexpr std::uint64_t kPool[] = {
      0, 1, 2, 3, 0x7Full, 0x80ull, 0xFFull, 0x100ull,
      0x7FFFull, 0x8000ull, 0xFFFFull, 0x10000ull,
      0x7FFFFFFFull, 0x80000000ull, 0xFFFFFFFFull, 0x100000000ull,
      0x7FFFFFFFFFFFFFFFull, 0x8000000000000000ull, 0xFFFFFFFFFFFFFFFFull,
      0xAAAAAAAAAAAAAAAAull, 0x5555555555555555ull,
  };
  if (rand_int(rng, 0, 99) < 40) {
    std::uint64_t v = kPool[static_cast<std::size_t>(rand_int(rng, 0, static_cast<int>(std::size(kPool)) - 1))];
    if (rand_int(rng, 0, 99) < 30) {
      v = static_cast<std::uint64_t>(static_cast<std::int64_t>(v) + rand_int(rng, -2, 2));
    }
    return v;
  }
  std::uniform_int_distribution<std::uint64_t> full(0, UINT64_MAX);
  return full(rng);
}

[[nodiscard]] std::uint64_t random_imm(std::mt19937_64& rng, int bits) {
  std::uint64_t v = random_interesting_u64(rng);
  if (bits < 64) v &= (std::uint64_t{1} << bits) - 1;
  return v;
}

[[nodiscard]] std::int8_t random_disp8(std::mt19937_64& rng) {
  // Non-negative only: kDataBase sits at the start of its page, so a
  // negative displacement here would spill into the code page instead of
  // staying inside the compared kDataWindow.
  // rand_int is inclusive on both ends -- 0x80 (128) doesn't fit in a
  // signed int8_t and wraps to -128, which silently reintroduces the
  // negative-displacement bug this function exists to avoid. Exclusive
  // upper bound: [0, 0x7F].
  return static_cast<std::int8_t>(rand_int(rng, 0, 0x7F));
}

// Same window and sign constraints as random_disp8, snapped down to a multiple of `alignment`.
// kDataBase is page-aligned, so an aligned displacement gives an aligned effective address -- which
// is what the legacy SIMD m128 forms need to get past their #GP check and actually exercise the
// operation rather than the check.
[[nodiscard]] std::int8_t aligned_disp8(std::mt19937_64& rng, int alignment) {
  const int raw = rand_int(rng, 0, 0x7F);
  return static_cast<std::int8_t>(raw - (raw % alignment));
}

[[nodiscard]] int pick_reg_index(std::mt19937_64& rng) { return rand_int(rng, 0, 15); }

[[nodiscard]] MemoryOperand mem_operand(std::int8_t disp) {
  return MemoryOperand::with_base_displ_size(Register::RDI, disp, 1);
}

// Somewhere inside the code page, for anything that has to branch to a real mapped address.
[[nodiscard]] std::uint64_t random_code_target(std::mt19937_64& rng) {
  return kCodeBase + static_cast<std::uint64_t>(rand_int(rng, 0, static_cast<int>(kPageSize) - 16));
}

// Picks a qword-aligned slot in the scratch window and plants a code-page address there, for the
// memory-indirect JMP/CALL forms. Returns the displacement to address it with.
[[nodiscard]] std::int8_t indirect_target_disp(Ctx& c) {
  // Capped at 120, not the window size: the displacement rides in a signed byte, same constraint
  // random_disp8 documents.
  const auto offset = static_cast<std::size_t>(rand_int(c.rng, 0, 15)) * 8;
  c.force_data_qword = Ctx::ForcedQword{offset, random_code_target(c.rng)};
  return static_cast<std::int8_t>(offset);
}

[[nodiscard]] Width width16_32_64(std::mt19937_64& rng) {
  const int p = rand_int(rng, 0, 2);
  return p == 0 ? Width::W16 : p == 1 ? Width::W32 : Width::W64;
}
[[nodiscard]] int widx16_32_64(Width w) { return w == Width::W16 ? 0 : w == Width::W32 ? 1 : 2; }

// InstructionFactory's integer overloads always build the immediate as OpKind::IMMEDIATE32, and
// the encoder refuses anything but the exact kind the opcode wants ("Expected OpKind N, actual
// OpKind 9"). next()'s catch-and-retry swallowed those failures, so every family below looked
// like it was covering its immediate forms while actually only ever emitting the one code that
// genuinely takes an imm32. Stamping the right kind is what makes the rest reachable: the
// sign-extended imm8 ALU forms, every 8/16/64-bit immediate, and all the shift-by-imm8 codes.
void set_imm_kind(Instruction& instr, std::uint32_t index, OpKind kind) {
  switch (index) {
    case 0: instr.set_op0_kind(kind); break;
    case 1: instr.set_op1_kind(kind); break;
    default: instr.set_op2_kind(kind); break;
  }
}

// A plain 8-bit immediate that is not widened: shift counts, bit offsets, SIMD selectors.
void set_imm8(Instruction& instr, std::uint32_t index, std::uint64_t value) {
  set_imm_kind(instr, index, OpKind::IMMEDIATE8);
  instr.set_immediate8(static_cast<std::uint8_t>(value));
}

// An immediate that is part of an operand-sized form. `sign_extended_imm8` selects the short
// encoding (the 83 /r family and its relatives), which the operand width then widens.
void set_imm_sized(Instruction& instr, std::uint32_t index, Width w, bool sign_extended_imm8,
                   std::uint64_t value) {
  if (sign_extended_imm8) {
    const auto byte = static_cast<std::int8_t>(value);
    switch (w) {
      case Width::W16:
        set_imm_kind(instr, index, OpKind::IMMEDIATE8TO16);
        instr.set_immediate8to16(static_cast<std::int16_t>(byte));
        return;
      case Width::W32:
        set_imm_kind(instr, index, OpKind::IMMEDIATE8TO32);
        instr.set_immediate8to32(static_cast<std::int32_t>(byte));
        return;
      default:
        set_imm_kind(instr, index, OpKind::IMMEDIATE8TO64);
        instr.set_immediate8to64(static_cast<std::int64_t>(byte));
        return;
    }
  }
  switch (w) {
    case Width::W8:
      set_imm_kind(instr, index, OpKind::IMMEDIATE8);
      instr.set_immediate8(static_cast<std::uint8_t>(value));
      return;
    case Width::W16:
      set_imm_kind(instr, index, OpKind::IMMEDIATE16);
      instr.set_immediate16(static_cast<std::uint16_t>(value));
      return;
    case Width::W32:
      set_imm_kind(instr, index, OpKind::IMMEDIATE32);
      instr.set_immediate32(static_cast<std::uint32_t>(value));
      return;
    default:
      set_imm_kind(instr, index, OpKind::IMMEDIATE32TO64);
      instr.set_immediate32to64(static_cast<std::int64_t>(static_cast<std::int32_t>(value)));
      return;
  }
}

// ---------------------------------------------------------------- ALU/TEST

struct AluCodes {
  std::array<Code, 4> rm_r;
  std::array<Code, 4> r_rm;
  std::array<Code, 4> rm_imm8;
  std::array<Code, 4> rm_immfull;
  // The one-byte accumulator forms (04 ib, 0C ib, ...). Same semantics as the r/m,imm forms
  // but a different encoding and a different Code, so they need asking for by name.
  std::array<Code, 4> acc_imm;
};

#define SF_ALU(OP)                                                                                     \
  AluCodes {                                                                                            \
    {Code::OP##_RM8_R8, Code::OP##_RM16_R16, Code::OP##_RM32_R32, Code::OP##_RM64_R64},                 \
    {Code::OP##_R8_RM8, Code::OP##_R16_RM16, Code::OP##_R32_RM32, Code::OP##_R64_RM64},                 \
    {Code::OP##_RM8_IMM8, Code::OP##_RM16_IMM8, Code::OP##_RM32_IMM8, Code::OP##_RM64_IMM8},             \
    {Code::OP##_RM8_IMM8, Code::OP##_RM16_IMM16, Code::OP##_RM32_IMM32, Code::OP##_RM64_IMM32},          \
    {Code::OP##_AL_IMM8, Code::OP##_AX_IMM16, Code::OP##_EAX_IMM32, Code::OP##_RAX_IMM32},               \
  }

const std::array<AluCodes, 8> kAluOps = {
    SF_ALU(ADD), SF_ALU(OR), SF_ALU(ADC), SF_ALU(SBB), SF_ALU(AND), SF_ALU(SUB), SF_ALU(XOR), SF_ALU(CMP),
};
#undef SF_ALU

[[nodiscard]] std::optional<Instruction> gen_alu(Ctx& c) {
  const AluCodes& t = kAluOps[static_cast<std::size_t>(rand_int(c.rng, 0, 7))];
  const Width w = static_cast<Width>(rand_int(c.rng, 0, 3));
  const int wi = static_cast<int>(w);
  const int form = rand_int(c.rng, 0, 3);  // 0=reg,reg  1=reg,imm  2=memory  3=accumulator,imm

  if (form == 3) {
    const Register acc = w == Width::W8    ? Register::AL
                         : w == Width::W16 ? Register::AX
                         : w == Width::W32 ? Register::EAX
                                           : Register::RAX;
    const std::int32_t imm =
        static_cast<std::int32_t>(random_imm(c.rng, w == Width::W8 ? 8 : w == Width::W16 ? 16 : 32));
    auto instr = InstructionFactory::with2(t.acc_imm[static_cast<std::size_t>(wi)], acc, imm);
    set_imm_sized(instr, 1, w, false, static_cast<std::uint64_t>(imm));
    return instr;
  }

  if (form == 0) {
    const int a = pick_reg_index(c.rng);
    const int b = pick_reg_index(c.rng);
    return InstructionFactory::with2(t.rm_r[static_cast<std::size_t>(wi)], reg_of(w, a), reg_of(w, b));
  }
  if (form == 1) {
    const int a = pick_reg_index(c.rng);
    const bool short_imm = w != Width::W8 && rand_int(c.rng, 0, 1) == 0;
    const Code code = short_imm ? t.rm_imm8[static_cast<std::size_t>(wi)] : t.rm_immfull[static_cast<std::size_t>(wi)];
    const std::int32_t imm = short_imm
        ? static_cast<std::int32_t>(static_cast<std::int8_t>(random_imm(c.rng, 8)))
        : static_cast<std::int32_t>(random_imm(c.rng, w == Width::W8 ? 8 : w == Width::W16 ? 16 : 32));
    auto instr = InstructionFactory::with2(code, reg_of(w, a), imm);
    set_imm_sized(instr, 1, w, short_imm, static_cast<std::uint64_t>(imm));
    return instr;
  }

  c.touches_memory = true;
  const std::int8_t disp = random_disp8(c.rng);
  if (rand_int(c.rng, 0, 1) == 0) {
    const bool short_imm = w != Width::W8 && rand_int(c.rng, 0, 1) == 0;
    const Code code = short_imm ? t.rm_imm8[static_cast<std::size_t>(wi)] : t.rm_immfull[static_cast<std::size_t>(wi)];
    const std::int32_t imm = short_imm
        ? static_cast<std::int32_t>(static_cast<std::int8_t>(random_imm(c.rng, 8)))
        : static_cast<std::int32_t>(random_imm(c.rng, w == Width::W8 ? 8 : w == Width::W16 ? 16 : 32));
    auto instr = InstructionFactory::with2(code, mem_operand(disp), imm);
    set_imm_sized(instr, 1, w, short_imm, static_cast<std::uint64_t>(imm));
    return instr;
  }
  const int r = pick_reg_index(c.rng);
  if (rand_int(c.rng, 0, 1) == 0) {
    return InstructionFactory::with2(t.rm_r[static_cast<std::size_t>(wi)], mem_operand(disp), reg_of(w, r));
  }
  return InstructionFactory::with2(t.r_rm[static_cast<std::size_t>(wi)], reg_of(w, r), mem_operand(disp));
}

struct TestCodes {
  std::array<Code, 4> rm_r;
  std::array<Code, 4> rm_imm;
};
const TestCodes kTest = {
    {Code::TEST_RM8_R8, Code::TEST_RM16_R16, Code::TEST_RM32_R32, Code::TEST_RM64_R64},
    {Code::TEST_RM8_IMM8, Code::TEST_RM16_IMM16, Code::TEST_RM32_IMM32, Code::TEST_RM64_IMM32},
};
// F6/F7 /1 is a second, undocumented-but-real encoding of TEST r/m, imm that decodes to its own
// Code. Silicon treats it exactly like /0, which is worth confirming rather than assuming.
constexpr std::array<Code, 4> kTestImmF7R1 = {Code::TEST_RM8_IMM8_F6R1, Code::TEST_RM16_IMM16_F7R1,
                                               Code::TEST_RM32_IMM32_F7R1, Code::TEST_RM64_IMM32_F7R1};
constexpr std::array<Code, 4> kTestAccImm = {Code::TEST_AL_IMM8, Code::TEST_AX_IMM16,
                                              Code::TEST_EAX_IMM32, Code::TEST_RAX_IMM32};

[[nodiscard]] std::optional<Instruction> gen_test(Ctx& c) {
  const Width w = static_cast<Width>(rand_int(c.rng, 0, 3));
  const int wi = static_cast<int>(w);
  const int imm_bits = w == Width::W8 ? 8 : w == Width::W16 ? 16 : 32;
  const int form = rand_int(c.rng, 0, 3);

  if (form == 3) {
    const std::int32_t imm = static_cast<std::int32_t>(random_imm(c.rng, imm_bits));
    const bool accumulator = rand_int(c.rng, 0, 1) == 0;
    const auto wu = static_cast<std::size_t>(wi);
    auto instr = accumulator
        ? InstructionFactory::with2(kTestAccImm[wu],
                                    w == Width::W8    ? Register::AL
                                    : w == Width::W16 ? Register::AX
                                    : w == Width::W32 ? Register::EAX
                                                      : Register::RAX,
                                    imm)
        : InstructionFactory::with2(kTestImmF7R1[wu], reg_of(w, pick_reg_index(c.rng)), imm);
    set_imm_sized(instr, 1, w, false, static_cast<std::uint64_t>(imm));
    return instr;
  }
  if (form == 0) {
    const int a = pick_reg_index(c.rng), b = pick_reg_index(c.rng);
    return InstructionFactory::with2(kTest.rm_r[static_cast<std::size_t>(wi)], reg_of(w, a), reg_of(w, b));
  }
  if (form == 1) {
    const int a = pick_reg_index(c.rng);
    const std::int32_t imm = static_cast<std::int32_t>(random_imm(c.rng, imm_bits));
    auto instr = InstructionFactory::with2(kTest.rm_imm[static_cast<std::size_t>(wi)], reg_of(w, a), imm);
    set_imm_sized(instr, 1, w, false, static_cast<std::uint64_t>(imm));
    return instr;
  }
  c.touches_memory = true;
  const std::int8_t disp = random_disp8(c.rng);
  if (rand_int(c.rng, 0, 1) == 0) {
    const int b = pick_reg_index(c.rng);
    return InstructionFactory::with2(kTest.rm_r[static_cast<std::size_t>(wi)], mem_operand(disp), reg_of(w, b));
  }
  const std::int32_t imm = static_cast<std::int32_t>(random_imm(c.rng, imm_bits));
  auto test_mem = InstructionFactory::with2(kTest.rm_imm[static_cast<std::size_t>(wi)], mem_operand(disp), imm);
  set_imm_sized(test_mem, 1, w, false, static_cast<std::uint64_t>(imm));
  return test_mem;
}

// ------------------------------------------------------------- INC/DEC/etc

struct Unary1Codes {
  std::array<Code, 4> rm;
};
#define SF_UN(OP) Unary1Codes{{Code::OP##_RM8, Code::OP##_RM16, Code::OP##_RM32, Code::OP##_RM64}}
const std::array<Unary1Codes, 4> kUnaryOps = {SF_UN(INC), SF_UN(DEC), SF_UN(NEG), SF_UN(NOT)};
#undef SF_UN

[[nodiscard]] std::optional<Instruction> gen_unary(Ctx& c) {
  const Unary1Codes& t = kUnaryOps[static_cast<std::size_t>(rand_int(c.rng, 0, 3))];
  const Width w = static_cast<Width>(rand_int(c.rng, 0, 3));
  if (rand_int(c.rng, 0, 1) == 0) {
    const int a = pick_reg_index(c.rng);
    return InstructionFactory::with1(t.rm[static_cast<std::size_t>(w)], reg_of(w, a));
  }
  c.touches_memory = true;
  return InstructionFactory::with1(t.rm[static_cast<std::size_t>(w)], mem_operand(random_disp8(c.rng)));
}

// ---------------------------------------------------------------- shift/rot

struct ShiftCodes {
  std::array<Code, 4> imm8;
  std::array<Code, 4> cl;
  // The D1 /r forms, where the count of 1 is part of the opcode rather than an immediate byte.
  // They are the only shifts that leave OF architecturally defined.
  std::array<Code, 4> one;
};
#define SF_SH(OP)                                                                                \
  ShiftCodes {                                                                                     \
    {Code::OP##_RM8_IMM8, Code::OP##_RM16_IMM8, Code::OP##_RM32_IMM8, Code::OP##_RM64_IMM8},        \
    {Code::OP##_RM8_CL, Code::OP##_RM16_CL, Code::OP##_RM32_CL, Code::OP##_RM64_CL},                \
    {Code::OP##_RM8_1, Code::OP##_RM16_1, Code::OP##_RM32_1, Code::OP##_RM64_1},                    \
  }
// SAL is the /4 alias of SHL and decodes to its own Code, so it needs its own entry to be reached.
const std::array<ShiftCodes, 8> kShiftOps = {
    SF_SH(SHL), SF_SH(SHR), SF_SH(SAR), SF_SH(ROL), SF_SH(ROR), SF_SH(RCL), SF_SH(RCR), SF_SH(SAL),
};
#undef SF_SH

[[nodiscard]] std::optional<Instruction> gen_shift(Ctx& c) {
  // OF is only architecturally defined for single-bit shifts/rotates; for
  // multi-bit counts it's undefined, so don't compare it (the actual count
  // isn't known until CL's randomized value is picked, so this excludes OF
  // unconditionally rather than trying to predict count==1).
  const ShiftCodes& t = kShiftOps[static_cast<std::size_t>(rand_int(c.rng, 0, 7))];
  const Width w = static_cast<Width>(rand_int(c.rng, 0, 3));
  const int wi = static_cast<int>(w);
  const int count_form = rand_int(c.rng, 0, 2);  // 0=imm8  1=CL  2=the shift-by-one opcode
  const bool use_one = count_form == 2;
  const bool use_cl = count_form == 1;
  const bool use_mem = rand_int(c.rng, 0, 3) == 0;
  // OF is only defined for a single-bit shift or rotate; for a multi-bit count it is undefined,
  // and CL's value isn't known until it is randomized, so it can only be compared on the
  // shift-by-one form.
  if (!use_one) c.flags_mask &= ~0x0800ull;

  if (!use_mem) {
    const int a = pick_reg_index(c.rng);
    if (use_one) {
      // iced still models the count as a second operand for these, it just has to be exactly 1.
      auto one = InstructionFactory::with2(t.one[static_cast<std::size_t>(wi)], reg_of(w, a), 1);
      set_imm8(one, 1, 1);
      return one;
    }
    if (use_cl) return InstructionFactory::with2(t.cl[static_cast<std::size_t>(wi)], reg_of(w, a), Register::CL);
    std::int32_t imm = static_cast<std::int32_t>(random_imm(c.rng, 8));
    if (rand_int(c.rng, 0, 4) != 0) imm &= 0x3F;  // usually in-range; sometimes probe masking of the high bits
    auto instr = InstructionFactory::with2(t.imm8[static_cast<std::size_t>(wi)], reg_of(w, a), imm);
    set_imm8(instr, 1, static_cast<std::uint64_t>(imm));
    return instr;
  }
  c.touches_memory = true;
  const std::int8_t disp = random_disp8(c.rng);
  if (use_one) {
    auto one_mem = InstructionFactory::with2(t.one[static_cast<std::size_t>(wi)], mem_operand(disp), 1);
    set_imm8(one_mem, 1, 1);
    return one_mem;
  }
  if (use_cl) return InstructionFactory::with2(t.cl[static_cast<std::size_t>(wi)], mem_operand(disp), Register::CL);
  const std::int32_t imm = static_cast<std::int32_t>(random_imm(c.rng, 8));
  auto shift_mem = InstructionFactory::with2(t.imm8[static_cast<std::size_t>(wi)], mem_operand(disp), imm);
  set_imm8(shift_mem, 1, static_cast<std::uint64_t>(imm));
  return shift_mem;
}

// -------------------------------------------------------------------- MOV

struct MovCodes {
  std::array<Code, 4> rm_r;
  std::array<Code, 4> r_rm;
  std::array<Code, 4> rm_imm;  // width64 entry is imm32 sign-extended, not imm64
};
const MovCodes kMov = {
    {Code::MOV_RM8_R8, Code::MOV_RM16_R16, Code::MOV_RM32_R32, Code::MOV_RM64_R64},
    {Code::MOV_R8_RM8, Code::MOV_R16_RM16, Code::MOV_R32_RM32, Code::MOV_R64_RM64},
    {Code::MOV_RM8_IMM8, Code::MOV_RM16_IMM16, Code::MOV_RM32_IMM32, Code::MOV_RM64_IMM32},
};

[[nodiscard]] std::optional<Instruction> gen_mov(Ctx& c) {
  const Width w = static_cast<Width>(rand_int(c.rng, 0, 3));
  const int wi = static_cast<int>(w);
  const int form = rand_int(c.rng, 0, 3);

  if (form == 0) {
    const int a = pick_reg_index(c.rng), b = pick_reg_index(c.rng);
    return InstructionFactory::with2(kMov.rm_r[static_cast<std::size_t>(wi)], reg_of(w, a), reg_of(w, b));
  }
  if (form == 1) {
    const int a = pick_reg_index(c.rng);
    if (w == Width::W64) {
      const std::uint64_t imm = random_interesting_u64(c.rng);
      return InstructionFactory::with2(Code::MOV_R64_IMM64, reg_of(w, a), static_cast<std::int64_t>(imm));
    }
    const int bits = bits_of(w);
    const std::int32_t imm = static_cast<std::int32_t>(random_imm(c.rng, bits));
    const Code code = w == Width::W8 ? Code::MOV_R8_IMM8 : w == Width::W16 ? Code::MOV_R16_IMM16 : Code::MOV_R32_IMM32;
    auto instr = InstructionFactory::with2(code, reg_of(w, a), imm);
    set_imm_sized(instr, 1, w, false, static_cast<std::uint64_t>(imm));
    return instr;
  }

  c.touches_memory = true;
  const std::int8_t disp = random_disp8(c.rng);
  const int mform = rand_int(c.rng, 0, 2);
  if (mform == 0) {
    const int r = pick_reg_index(c.rng);
    return InstructionFactory::with2(kMov.rm_r[static_cast<std::size_t>(wi)], mem_operand(disp), reg_of(w, r));
  }
  if (mform == 1) {
    const int r = pick_reg_index(c.rng);
    return InstructionFactory::with2(kMov.r_rm[static_cast<std::size_t>(wi)], reg_of(w, r), mem_operand(disp));
  }
  const std::int32_t imm = static_cast<std::int32_t>(random_imm(c.rng, w == Width::W8 ? 8 : w == Width::W16 ? 16 : 32));
  auto mov_mem = InstructionFactory::with2(kMov.rm_imm[static_cast<std::size_t>(wi)], mem_operand(disp), imm);
  set_imm_sized(mov_mem, 1, w, false, static_cast<std::uint64_t>(imm));
  return mov_mem;
}

// --------------------------------------------------------- MOVZX/MOVSX(D)

struct MovxCodes {
  std::array<Code, 6> zx;
  std::array<Code, 6> sx;
};
// order: {dst16<-src8, dst32<-src8, dst64<-src8, dst16<-src16, dst32<-src16, dst64<-src16}
const MovxCodes kMovx = {
    {Code::MOVZX_R16_RM8, Code::MOVZX_R32_RM8, Code::MOVZX_R64_RM8,
     Code::MOVZX_R16_RM16, Code::MOVZX_R32_RM16, Code::MOVZX_R64_RM16},
    {Code::MOVSX_R16_RM8, Code::MOVSX_R32_RM8, Code::MOVSX_R64_RM8,
     Code::MOVSX_R16_RM16, Code::MOVSX_R32_RM16, Code::MOVSX_R64_RM16},
};
constexpr std::array<Width, 6> kMovxDstW = {Width::W16, Width::W32, Width::W64, Width::W16, Width::W32, Width::W64};
constexpr std::array<Width, 6> kMovxSrcW = {Width::W8, Width::W8, Width::W8, Width::W16, Width::W16, Width::W16};

[[nodiscard]] std::optional<Instruction> gen_movx(Ctx& c) {
  const bool zero_ext = rand_int(c.rng, 0, 1) == 0;
  const std::size_t idx = static_cast<std::size_t>(rand_int(c.rng, 0, 5));
  const Code code = zero_ext ? kMovx.zx[idx] : kMovx.sx[idx];
  const Width dstw = kMovxDstW[idx], srcw = kMovxSrcW[idx];
  const int d = pick_reg_index(c.rng);
  if (rand_int(c.rng, 0, 1) == 0) {
    const int s = pick_reg_index(c.rng);
    return InstructionFactory::with2(code, reg_of(dstw, d), reg_of(srcw, s));
  }
  c.touches_memory = true;
  return InstructionFactory::with2(code, reg_of(dstw, d), mem_operand(random_disp8(c.rng)));
}

[[nodiscard]] std::optional<Instruction> gen_movsxd(Ctx& c) {
  const int d = pick_reg_index(c.rng);
  if (rand_int(c.rng, 0, 1) == 0) {
    const int s = pick_reg_index(c.rng);
    return InstructionFactory::with2(Code::MOVSXD_R64_RM32, reg_of(Width::W64, d), reg_of(Width::W32, s));
  }
  c.touches_memory = true;
  return InstructionFactory::with2(Code::MOVSXD_R64_RM32, reg_of(Width::W64, d), mem_operand(random_disp8(c.rng)));
}

// ---------------------------------------------------------- stack/LEA/misc

[[nodiscard]] std::optional<Instruction> gen_pushpop(Ctx& c) {
  const bool is_push = rand_int(c.rng, 0, 1) == 0;
  const int r = pick_reg_index(c.rng);
  return InstructionFactory::with1(is_push ? Code::PUSH_R64 : Code::POP_R64, reg_of(Width::W64, r));
}

[[nodiscard]] std::optional<Instruction> gen_lea(Ctx& c) {
  c.touches_memory = true;  // address arithmetic references RDI; no actual dereference
  const int d = pick_reg_index(c.rng);
  // The destination width truncates the computed address, and each width is its own Code.
  const Width w = width16_32_64(c.rng);
  static constexpr std::array<Code, 3> kLea = {Code::LEA_R16_M, Code::LEA_R32_M, Code::LEA_R64_M};
  return InstructionFactory::with2(kLea[static_cast<std::size_t>(widx16_32_64(w))], reg_of(w, d),
                                   mem_operand(random_disp8(c.rng)));
}

// MOVS/CMPS/SCAS/STOS/LODS. These are the only instructions whose direction depends on DF, they
// advance RSI/RDI themselves, and a rep prefix makes one instruction a whole loop -- all state the
// rest of the generator never touches. seven has already had two serious bugs in exactly this area
// (a rep-string cancellation gap, and a DF leak that reversed the host's own copies), and until now
// not one of these twenty codes had ever been generated.
//
// RSI/RDI are pinned mid-window rather than at its base: DF=1 walks them DOWNWARD, and starting at
// kDataBase would march straight out of the compared window and off the page for reasons that have
// nothing to do with the instruction under test. A small count keeps every iteration inside the
// window in either direction.
[[nodiscard]] std::optional<Instruction> gen_string_ops(Ctx& c) {
  using iced_x86::RepPrefixKind;
  c.touches_memory = true;

  constexpr std::uint64_t kMidWindow = kDataBase + (kDataWindow / 2);
  c.force_gpr[6] = kMidWindow;  // RSI
  c.force_gpr[7] = kMidWindow;  // RDI

  const int family = rand_int(c.rng, 0, 4);
  const int wi = rand_int(c.rng, 0, 3);
  // A rep prefix turns this into a loop, so the count has to stay small enough that every iteration
  // lands inside the window whichever way DF sends it.
  const int rep_pick = rand_int(c.rng, 0, 2);
  const auto rep = rep_pick == 0   ? RepPrefixKind::NONE
                   : rep_pick == 1 ? RepPrefixKind::REPE
                                   : RepPrefixKind::REPNE;
  if (rep != RepPrefixKind::NONE) {
    // Only 0 or 1. A REP string instruction is architecturally interruptible between iterations,
    // and both oracles single-step it that way -- hardware traps after each iteration with TF set,
    // and Unicorn stops after one too -- while seven runs the whole loop inside one step(). With a
    // count above 1 every single rep form reports a divergence that is purely that difference in
    // what one step means, which buries anything real. At 0 and 1 the two models agree exactly, so
    // what is left compares the actual string semantics: DF direction, RSI/RDI advance, the flags
    // CMPS/SCAS set, and the zero-count early out.
    c.force_gpr[1] = static_cast<std::uint64_t>(rand_int(c.rng, 0, 1));  // RCX
  }
  constexpr std::uint32_t kAddr64 = 64;

  switch (family) {
    case 0:
      switch (wi) {
        case 0: return InstructionFactory::with_movsb(kAddr64, Register::NONE, rep);
        case 1: return InstructionFactory::with_movsw(kAddr64, Register::NONE, rep);
        case 2: return InstructionFactory::with_movsd(kAddr64, Register::NONE, rep);
        default: return InstructionFactory::with_movsq(kAddr64, Register::NONE, rep);
      }
    case 1:
      switch (wi) {
        case 0: return InstructionFactory::with_cmpsb(kAddr64, Register::NONE, rep);
        case 1: return InstructionFactory::with_cmpsw(kAddr64, Register::NONE, rep);
        case 2: return InstructionFactory::with_cmpsd(kAddr64, Register::NONE, rep);
        default: return InstructionFactory::with_cmpsq(kAddr64, Register::NONE, rep);
      }
    case 2:
      switch (wi) {
        case 0: return InstructionFactory::with_scasb(kAddr64, rep);
        case 1: return InstructionFactory::with_scasw(kAddr64, rep);
        case 2: return InstructionFactory::with_scasd(kAddr64, rep);
        default: return InstructionFactory::with_scasq(kAddr64, rep);
      }
    case 3:
      switch (wi) {
        case 0: return InstructionFactory::with_stosb(kAddr64, rep);
        case 1: return InstructionFactory::with_stosw(kAddr64, rep);
        case 2: return InstructionFactory::with_stosd(kAddr64, rep);
        default: return InstructionFactory::with_stosq(kAddr64, rep);
      }
    default:
      switch (wi) {
        case 0: return InstructionFactory::with_lodsb(kAddr64, Register::NONE, rep);
        case 1: return InstructionFactory::with_lodsw(kAddr64, Register::NONE, rep);
        case 2: return InstructionFactory::with_lodsd(kAddr64, Register::NONE, rep);
        default: return InstructionFactory::with_lodsq(kAddr64, Register::NONE, rep);
      }
  }
}

// The A0-A3 absolute-address forms. No ModRM and no base register at all -- the address is an
// 8-byte immediate -- which is why nothing else in the generator produces them, and why the store
// direction being implemented backwards went unnoticed.
[[nodiscard]] std::optional<Instruction> gen_moffs(Ctx& c) {
  c.flags_mask = 0;  // MOV defines no flags
  c.touches_memory = true;
  static constexpr std::array<Code, 4> kLoad = {Code::MOV_AL_MOFFS8, Code::MOV_AX_MOFFS16,
                                                 Code::MOV_EAX_MOFFS32, Code::MOV_RAX_MOFFS64};
  static constexpr std::array<Code, 4> kStore = {Code::MOV_MOFFS8_AL, Code::MOV_MOFFS16_AX,
                                                  Code::MOV_MOFFS32_EAX, Code::MOV_MOFFS64_RAX};
  const Width w = static_cast<Width>(rand_int(c.rng, 0, 3));
  const auto wu = static_cast<std::size_t>(w);
  const Register acc = w == Width::W8    ? Register::AL
                       : w == Width::W16 ? Register::AX
                       : w == Width::W32 ? Register::EAX
                                         : Register::RAX;
  // Keep the whole access inside the compared scratch window.
  const auto offset = static_cast<std::uint64_t>(rand_int(c.rng, 0, 15)) * 8;
  const auto addr = MemoryOperand::with_displ(kDataBase + offset, 8);
  return rand_int(c.rng, 0, 1) == 0 ? InstructionFactory::with2(kLoad[wu], acc, addr)
                                     : InstructionFactory::with2(kStore[wu], addr, acc);
}

// MOVBE byte-swaps on the way to or from memory, CRC32 accumulates the SSE4.2 polynomial, and the
// 0F 1F multi-byte NOP has to genuinely do nothing at every width -- three separate families that
// only share the fact that none of them had any generator coverage at all.
[[nodiscard]] std::optional<Instruction> gen_movbe_crc32_nop(Ctx& c) {
  const int pick = rand_int(c.rng, 0, 2);
  if (pick == 0) {
    c.flags_mask = 0;  // MOVBE defines no flags
    c.touches_memory = true;
    const Width w = width16_32_64(c.rng);
    const auto wu = static_cast<std::size_t>(widx16_32_64(w));
    const int r = pick_reg_index(c.rng);
    static constexpr std::array<Code, 3> kLoad = {Code::MOVBE_R16_M16, Code::MOVBE_R32_M32,
                                                   Code::MOVBE_R64_M64};
    static constexpr std::array<Code, 3> kStore = {Code::MOVBE_M16_R16, Code::MOVBE_M32_R32,
                                                    Code::MOVBE_M64_R64};
    const std::int8_t disp = random_disp8(c.rng);
    return rand_int(c.rng, 0, 1) == 0
        ? InstructionFactory::with2(kLoad[wu], reg_of(w, r), mem_operand(disp))
        : InstructionFactory::with2(kStore[wu], mem_operand(disp), reg_of(w, r));
  }
  if (pick == 1) {
    c.flags_mask = 0;  // CRC32 defines no flags
    const int d = pick_reg_index(c.rng);
    const int src = pick_reg_index(c.rng);
    // The destination is always 32- or 64-bit; only the SOURCE width varies.
    const bool dest64 = rand_int(c.rng, 0, 1) == 0;
    const Width sw = dest64 ? (rand_int(c.rng, 0, 1) == 0 ? Width::W8 : Width::W64)
                            : static_cast<Width>(rand_int(c.rng, 0, 2));
    Code code = Code::CRC32_R32_RM8;
    if (dest64) {
      code = sw == Width::W8 ? Code::CRC32_R64_RM8 : Code::CRC32_R64_RM64;
    } else {
      code = sw == Width::W8 ? Code::CRC32_R32_RM8 : sw == Width::W16 ? Code::CRC32_R32_RM16
                                                                      : Code::CRC32_R32_RM32;
    }
    const Width dw = dest64 ? Width::W64 : Width::W32;
    if (rand_int(c.rng, 0, 2) == 0) {
      c.touches_memory = true;
      return InstructionFactory::with2(code, reg_of(dw, d), mem_operand(random_disp8(c.rng)));
    }
    return InstructionFactory::with2(code, reg_of(dw, d), reg_of(sw, src));
  }
  c.flags_mask = 0;
  const Width w = width16_32_64(c.rng);
  static constexpr std::array<Code, 3> kNop = {Code::NOP_RM16, Code::NOP_RM32, Code::NOP_RM64};
  const Code code = kNop[static_cast<std::size_t>(widx16_32_64(w))];
  if (rand_int(c.rng, 0, 1) == 0) {
    c.touches_memory = true;
    return InstructionFactory::with1(code, mem_operand(random_disp8(c.rng)));
  }
  return InstructionFactory::with1(code, reg_of(w, pick_reg_index(c.rng)));
}

// --------------------------------------------------------- branches/Jcc

constexpr std::array<Code, 16> kJcc = {
    Code::JO_REL8_64, Code::JNO_REL8_64, Code::JB_REL8_64, Code::JAE_REL8_64,
    Code::JE_REL8_64, Code::JNE_REL8_64, Code::JBE_REL8_64, Code::JA_REL8_64,
    Code::JS_REL8_64, Code::JNS_REL8_64, Code::JP_REL8_64, Code::JNP_REL8_64,
    Code::JL_REL8_64, Code::JGE_REL8_64, Code::JLE_REL8_64, Code::JG_REL8_64,
};
// Same condition order as kJcc. The rel32 forms are a separate Code entirely, and the encoder
// never shrinks one into the other, so both have to be asked for by name to get covered.
constexpr std::array<Code, 16> kJcc32 = {
    Code::JO_REL32_64, Code::JNO_REL32_64, Code::JB_REL32_64, Code::JAE_REL32_64,
    Code::JE_REL32_64, Code::JNE_REL32_64, Code::JBE_REL32_64, Code::JA_REL32_64,
    Code::JS_REL32_64, Code::JNS_REL32_64, Code::JP_REL32_64, Code::JNP_REL32_64,
    Code::JL_REL32_64, Code::JGE_REL32_64, Code::JLE_REL32_64, Code::JG_REL32_64,
};
constexpr std::array<Code, 16> kSetcc = {
    Code::SETO_RM8, Code::SETNO_RM8, Code::SETB_RM8, Code::SETAE_RM8,
    Code::SETE_RM8, Code::SETNE_RM8, Code::SETBE_RM8, Code::SETA_RM8,
    Code::SETS_RM8, Code::SETNS_RM8, Code::SETP_RM8, Code::SETNP_RM8,
    Code::SETL_RM8, Code::SETGE_RM8, Code::SETLE_RM8, Code::SETG_RM8,
};
constexpr std::array<Code, 16> kCmov16 = {
    Code::CMOVO_R16_RM16, Code::CMOVNO_R16_RM16, Code::CMOVB_R16_RM16, Code::CMOVAE_R16_RM16,
    Code::CMOVE_R16_RM16, Code::CMOVNE_R16_RM16, Code::CMOVBE_R16_RM16, Code::CMOVA_R16_RM16,
    Code::CMOVS_R16_RM16, Code::CMOVNS_R16_RM16, Code::CMOVP_R16_RM16, Code::CMOVNP_R16_RM16,
    Code::CMOVL_R16_RM16, Code::CMOVGE_R16_RM16, Code::CMOVLE_R16_RM16, Code::CMOVG_R16_RM16,
};
constexpr std::array<Code, 16> kCmov32 = {
    Code::CMOVO_R32_RM32, Code::CMOVNO_R32_RM32, Code::CMOVB_R32_RM32, Code::CMOVAE_R32_RM32,
    Code::CMOVE_R32_RM32, Code::CMOVNE_R32_RM32, Code::CMOVBE_R32_RM32, Code::CMOVA_R32_RM32,
    Code::CMOVS_R32_RM32, Code::CMOVNS_R32_RM32, Code::CMOVP_R32_RM32, Code::CMOVNP_R32_RM32,
    Code::CMOVL_R32_RM32, Code::CMOVGE_R32_RM32, Code::CMOVLE_R32_RM32, Code::CMOVG_R32_RM32,
};
constexpr std::array<Code, 16> kCmov64 = {
    Code::CMOVO_R64_RM64, Code::CMOVNO_R64_RM64, Code::CMOVB_R64_RM64, Code::CMOVAE_R64_RM64,
    Code::CMOVE_R64_RM64, Code::CMOVNE_R64_RM64, Code::CMOVBE_R64_RM64, Code::CMOVA_R64_RM64,
    Code::CMOVS_R64_RM64, Code::CMOVNS_R64_RM64, Code::CMOVP_R64_RM64, Code::CMOVNP_R64_RM64,
    Code::CMOVL_R64_RM64, Code::CMOVGE_R64_RM64, Code::CMOVLE_R64_RM64, Code::CMOVG_R64_RM64,
};

// Branch targets are kept inside the mapped, executable code page. If a
// target instead landed in unmapped memory, Unicorn's count=1 single-step
// appears to eagerly fault on the *destination* fetch (to decide whether to
// keep going) even though only the branch itself should retire — real
// hardware (and seven) don't fault until something actually fetches there.
// Comparing that would be comparing a single-step measurement artifact, not
// a real execution difference, so every branch target stays in-page.
[[nodiscard]] std::optional<Instruction> gen_jcc(Ctx& c) {
  const auto which = static_cast<std::size_t>(rand_int(c.rng, 0, 15));
  if (rand_int(c.rng, 0, 1) == 0) {
    // rel8 reach: stay comfortably inside +/-127 of kCodeBase regardless of
    // this instruction's own length.
    const auto target = kCodeBase + static_cast<std::uint64_t>(rand_int(c.rng, 2, 120));
    return InstructionFactory::with_branch(kJcc[which], target);
  }
  return InstructionFactory::with_branch(kJcc32[which], random_code_target(c.rng));
}
[[nodiscard]] std::optional<Instruction> gen_jmp(Ctx& c) {
  switch (rand_int(c.rng, 0, 3)) {
    case 0: {
      const auto target = kCodeBase + static_cast<std::uint64_t>(rand_int(c.rng, 2, 120));
      return InstructionFactory::with_branch(Code::JMP_REL8_64, target);
    }
    case 1:
      return InstructionFactory::with_branch(Code::JMP_REL32_64, random_code_target(c.rng));
    case 2: {
      const int r = pick_reg_index(c.rng);
      c.force_gpr[static_cast<std::size_t>(r)] = random_code_target(c.rng);
      return InstructionFactory::with1(Code::JMP_RM64, reg_of(Width::W64, r));
    }
    default: {
      const std::int8_t disp = indirect_target_disp(c);
      c.touches_memory = true;
      return InstructionFactory::with1(Code::JMP_RM64, mem_operand(disp));
    }
  }
}
[[nodiscard]] std::optional<Instruction> gen_call(Ctx& c) {
  switch (rand_int(c.rng, 0, 2)) {
    case 0:
      return InstructionFactory::with_branch(Code::CALL_REL32_64, random_code_target(c.rng));
    case 1: {
      const int r = pick_reg_index(c.rng);
      // RSP is pinned to the harness stack top; overwriting it with a code address would send the
      // pushed return address somewhere unmapped instead of exercising the call itself.
      if (r == 4) { return std::nullopt; }
      c.force_gpr[static_cast<std::size_t>(r)] = random_code_target(c.rng);
      return InstructionFactory::with1(Code::CALL_RM64, reg_of(Width::W64, r));
    }
    default: {
      const std::int8_t disp = indirect_target_disp(c);
      c.touches_memory = true;
      return InstructionFactory::with1(Code::CALL_RM64, mem_operand(disp));
    }
  }
}
[[nodiscard]] std::optional<Instruction> gen_ret(Ctx& c) {
  if (rand_int(c.rng, 0, 1) == 0) { return InstructionFactory::with(Code::RETNQ); }
  // Small, aligned pop counts only: the harness stack has 0x0F00 bytes of headroom above RSP, and
  // a random imm16 would walk RSP off the mapped page for reasons that have nothing to do with RET.
  const auto pop_bytes = rand_int(c.rng, 0, 32) * 8;
  auto instr = InstructionFactory::with1(Code::RETNQ_IMM16, pop_bytes);
  set_imm_kind(instr, 0, OpKind::IMMEDIATE16);
  instr.set_immediate16(static_cast<std::uint16_t>(pop_bytes));
  return instr;
}

// LOOP/JRCXZ read (and LOOP writes) the count register but define no flags at all, which makes
// them a clean check that a branch family isn't clobbering flags on the side.
constexpr std::array<Code, 6> kLoop = {
    Code::LOOP_REL8_64_RCX,   Code::LOOP_REL8_64_ECX,   Code::LOOPE_REL8_64_RCX,
    Code::LOOPE_REL8_64_ECX,  Code::LOOPNE_REL8_64_RCX, Code::LOOPNE_REL8_64_ECX,
};

[[nodiscard]] std::optional<Instruction> gen_loop(Ctx& c) {
  const auto target = kCodeBase + static_cast<std::uint64_t>(rand_int(c.rng, 2, 120));
  const int pick = rand_int(c.rng, 0, 7);
  // A fully random RCX is essentially never near the zero boundary, which is the only interesting
  // one here. Bias it small so both the taken and not-taken edges actually get exercised.
  if (rand_int(c.rng, 0, 1) == 0) {
    c.force_gpr[1] = static_cast<std::uint64_t>(rand_int(c.rng, 0, 3));
  }
  if (pick == 6) { return InstructionFactory::with_branch(Code::JRCXZ_REL8_64, target); }
  if (pick == 7) { return InstructionFactory::with_branch(Code::JECXZ_REL8_64, target); }
  return InstructionFactory::with_branch(kLoop[static_cast<std::size_t>(pick)], target);
}

[[nodiscard]] std::optional<Instruction> gen_setcc(Ctx& c) {
  const Code code = kSetcc[static_cast<std::size_t>(rand_int(c.rng, 0, 15))];
  if (rand_int(c.rng, 0, 1) == 0) {
    const int r = pick_reg_index(c.rng);
    return InstructionFactory::with1(code, reg_of(Width::W8, r));
  }
  c.touches_memory = true;
  return InstructionFactory::with1(code, mem_operand(random_disp8(c.rng)));
}

[[nodiscard]] std::optional<Instruction> gen_cmovcc(Ctx& c) {
  const int cc = rand_int(c.rng, 0, 15);
  const int wpick = rand_int(c.rng, 0, 2);
  const Width w = wpick == 0 ? Width::W16 : wpick == 1 ? Width::W32 : Width::W64;
  const Code code = wpick == 0 ? kCmov16[static_cast<std::size_t>(cc)]
                    : wpick == 1 ? kCmov32[static_cast<std::size_t>(cc)]
                                 : kCmov64[static_cast<std::size_t>(cc)];
  const int d = pick_reg_index(c.rng);
  if (rand_int(c.rng, 0, 1) == 0) {
    const int s = pick_reg_index(c.rng);
    return InstructionFactory::with2(code, reg_of(w, d), reg_of(w, s));
  }
  c.touches_memory = true;
  return InstructionFactory::with2(code, reg_of(w, d), mem_operand(random_disp8(c.rng)));
}

// -------------------------------------------------------------- BT family

struct BtCodes {
  std::array<Code, 3> rm_r;    // 16,32,64
  std::array<Code, 3> rm_imm;  // 16,32,64
};
const BtCodes kBt = {{Code::BT_RM16_R16, Code::BT_RM32_R32, Code::BT_RM64_R64},
                      {Code::BT_RM16_IMM8, Code::BT_RM32_IMM8, Code::BT_RM64_IMM8}};
const BtCodes kBts = {{Code::BTS_RM16_R16, Code::BTS_RM32_R32, Code::BTS_RM64_R64},
                       {Code::BTS_RM16_IMM8, Code::BTS_RM32_IMM8, Code::BTS_RM64_IMM8}};
const BtCodes kBtr = {{Code::BTR_RM16_R16, Code::BTR_RM32_R32, Code::BTR_RM64_R64},
                       {Code::BTR_RM16_IMM8, Code::BTR_RM32_IMM8, Code::BTR_RM64_IMM8}};
const BtCodes kBtc = {{Code::BTC_RM16_R16, Code::BTC_RM32_R32, Code::BTC_RM64_R64},
                       {Code::BTC_RM16_IMM8, Code::BTC_RM32_IMM8, Code::BTC_RM64_IMM8}};
const std::array<const BtCodes*, 4> kBtFamily = {&kBt, &kBts, &kBtr, &kBtc};

[[nodiscard]] std::optional<Instruction> gen_bt(Ctx& c) {
  // BT/BTS/BTR/BTC only define CF (the prior value of the tested bit); the
  // rest (OF/SF/AF/PF) are architecturally undefined.
  c.flags_mask = 0x0001ull;
  const BtCodes& t = *kBtFamily[static_cast<std::size_t>(rand_int(c.rng, 0, 3))];
  const Width w = width16_32_64(c.rng);
  const std::size_t wi = static_cast<std::size_t>(widx16_32_64(w));
  const bool use_imm = rand_int(c.rng, 0, 1) == 0;

  if (rand_int(c.rng, 0, 1) == 0) {
    const int a = pick_reg_index(c.rng);
    if (use_imm) {
      const std::int32_t imm = static_cast<std::int32_t>(random_imm(c.rng, 8));
      auto instr = InstructionFactory::with2(t.rm_imm[wi], reg_of(w, a), imm);
      set_imm8(instr, 1, static_cast<std::uint64_t>(imm));
      return instr;
    }
    const int b = pick_reg_index(c.rng);
    return InstructionFactory::with2(t.rm_r[wi], reg_of(w, a), reg_of(w, b));
  }
  c.touches_memory = true;
  const std::int8_t disp = random_disp8(c.rng);
  if (use_imm) {
    const std::int32_t imm = static_cast<std::int32_t>(random_imm(c.rng, 8));
    auto instr = InstructionFactory::with2(t.rm_imm[wi], mem_operand(disp), imm);
    set_imm8(instr, 1, static_cast<std::uint64_t>(imm));
    return instr;
  }
  const int b = pick_reg_index(c.rng);
  return InstructionFactory::with2(t.rm_r[wi], mem_operand(disp), reg_of(w, b));
}

// ---------------------------------------------------- BSF/BSR/POPCNT/etc.

struct RmSrcCodes {
  std::array<Code, 3> code;  // 16,32,64
};
const RmSrcCodes kBsf = {{Code::BSF_R16_RM16, Code::BSF_R32_RM32, Code::BSF_R64_RM64}};
const RmSrcCodes kBsr = {{Code::BSR_R16_RM16, Code::BSR_R32_RM32, Code::BSR_R64_RM64}};
const RmSrcCodes kPopcnt = {{Code::POPCNT_R16_RM16, Code::POPCNT_R32_RM32, Code::POPCNT_R64_RM64}};
const RmSrcCodes kLzcnt = {{Code::LZCNT_R16_RM16, Code::LZCNT_R32_RM32, Code::LZCNT_R64_RM64}};
const RmSrcCodes kTzcnt = {{Code::TZCNT_R16_RM16, Code::TZCNT_R32_RM32, Code::TZCNT_R64_RM64}};
const std::array<const RmSrcCodes*, 5> kRmSrcFamily = {&kBsf, &kBsr, &kPopcnt, &kLzcnt, &kTzcnt};

[[nodiscard]] std::optional<Instruction> gen_rmsrc(Ctx& c) {
  const int pick = rand_int(c.rng, 0, 4);
  const bool is_bsf_bsr = pick == 0 || pick == 1;
  if (is_bsf_bsr) {
    // BSF/BSR define only ZF; CF/OF/SF/AF/PF are architecturally undefined
    // (unlike POPCNT/LZCNT/TZCNT, which fully define all of them).
    c.flags_mask = 0x0040ull;
  }
  const RmSrcCodes& t = *kRmSrcFamily[static_cast<std::size_t>(pick)];
  const Width w = width16_32_64(c.rng);
  const std::size_t wi = static_cast<std::size_t>(widx16_32_64(w));
  const int d = pick_reg_index(c.rng);
  if (rand_int(c.rng, 0, 1) == 0) {
    const int s = pick_reg_index(c.rng);
    if (is_bsf_bsr) {
      c.bsf_bsr_dest = d;
      c.bsf_bsr_src_reg = s;
      c.bsf_bsr_width_bytes = bits_of(w) / 8;
    }
    return InstructionFactory::with2(t.code[wi], reg_of(w, d), reg_of(w, s));
  }
  c.touches_memory = true;
  const std::int8_t disp = random_disp8(c.rng);
  if (is_bsf_bsr) {
    c.bsf_bsr_dest = d;
    c.bsf_bsr_src_mem_disp = disp;
    c.bsf_bsr_width_bytes = bits_of(w) / 8;
  }
  return InstructionFactory::with2(t.code[wi], reg_of(w, d), mem_operand(disp));
}

[[nodiscard]] std::optional<Instruction> gen_bswap(Ctx& c) {
  const bool w64 = rand_int(c.rng, 0, 1) == 0;
  const int r = pick_reg_index(c.rng);
  return InstructionFactory::with1(w64 ? Code::BSWAP_R64 : Code::BSWAP_R32, reg_of(w64 ? Width::W64 : Width::W32, r));
}

// ------------------------------------------------------------------- BMI1/BMI2
// The only VEX-encoded family in this generator. Everything in the SIMD sections above is legacy
// SSE, so no VEX prefix had ever been produced at all and not one of these twenty-six codes had
// ever been generated -- seven's implementations of them had only ever been checked by reading.
//
// AF and PF are architecturally undefined for every BMI1 code, so they come out of the comparison
// mask. The BMI2 half writes no flags whatsoever, which the full mask already covers since both
// engines then leave them alone.
enum class BmiShape { kDstSrcRm, kDstRmSrc, kDstRm, kDstRmImm };

struct BmiEntry {
  std::array<Code, 2> code;  // 32, 64
  BmiShape shape;
  bool writes_flags;
};

const std::array<BmiEntry, 13> kBmi = {{
    {{Code::VEX_ANDN_R32_R32_RM32, Code::VEX_ANDN_R64_R64_RM64}, BmiShape::kDstSrcRm, true},
    {{Code::VEX_BEXTR_R32_RM32_R32, Code::VEX_BEXTR_R64_RM64_R64}, BmiShape::kDstRmSrc, true},
    {{Code::VEX_BLSI_R32_RM32, Code::VEX_BLSI_R64_RM64}, BmiShape::kDstRm, true},
    {{Code::VEX_BLSMSK_R32_RM32, Code::VEX_BLSMSK_R64_RM64}, BmiShape::kDstRm, true},
    {{Code::VEX_BLSR_R32_RM32, Code::VEX_BLSR_R64_RM64}, BmiShape::kDstRm, true},
    {{Code::VEX_BZHI_R32_RM32_R32, Code::VEX_BZHI_R64_RM64_R64}, BmiShape::kDstRmSrc, true},
    {{Code::VEX_PDEP_R32_R32_RM32, Code::VEX_PDEP_R64_R64_RM64}, BmiShape::kDstSrcRm, false},
    {{Code::VEX_PEXT_R32_R32_RM32, Code::VEX_PEXT_R64_R64_RM64}, BmiShape::kDstSrcRm, false},
    {{Code::VEX_MULX_R32_R32_RM32, Code::VEX_MULX_R64_R64_RM64}, BmiShape::kDstSrcRm, false},
    {{Code::VEX_RORX_R32_RM32_IMM8, Code::VEX_RORX_R64_RM64_IMM8}, BmiShape::kDstRmImm, false},
    {{Code::VEX_SARX_R32_RM32_R32, Code::VEX_SARX_R64_RM64_R64}, BmiShape::kDstRmSrc, false},
    {{Code::VEX_SHLX_R32_RM32_R32, Code::VEX_SHLX_R64_RM64_R64}, BmiShape::kDstRmSrc, false},
    {{Code::VEX_SHRX_R32_RM32_R32, Code::VEX_SHRX_R64_RM64_R64}, BmiShape::kDstRmSrc, false},
}};

[[nodiscard]] std::optional<Instruction> gen_bmi(Ctx& c) {
  const BmiEntry& e = kBmi[static_cast<std::size_t>(rand_int(c.rng, 0, static_cast<int>(kBmi.size()) - 1))];
  const bool w64 = rand_int(c.rng, 0, 1) == 0;
  const Width w = w64 ? Width::W64 : Width::W32;
  const Code code = e.code[w64 ? 1 : 0];
  if (e.writes_flags) {
    c.flags_mask = 0x001ull | 0x040ull | 0x080ull | 0x800ull;  // CF, ZF, SF, OF
  }
  const int d = pick_reg_index(c.rng);
  const int s = pick_reg_index(c.rng);
  const bool rm_is_mem = rand_int(c.rng, 0, 2) == 0;
  if (rm_is_mem) c.touches_memory = true;
  const std::int8_t disp = random_disp8(c.rng);
  const auto imm8 = static_cast<std::int32_t>(random_imm(c.rng, 8));
  switch (e.shape) {
    case BmiShape::kDstSrcRm:
      return rm_is_mem ? InstructionFactory::with3(code, reg_of(w, d), reg_of(w, s), mem_operand(disp))
                       : InstructionFactory::with3(code, reg_of(w, d), reg_of(w, s), reg_of(w, pick_reg_index(c.rng)));
    case BmiShape::kDstRmSrc:
      return rm_is_mem ? InstructionFactory::with3(code, reg_of(w, d), mem_operand(disp), reg_of(w, s))
                       : InstructionFactory::with3(code, reg_of(w, d), reg_of(w, pick_reg_index(c.rng)), reg_of(w, s));
    case BmiShape::kDstRm:
      return rm_is_mem ? InstructionFactory::with2(code, reg_of(w, d), mem_operand(disp))
                       : InstructionFactory::with2(code, reg_of(w, d), reg_of(w, s));
    default:
      return rm_is_mem ? InstructionFactory::with3(code, reg_of(w, d), mem_operand(disp), imm8)
                       : InstructionFactory::with3(code, reg_of(w, d), reg_of(w, s), imm8);
  }
}

// ---------------------------------------------------------- MUL/IMUL/DIV/IDIV

struct W4Codes {
  std::array<Code, 4> code;  // 8,16,32,64
};
const W4Codes kMul = {{Code::MUL_RM8, Code::MUL_RM16, Code::MUL_RM32, Code::MUL_RM64}};
const W4Codes kImul1 = {{Code::IMUL_RM8, Code::IMUL_RM16, Code::IMUL_RM32, Code::IMUL_RM64}};
const W4Codes kDiv = {{Code::DIV_RM8, Code::DIV_RM16, Code::DIV_RM32, Code::DIV_RM64}};
const W4Codes kIdiv = {{Code::IDIV_RM8, Code::IDIV_RM16, Code::IDIV_RM32, Code::IDIV_RM64}};

// x86 register numbering, excluding RAX(0)/RDX(2) -- those hold the dividend
// in the "in-range" DIV/IDIV construction below, so the divisor register
// can't also be one of them without the two roles colliding -- and RSP(4),
// just to avoid any doubt about the harness's own stack-pointer convention,
// even though this instruction never touches the stack.
constexpr std::array<int, 13> kNonAccumRegs = {1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

[[nodiscard]] std::optional<Instruction> gen_muldiv(Ctx& c) {
  const int family = rand_int(c.rng, 0, 3);  // 0=MUL 1=IMUL(1-op) 2=DIV 3=IDIV
  const Width w = static_cast<Width>(rand_int(c.rng, 0, 3));
  const std::size_t wi = static_cast<std::size_t>(w);
  const bool is_div_family = family >= 2;
  // MUL/IMUL(1-op) define only CF/OF (SF/ZF/AF/PF undefined); DIV/IDIV
  // leave every flag undefined per spec.
  c.flags_mask = is_div_family ? 0ull : 0x0801ull;
  const Code code = family == 0 ? kMul.code[wi] : family == 1 ? kImul1.code[wi] : family == 2 ? kDiv.code[wi] : kIdiv.code[wi];

  if (is_div_family && rand_int(c.rng, 0, 2) != 0) {
    // Bias toward an in-range dividend/divisor pair ~2/3 of the time --
    // fully random operands overflow the quotient (and thus #DE) almost
    // every time, which would mean barely exercising the actual division
    // algorithm. Register-only rm here (never memory) so the divisor value
    // can be forced directly. Best-effort for IDIV: the construction is
    // unsigned-shaped, so an occasional unexpected #DE from a sign mismatch
    // is fine -- this only needs to bias the odds, not guarantee in-range.
    const int mapped = kNonAccumRegs[static_cast<std::size_t>(rand_int(c.rng, 0, 12))];
    const int bits = bits_of(w);
    std::uint64_t divisor = random_imm(c.rng, bits);
    if (divisor == 0) divisor = 1;
    const std::uint64_t quotient = random_imm(c.rng, bits);
    const std::uint64_t remainder = divisor > 1 ? (random_interesting_u64(c.rng) % divisor) : 0;
    std::uint64_t hi = 0;
    const std::uint64_t product_lo = _umul128(quotient, divisor, &hi);
    std::uint64_t lo = 0;
    const unsigned char carry = _addcarry_u64(0, product_lo, remainder, &lo);
    _addcarry_u64(carry, hi, 0ull, &hi);
    c.force_gpr[static_cast<std::size_t>(mapped)] = divisor;
    c.force_gpr[0] = lo;                        // RAX (also the whole AX for the 8-bit form)
    if (w != Width::W8) c.force_gpr[2] = hi;     // RDX -- unused/ignored by the 8-bit form
    return InstructionFactory::with1(code, reg_of(w, mapped));
  }

  if (rand_int(c.rng, 0, 1) == 0) {
    const int r = pick_reg_index(c.rng);
    return InstructionFactory::with1(code, reg_of(w, r));
  }
  c.touches_memory = true;
  return InstructionFactory::with1(code, mem_operand(random_disp8(c.rng)));
}

struct Imul2Codes {
  std::array<Code, 3> rm;  // 16,32,64
};
struct Imul3Codes {
  std::array<Code, 3> imm8;
  std::array<Code, 3> immfull;
};
const Imul2Codes kImul2 = {{Code::IMUL_R16_RM16, Code::IMUL_R32_RM32, Code::IMUL_R64_RM64}};
const Imul3Codes kImul3 = {
    {Code::IMUL_R16_RM16_IMM8, Code::IMUL_R32_RM32_IMM8, Code::IMUL_R64_RM64_IMM8},
    {Code::IMUL_R16_RM16_IMM16, Code::IMUL_R32_RM32_IMM32, Code::IMUL_R64_RM64_IMM32},
};

[[nodiscard]] std::optional<Instruction> gen_imul_multi(Ctx& c) {
  c.flags_mask = 0x0801ull;  // same as the 1-op form: CF/OF defined, rest undefined
  const Width w = width16_32_64(c.rng);
  const std::size_t wi = static_cast<std::size_t>(widx16_32_64(w));
  const int form = rand_int(c.rng, 0, 2);  // 0=2-op reg,reg  1=2-op reg,mem  2=3-op reg,rm,imm

  if (form == 0) {
    const int d = pick_reg_index(c.rng), s = pick_reg_index(c.rng);
    return InstructionFactory::with2(kImul2.rm[wi], reg_of(w, d), reg_of(w, s));
  }
  if (form == 1) {
    c.touches_memory = true;
    const int d = pick_reg_index(c.rng);
    return InstructionFactory::with2(kImul2.rm[wi], reg_of(w, d), mem_operand(random_disp8(c.rng)));
  }
  const int d = pick_reg_index(c.rng);
  const bool short_imm = rand_int(c.rng, 0, 1) == 0;
  const std::int32_t imm = short_imm
      ? static_cast<std::int32_t>(static_cast<std::int8_t>(random_imm(c.rng, 8)))
      : static_cast<std::int32_t>(random_imm(c.rng, w == Width::W16 ? 16 : 32));
  const Code code = short_imm ? kImul3.imm8[wi] : kImul3.immfull[wi];
  if (rand_int(c.rng, 0, 1) == 0) {
    const int s = pick_reg_index(c.rng);
    auto instr = InstructionFactory::with3(code, reg_of(w, d), reg_of(w, s), imm);
    set_imm_sized(instr, 2, w, short_imm, static_cast<std::uint64_t>(imm));
    return instr;
  }
  c.touches_memory = true;
  auto imul_mem = InstructionFactory::with3(code, reg_of(w, d), mem_operand(random_disp8(c.rng)), imm);
  set_imm_sized(imul_mem, 2, w, short_imm, static_cast<std::uint64_t>(imm));
  return imul_mem;
}

// ------------------------------------------------------------- SIMD (XMM)
//
// Legacy-SSE (non-VEX/EVEX) forms only, XMM0-15, 128-bit width. VEX.128 forms
// (e.g. VEX_VSHUFPS) now have registered handlers too (see the
// handled_codes.def orphaned-handler sweep) but are deliberately not
// generated yet -- this project has no host-CPU-capability detection at all,
// and while AVX is effectively universal, EVEX/AVX-512 is not: seven_core is
// built here with SEVEN_ENABLE_AVX512=1 unconditionally, so on a host that
// lacks real AVX-512, seven would execute EVEX instructions "successfully"
// while hardware #UDs on every single case -- a systematic, uninteresting
// divergence, the same class of noise privileged-instruction generation was
// excluded for (see the comment on gen_privileged below). Left as a natural
// follow-up requiring a runtime CPUID/XGETBV check, not attempted here.

constexpr std::array<Register, 16> kRegsXmm = {
    Register::XMM0,  Register::XMM1,  Register::XMM2,  Register::XMM3,
    Register::XMM4,  Register::XMM5,  Register::XMM6,  Register::XMM7,
    Register::XMM8,  Register::XMM9,  Register::XMM10, Register::XMM11,
    Register::XMM12, Register::XMM13, Register::XMM14, Register::XMM15,
};
[[nodiscard]] Register xmm_of(int idx) { return kRegsXmm[static_cast<std::size_t>(idx)]; }
[[nodiscard]] int pick_xmm_index(std::mt19937_64& rng) { return rand_int(rng, 0, 15); }

// dst, src(xmm-or-mem), imm8 -- SHUFPS/SHUFPD/PSHUFD/PSHUFLW/PSHUFHW. SHUFPS
// is the instruction that started this: it had a correct handler in
// simd_shuffle.cpp that was never wired into handled_codes.def, so every real
// SHUFPS hit unsupported_instruction while hardware executed it fine.
constexpr std::array<Code, 5> kSimdShuffleImm = {
    Code::SHUFPS_XMM_XMMM128_IMM8,  Code::SHUFPD_XMM_XMMM128_IMM8, Code::PSHUFD_XMM_XMMM128_IMM8,
    Code::PSHUFLW_XMM_XMMM128_IMM8, Code::PSHUFHW_XMM_XMMM128_IMM8,
};
// dst, src(xmm-or-mem), no imm -- UNPCK*/MOVSLDUP/MOVSHDUP/MOVDDUP
constexpr std::array<Code, 7> kSimdShuffleNoImm = {
    Code::UNPCKLPS_XMM_XMMM128, Code::UNPCKLPD_XMM_XMMM128, Code::UNPCKHPS_XMM_XMMM128,
    Code::UNPCKHPD_XMM_XMMM128, Code::MOVSLDUP_XMM_XMMM128, Code::MOVSHDUP_XMM_XMMM128,
    Code::MOVDDUP_XMM_XMMM64,
};

[[nodiscard]] std::optional<Instruction> gen_simd_shuffle(Ctx& c) {
  c.flags_mask = 0;  // no shuffle/permute variant here touches EFLAGS
  const int d = pick_xmm_index(c.rng);
  const bool use_mem = rand_int(c.rng, 0, 3) == 0;
  if (rand_int(c.rng, 0, 1) == 0) {
    const Code code =
        kSimdShuffleImm[static_cast<std::size_t>(rand_int(c.rng, 0, static_cast<int>(kSimdShuffleImm.size()) - 1))];
    const auto imm = static_cast<std::int32_t>(random_imm(c.rng, 8));
    if (use_mem) {
      c.touches_memory = true;
      auto instr = InstructionFactory::with3(code, xmm_of(d), mem_operand(random_disp8(c.rng)), imm);
      set_imm8(instr, 2, static_cast<std::uint64_t>(imm));
      return instr;
    }
    const int s = pick_xmm_index(c.rng);
    auto instr = InstructionFactory::with3(code, xmm_of(d), xmm_of(s), imm);
    set_imm8(instr, 2, static_cast<std::uint64_t>(imm));
    return instr;
  }
  const Code code =
      kSimdShuffleNoImm[static_cast<std::size_t>(rand_int(c.rng, 0, static_cast<int>(kSimdShuffleNoImm.size()) - 1))];
  if (use_mem) {
    c.touches_memory = true;
    return InstructionFactory::with2(code, xmm_of(d), mem_operand(random_disp8(c.rng)));
  }
  const int s = pick_xmm_index(c.rng);
  return InstructionFactory::with2(code, xmm_of(d), xmm_of(s));
}

constexpr std::array<Code, 12> kSimdLogic = {
    Code::ANDPS_XMM_XMMM128, Code::ANDPD_XMM_XMMM128, Code::ANDNPS_XMM_XMMM128, Code::ANDNPD_XMM_XMMM128,
    Code::ORPS_XMM_XMMM128,  Code::ORPD_XMM_XMMM128,  Code::XORPS_XMM_XMMM128,  Code::XORPD_XMM_XMMM128,
    Code::PAND_XMM_XMMM128,  Code::PANDN_XMM_XMMM128, Code::POR_XMM_XMMM128,    Code::PXOR_XMM_XMMM128,
};

[[nodiscard]] std::optional<Instruction> gen_simd_logic(Ctx& c) {
  c.flags_mask = 0;
  const Code code = kSimdLogic[static_cast<std::size_t>(rand_int(c.rng, 0, static_cast<int>(kSimdLogic.size()) - 1))];
  const int d = pick_xmm_index(c.rng);
  if (rand_int(c.rng, 0, 3) == 0) {
    c.touches_memory = true;
    return InstructionFactory::with2(code, xmm_of(d), mem_operand(random_disp8(c.rng)));
  }
  const int s = pick_xmm_index(c.rng);
  return InstructionFactory::with2(code, xmm_of(d), xmm_of(s));
}

constexpr std::array<Code, 11> kSimdPack = {
    Code::PACKSSWB_XMM_XMMM128,   Code::PACKSSDW_XMM_XMMM128,   Code::PACKUSWB_XMM_XMMM128,
    Code::PUNPCKLBW_XMM_XMMM128,  Code::PUNPCKHBW_XMM_XMMM128,  Code::PUNPCKLWD_XMM_XMMM128,
    Code::PUNPCKHWD_XMM_XMMM128,  Code::PUNPCKLDQ_XMM_XMMM128,  Code::PUNPCKHDQ_XMM_XMMM128,
    Code::PUNPCKLQDQ_XMM_XMMM128, Code::PUNPCKHQDQ_XMM_XMMM128,
};

[[nodiscard]] std::optional<Instruction> gen_simd_pack(Ctx& c) {
  c.flags_mask = 0;
  const Code code = kSimdPack[static_cast<std::size_t>(rand_int(c.rng, 0, static_cast<int>(kSimdPack.size()) - 1))];
  const int d = pick_xmm_index(c.rng);
  if (rand_int(c.rng, 0, 3) == 0) {
    c.touches_memory = true;
    return InstructionFactory::with2(code, xmm_of(d), mem_operand(random_disp8(c.rng)));
  }
  const int s = pick_xmm_index(c.rng);
  return InstructionFactory::with2(code, xmm_of(d), xmm_of(s));
}

constexpr std::array<Code, 8> kSimdShiftReg = {
    Code::PSLLW_XMM_XMMM128, Code::PSLLD_XMM_XMMM128, Code::PSLLQ_XMM_XMMM128, Code::PSRLW_XMM_XMMM128,
    Code::PSRLD_XMM_XMMM128, Code::PSRLQ_XMM_XMMM128, Code::PSRAW_XMM_XMMM128, Code::PSRAD_XMM_XMMM128,
};
constexpr std::array<Code, 10> kSimdShiftImm = {
    Code::PSLLW_XMM_IMM8,  Code::PSLLD_XMM_IMM8,  Code::PSLLQ_XMM_IMM8,   Code::PSRLW_XMM_IMM8, Code::PSRLD_XMM_IMM8,
    Code::PSRLQ_XMM_IMM8,  Code::PSRAW_XMM_IMM8,  Code::PSRAD_XMM_IMM8,
    Code::PSRLDQ_XMM_IMM8, Code::PSLLDQ_XMM_IMM8,
};

[[nodiscard]] std::optional<Instruction> gen_simd_shift(Ctx& c) {
  c.flags_mask = 0;
  const int d = pick_xmm_index(c.rng);
  if (rand_int(c.rng, 0, 1) == 0) {
    const Code code =
        kSimdShiftImm[static_cast<std::size_t>(rand_int(c.rng, 0, static_cast<int>(kSimdShiftImm.size()) - 1))];
    const auto imm = static_cast<std::int32_t>(random_imm(c.rng, 8));
    auto instr = InstructionFactory::with2(code, xmm_of(d), imm);
    set_imm8(instr, 1, static_cast<std::uint64_t>(imm));
    return instr;
  }
  const Code code =
      kSimdShiftReg[static_cast<std::size_t>(rand_int(c.rng, 0, static_cast<int>(kSimdShiftReg.size()) - 1))];
  if (rand_int(c.rng, 0, 3) == 0) {
    c.touches_memory = true;
    return InstructionFactory::with2(code, xmm_of(d), mem_operand(random_disp8(c.rng)));
  }
  const int s = pick_xmm_index(c.rng);
  return InstructionFactory::with2(code, xmm_of(d), xmm_of(s));
}

constexpr std::array<Code, 6> kSimdFpNoFlags = {
    Code::ADDSUBPS_XMM_XMMM128, Code::ADDSUBPD_XMM_XMMM128, Code::HADDPS_XMM_XMMM128,
    Code::HADDPD_XMM_XMMM128,   Code::HSUBPS_XMM_XMMM128,   Code::HSUBPD_XMM_XMMM128,
};
constexpr std::array<Code, 4> kSimdFpCompare = {
    Code::COMISS_XMM_XMMM32, Code::UCOMISS_XMM_XMMM32, Code::COMISD_XMM_XMMM64, Code::UCOMISD_XMM_XMMM64,
};

// The SSE/SSE2 float workhorses. Every one of these twenty-eight is a real-code hot path and not one
// had ever been generated, even though all three lanes already compare XMM state in full. MIN/MAX
// are the interesting pair: their NaN and signed-zero behaviour is asymmetric by design (the second
// operand wins whenever either input is a NaN, and -0.0 vs +0.0 is decided by the operand order, not
// by value), which an implementation written the obvious way gets wrong.
//
// RCP*/RSQRT* are deliberately absent: hardware only promises those to within a relative error
// bound, so no soft-float implementation can match them bit for bit and every case would report.
constexpr std::array<Code, 28> kSseArith = {
    Code::ADDPS_XMM_XMMM128,  Code::ADDPD_XMM_XMMM128,  Code::ADDSS_XMM_XMMM32,  Code::ADDSD_XMM_XMMM64,
    Code::SUBPS_XMM_XMMM128,  Code::SUBPD_XMM_XMMM128,  Code::SUBSS_XMM_XMMM32,  Code::SUBSD_XMM_XMMM64,
    Code::MULPS_XMM_XMMM128,  Code::MULPD_XMM_XMMM128,  Code::MULSS_XMM_XMMM32,  Code::MULSD_XMM_XMMM64,
    Code::DIVPS_XMM_XMMM128,  Code::DIVPD_XMM_XMMM128,  Code::DIVSS_XMM_XMMM32,  Code::DIVSD_XMM_XMMM64,
    Code::MINPS_XMM_XMMM128,  Code::MINPD_XMM_XMMM128,  Code::MINSS_XMM_XMMM32,  Code::MINSD_XMM_XMMM64,
    Code::MAXPS_XMM_XMMM128,  Code::MAXPD_XMM_XMMM128,  Code::MAXSS_XMM_XMMM32,  Code::MAXSD_XMM_XMMM64,
    Code::SQRTPS_XMM_XMMM128, Code::SQRTPD_XMM_XMMM128, Code::SQRTSS_XMM_XMMM32, Code::SQRTSD_XMM_XMMM64,
};

// The packed-integer workhorses. Saturating add/sub, the averaging pair, the high-half multiplies
// and PSADBW/PMADDWD are all places where the obvious implementation is off by one somewhere: the
// saturating forms have to clamp per lane at the right signedness, PAVG rounds up rather than
// truncating, PMULHW keeps the high half of a SIGNED product while PMULHUW keeps the unsigned one,
// and PSADBW/PMADDWD change lane width mid-operation. None of these thirty-four had ever run.
constexpr std::array<Code, 34> kPackedInt = {
    Code::PADDB_XMM_XMMM128,    Code::PADDW_XMM_XMMM128,    Code::PADDD_XMM_XMMM128,
    Code::PADDQ_XMM_XMMM128,    Code::PADDSB_XMM_XMMM128,   Code::PADDSW_XMM_XMMM128,
    Code::PADDUSB_XMM_XMMM128,  Code::PADDUSW_XMM_XMMM128,
    Code::PSUBB_XMM_XMMM128,    Code::PSUBW_XMM_XMMM128,    Code::PSUBD_XMM_XMMM128,
    Code::PSUBQ_XMM_XMMM128,    Code::PSUBSB_XMM_XMMM128,   Code::PSUBSW_XMM_XMMM128,
    Code::PSUBUSB_XMM_XMMM128,  Code::PSUBUSW_XMM_XMMM128,
    Code::PCMPEQB_XMM_XMMM128,  Code::PCMPEQW_XMM_XMMM128,  Code::PCMPEQD_XMM_XMMM128,
    Code::PCMPEQQ_XMM_XMMM128,  Code::PCMPGTB_XMM_XMMM128,  Code::PCMPGTW_XMM_XMMM128,
    Code::PCMPGTD_XMM_XMMM128,  Code::PCMPGTQ_XMM_XMMM128,
    Code::PAVGB_XMM_XMMM128,    Code::PAVGW_XMM_XMMM128,
    Code::PMAXUB_XMM_XMMM128,   Code::PMINSW_XMM_XMMM128,
    Code::PMULLW_XMM_XMMM128,   Code::PMULHW_XMM_XMMM128,   Code::PMULHUW_XMM_XMMM128,
    Code::PMULUDQ_XMM_XMMM128,  Code::PMADDWD_XMM_XMMM128,  Code::PSADBW_XMM_XMMM128,
};

// The SSE move forms, in both directions. Every SIMD family generated so far reads memory and writes
// a register; these are the first that write GUEST MEMORY from a vector register, which is a
// different code path in every lane. The alignment rule splits them: the "A"/non-temporal forms
// require 16 bytes, the "U" forms and LDDQU explicitly do not, and the scalar/m64 forms have no
// requirement at all. Feeding a deliberately misaligned address to an aligned form now and then
// keeps the #GP side compared too, since that is exactly where the packed float family had a hole.
struct SseMoveForm {
  Code code;
  bool needs_alignment;
};

constexpr std::array<SseMoveForm, 16> kSseMoveLoad = {{
    {Code::MOVAPS_XMM_XMMM128, true},  {Code::MOVAPD_XMM_XMMM128, true},
    {Code::MOVDQA_XMM_XMMM128, true},  {Code::MOVNTDQA_XMM_M128, true},
    {Code::MOVUPS_XMM_XMMM128, false}, {Code::MOVUPD_XMM_XMMM128, false},
    {Code::MOVDQU_XMM_XMMM128, false}, {Code::LDDQU_XMM_M128, false},
    {Code::MOVSS_XMM_XMMM32, false},   {Code::MOVSD_XMM_XMMM64, false},
    {Code::MOVHPS_XMM_M64, false},     {Code::MOVHPD_XMM_M64, false},
    {Code::MOVLPS_XMM_M64, false},     {Code::MOVLPD_XMM_M64, false},
    {Code::MOVQ_XMM_XMMM64, false},    {Code::MOVDDUP_XMM_XMMM64, false},
}};

constexpr std::array<SseMoveForm, 15> kSseMoveStore = {{
    {Code::MOVAPS_XMMM128_XMM, true},  {Code::MOVAPD_XMMM128_XMM, true},
    {Code::MOVDQA_XMMM128_XMM, true},  {Code::MOVNTPS_M128_XMM, true},
    {Code::MOVNTPD_M128_XMM, true},    {Code::MOVNTDQ_M128_XMM, true},
    {Code::MOVUPS_XMMM128_XMM, false}, {Code::MOVUPD_XMMM128_XMM, false},
    {Code::MOVDQU_XMMM128_XMM, false}, {Code::MOVSS_XMMM32_XMM, false},
    {Code::MOVSD_XMMM64_XMM, false},   {Code::MOVHPS_M64_XMM, false},
    {Code::MOVHPD_M64_XMM, false},     {Code::MOVLPS_M64_XMM, false},
    {Code::MOVLPD_M64_XMM, false},
}};

[[nodiscard]] std::optional<Instruction> gen_sse_move(Ctx& c) {
  c.flags_mask = 0;  // no move form touches EFLAGS
  const bool store = rand_int(c.rng, 0, 1) == 0;
  const auto form = store
      ? kSseMoveStore[static_cast<std::size_t>(rand_int(c.rng, 0, static_cast<int>(kSseMoveStore.size()) - 1))]
      : kSseMoveLoad[static_cast<std::size_t>(rand_int(c.rng, 0, static_cast<int>(kSseMoveLoad.size()) - 1))];

  // One in eight aligned forms gets a misaligned address on purpose, so the #GP path stays compared
  // rather than only the success path.
  const bool force_misaligned = form.needs_alignment && rand_int(c.rng, 0, 7) == 0;
  const auto disp = (form.needs_alignment && !force_misaligned) ? aligned_disp8(c.rng, 16)
                                                                : random_disp8(c.rng);

  const int reg = pick_xmm_index(c.rng);
  if (store) {
    c.touches_memory = true;
    return InstructionFactory::with2(form.code, mem_operand(disp), xmm_of(reg));
  }
  // The M-only load forms have no register-source encoding at all.
  const bool mem_only = form.code == Code::MOVNTDQA_XMM_M128 || form.code == Code::LDDQU_XMM_M128 ||
                        form.code == Code::MOVHPS_XMM_M64 || form.code == Code::MOVHPD_XMM_M64 ||
                        form.code == Code::MOVLPS_XMM_M64 || form.code == Code::MOVLPD_XMM_M64;
  if (mem_only || rand_int(c.rng, 0, 2) != 0) {
    c.touches_memory = true;
    return InstructionFactory::with2(form.code, xmm_of(reg), mem_operand(disp));
  }
  return InstructionFactory::with2(form.code, xmm_of(reg), xmm_of(pick_xmm_index(c.rng)));
}

[[nodiscard]] std::optional<Instruction> gen_packed_int(Ctx& c) {
  c.flags_mask = 0;  // none of these touch EFLAGS
  const Code code = kPackedInt[static_cast<std::size_t>(rand_int(c.rng, 0, static_cast<int>(kPackedInt.size()) - 1))];
  const int d = pick_xmm_index(c.rng);
  if (rand_int(c.rng, 0, 3) == 0) {
    c.touches_memory = true;
    // The legacy m128 forms all require a 16-byte-aligned source, so keep the displacement aligned
    // rather than spending three quarters of this family's budget re-proving the same #GP.
    return InstructionFactory::with2(code, xmm_of(d), mem_operand(aligned_disp8(c.rng, 16)));
  }
  const int s = pick_xmm_index(c.rng);
  return InstructionFactory::with2(code, xmm_of(d), xmm_of(s));
}

[[nodiscard]] std::optional<Instruction> gen_sse_arith(Ctx& c) {
  c.flags_mask = 0;  // none of these touch EFLAGS
  const Code code = kSseArith[static_cast<std::size_t>(rand_int(c.rng, 0, static_cast<int>(kSseArith.size()) - 1))];
  const int d = pick_xmm_index(c.rng);
  if (rand_int(c.rng, 0, 3) == 0) {
    c.touches_memory = true;
    return InstructionFactory::with2(code, xmm_of(d), mem_operand(random_disp8(c.rng)));
  }
  const int s = pick_xmm_index(c.rng);
  return InstructionFactory::with2(code, xmm_of(d), xmm_of(s));
}

[[nodiscard]] std::optional<Instruction> gen_simd_fp(Ctx& c) {
  const int d = pick_xmm_index(c.rng);
  const bool is_compare = rand_int(c.rng, 0, 1) == 0;
  const Code code =
      is_compare
          ? kSimdFpCompare[static_cast<std::size_t>(rand_int(c.rng, 0, static_cast<int>(kSimdFpCompare.size()) - 1))]
          : kSimdFpNoFlags[static_cast<std::size_t>(rand_int(c.rng, 0, static_cast<int>(kSimdFpNoFlags.size()) - 1))];
  // COMISS/UCOMISS/COMISD/UCOMISD define CF/PF/ZF from the comparison and
  // architecturally clear OF/SF/AF -- compared in full (not narrowed), since
  // a failure to clear them would be a real bug, not an undefined-flag
  // false positive. Everything else in this family doesn't touch EFLAGS.
  c.flags_mask = is_compare ? kCompareFlagsMask : 0;
  if (rand_int(c.rng, 0, 3) == 0) {
    c.touches_memory = true;
    return InstructionFactory::with2(code, xmm_of(d), mem_operand(random_disp8(c.rng)));
  }
  const int s = pick_xmm_index(c.rng);
  return InstructionFactory::with2(code, xmm_of(d), xmm_of(s));
}

// ------------------------------------------------ VEX / EVEX SIMD (128-bit)
//
// seven registers 519 VEX and EVEX handlers and until now the generator produced exactly none of
// them: everything above is legacy SSE. The reason it was held back was that seven is built with
// SEVEN_ENABLE_AVX512=1 whatever the host is, so on a machine without real AVX-512 every EVEX case
// would be "seven executed it, hardware #UD'd" -- thousands of identical divergences saying nothing
// about seven. host_caps() answers that question at startup instead of guessing, so the corpus now
// stays inside what the hardware oracle can actually adjudicate.
//
// Three things keep this to 128-bit forms with XMM0-15 and no mask register:
//   - RegState carries the low and high 64 bits of XMM0-15 and nothing else, because the hardware
//     lane reads its result out of CONTEXT.FltSave, which is the legacy FXSAVE area. YMM and ZMM
//     upper lanes and XMM16-31 have no slot to be compared in, so generating them would exercise
//     the engines without ever checking the answer.
//   - Nothing anywhere sets K0-K7, so a masked EVEX form would merge against whatever the register
//     happened to hold in each lane and diverge for a reason that isn't a bug. Every EVEX form here
//     encodes with no mask.
//   - AVX-512 is required whole (F+VL+BW+DQ) rather than per-instruction. A host missing one of the
//     four skips EVEX entirely, which loses coverage on that host but never invents a finding.
//
// The upper lanes a VEX.128 write is supposed to zero are therefore not compared. That is a real
// coverage gap and the reason for it is the FXSAVE area, not an opinion about it being unimportant.

enum class VexShape : std::uint8_t {
  kDstSrcRm,    // dst, src1, src2-or-m128
  kDstRm,       // dst, src-or-m128
  kRmDst,       // dst-or-m128, src   (store direction)
  kDstSrcImm,   // dst, src, imm8     (shifts by immediate)
  kDstSrcSrc,   // dst, src1, src2    (register only)
  kDstRmImm,    // dst, src-or-m128, imm8
  kDstSrcRm64,  // dst, src1, src2-or-m64   (scalar double)
  kDstSrcRm32,  // dst, src1, src2-or-m32   (scalar single)
  kDstRm64,     // dst, src-or-m64
  kDstSrcRmImm, // dst, src1, src2-or-m128, imm8
};

struct VexEntry {
  Code code;
  VexShape shape;
  bool evex;
  // The real Code name ends in B32 or B64, meaning an EVEX memory source may carry {1toN}: one
  // element read and repeated across the operand. Worth generating, since the element size comes
  // from a different field than the operand size and reading the wrong one is silent.
  bool broadcast;
};

// Derived from seven's own handled_codes.def rather than typed out, so the list cannot drift from
// what the emulator claims to support and cannot contain a mnemonic that does not exist.
constexpr std::array<VexEntry, 204> kVexSimd = {{
    {Code::EVEX_VMOVAPD_XMMM128_K1Z_XMM, VexShape::kRmDst, true, false},
    {Code::EVEX_VMOVAPD_XMM_K1Z_XMMM128, VexShape::kDstRm, true, false},
    {Code::EVEX_VMOVAPS_XMMM128_K1Z_XMM, VexShape::kRmDst, true, false},
    {Code::EVEX_VMOVAPS_XMM_K1Z_XMMM128, VexShape::kDstRm, true, false},
    {Code::EVEX_VMOVDDUP_XMM_K1Z_XMMM64, VexShape::kDstRm64, true, false},
    {Code::EVEX_VMOVDQA32_XMMM128_K1Z_XMM, VexShape::kRmDst, true, false},
    {Code::EVEX_VMOVDQA32_XMM_K1Z_XMMM128, VexShape::kDstRm, true, false},
    {Code::EVEX_VMOVDQA64_XMMM128_K1Z_XMM, VexShape::kRmDst, true, false},
    {Code::EVEX_VMOVDQA64_XMM_K1Z_XMMM128, VexShape::kDstRm, true, false},
    {Code::EVEX_VMOVDQU16_XMMM128_K1Z_XMM, VexShape::kRmDst, true, false},
    {Code::EVEX_VMOVDQU16_XMM_K1Z_XMMM128, VexShape::kDstRm, true, false},
    {Code::EVEX_VMOVDQU32_XMMM128_K1Z_XMM, VexShape::kRmDst, true, false},
    {Code::EVEX_VMOVDQU32_XMM_K1Z_XMMM128, VexShape::kDstRm, true, false},
    {Code::EVEX_VMOVDQU64_XMMM128_K1Z_XMM, VexShape::kRmDst, true, false},
    {Code::EVEX_VMOVDQU64_XMM_K1Z_XMMM128, VexShape::kDstRm, true, false},
    {Code::EVEX_VMOVDQU8_XMMM128_K1Z_XMM, VexShape::kRmDst, true, false},
    {Code::EVEX_VMOVDQU8_XMM_K1Z_XMMM128, VexShape::kDstRm, true, false},
    {Code::EVEX_VMOVHLPS_XMM_XMM_XMM, VexShape::kDstSrcSrc, true, false},
    {Code::EVEX_VMOVLHPS_XMM_XMM_XMM, VexShape::kDstSrcSrc, true, false},
    {Code::EVEX_VMOVQ_XMM_XMMM64, VexShape::kDstRm64, true, false},
    {Code::EVEX_VMOVSD_XMM_K1Z_XMM_XMM, VexShape::kDstSrcSrc, true, false},
    {Code::EVEX_VMOVSHDUP_XMM_K1Z_XMMM128, VexShape::kDstRm, true, false},
    {Code::EVEX_VMOVSLDUP_XMM_K1Z_XMMM128, VexShape::kDstRm, true, false},
    {Code::EVEX_VMOVSS_XMM_K1Z_XMM_XMM, VexShape::kDstSrcSrc, true, false},
    {Code::EVEX_VMOVUPD_XMMM128_K1Z_XMM, VexShape::kRmDst, true, false},
    {Code::EVEX_VMOVUPD_XMM_K1Z_XMMM128, VexShape::kDstRm, true, false},
    {Code::EVEX_VMOVUPS_XMMM128_K1Z_XMM, VexShape::kRmDst, true, false},
    {Code::EVEX_VMOVUPS_XMM_K1Z_XMMM128, VexShape::kDstRm, true, false},
    {Code::EVEX_VPACKSSDW_XMM_K1Z_XMM_XMMM128B32, VexShape::kDstSrcRm, true, true},
    {Code::EVEX_VPACKSSWB_XMM_K1Z_XMM_XMMM128, VexShape::kDstSrcRm, true, false},
    {Code::EVEX_VPACKUSWB_XMM_K1Z_XMM_XMMM128, VexShape::kDstSrcRm, true, false},
    {Code::EVEX_VPADDB_XMM_K1Z_XMM_XMMM128, VexShape::kDstSrcRm, true, false},
    {Code::EVEX_VPADDD_XMM_K1Z_XMM_XMMM128B32, VexShape::kDstSrcRm, true, true},
    {Code::EVEX_VPADDQ_XMM_K1Z_XMM_XMMM128B64, VexShape::kDstSrcRm, true, true},
    {Code::EVEX_VPADDSB_XMM_K1Z_XMM_XMMM128, VexShape::kDstSrcRm, true, false},
    {Code::EVEX_VPADDSW_XMM_K1Z_XMM_XMMM128, VexShape::kDstSrcRm, true, false},
    {Code::EVEX_VPADDUSB_XMM_K1Z_XMM_XMMM128, VexShape::kDstSrcRm, true, false},
    {Code::EVEX_VPADDUSW_XMM_K1Z_XMM_XMMM128, VexShape::kDstSrcRm, true, false},
    {Code::EVEX_VPADDW_XMM_K1Z_XMM_XMMM128, VexShape::kDstSrcRm, true, false},
    {Code::EVEX_VPANDD_XMM_K1Z_XMM_XMMM128B32, VexShape::kDstSrcRm, true, true},
    {Code::EVEX_VPANDND_XMM_K1Z_XMM_XMMM128B32, VexShape::kDstSrcRm, true, true},
    {Code::EVEX_VPANDNQ_XMM_K1Z_XMM_XMMM128B64, VexShape::kDstSrcRm, true, true},
    {Code::EVEX_VPANDQ_XMM_K1Z_XMM_XMMM128B64, VexShape::kDstSrcRm, true, true},
    {Code::EVEX_VPAVGB_XMM_K1Z_XMM_XMMM128, VexShape::kDstSrcRm, true, false},
    {Code::EVEX_VPAVGW_XMM_K1Z_XMM_XMMM128, VexShape::kDstSrcRm, true, false},
    {Code::EVEX_VPMULLW_XMM_K1Z_XMM_XMMM128, VexShape::kDstSrcRm, true, false},
    {Code::EVEX_VPORD_XMM_K1Z_XMM_XMMM128B32, VexShape::kDstSrcRm, true, true},
    {Code::EVEX_VPORQ_XMM_K1Z_XMM_XMMM128B64, VexShape::kDstSrcRm, true, true},
    {Code::EVEX_VPSLLD_XMM_K1Z_XMMM128B32_IMM8, VexShape::kDstRmImm, true, true},
    {Code::EVEX_VPSLLD_XMM_K1Z_XMM_XMMM128, VexShape::kDstSrcRm, true, false},
    {Code::EVEX_VPSLLQ_XMM_K1Z_XMMM128B64_IMM8, VexShape::kDstRmImm, true, true},
    {Code::EVEX_VPSLLQ_XMM_K1Z_XMM_XMMM128, VexShape::kDstSrcRm, true, false},
    {Code::EVEX_VPSLLW_XMM_K1Z_XMMM128_IMM8, VexShape::kDstRmImm, true, false},
    {Code::EVEX_VPSLLW_XMM_K1Z_XMM_XMMM128, VexShape::kDstSrcRm, true, false},
    {Code::EVEX_VPSRAD_XMM_K1Z_XMMM128B32_IMM8, VexShape::kDstRmImm, true, true},
    {Code::EVEX_VPSRAD_XMM_K1Z_XMM_XMMM128, VexShape::kDstSrcRm, true, false},
    {Code::EVEX_VPSRAW_XMM_K1Z_XMMM128_IMM8, VexShape::kDstRmImm, true, false},
    {Code::EVEX_VPSRAW_XMM_K1Z_XMM_XMMM128, VexShape::kDstSrcRm, true, false},
    {Code::EVEX_VPSRLD_XMM_K1Z_XMMM128B32_IMM8, VexShape::kDstRmImm, true, true},
    {Code::EVEX_VPSRLD_XMM_K1Z_XMM_XMMM128, VexShape::kDstSrcRm, true, false},
    {Code::EVEX_VPSRLQ_XMM_K1Z_XMMM128B64_IMM8, VexShape::kDstRmImm, true, true},
    {Code::EVEX_VPSRLQ_XMM_K1Z_XMM_XMMM128, VexShape::kDstSrcRm, true, false},
    {Code::EVEX_VPSRLW_XMM_K1Z_XMMM128_IMM8, VexShape::kDstRmImm, true, false},
    {Code::EVEX_VPSRLW_XMM_K1Z_XMM_XMMM128, VexShape::kDstSrcRm, true, false},
    {Code::EVEX_VPSUBB_XMM_K1Z_XMM_XMMM128, VexShape::kDstSrcRm, true, false},
    {Code::EVEX_VPSUBD_XMM_K1Z_XMM_XMMM128B32, VexShape::kDstSrcRm, true, true},
    {Code::EVEX_VPSUBQ_XMM_K1Z_XMM_XMMM128B64, VexShape::kDstSrcRm, true, true},
    {Code::EVEX_VPSUBSB_XMM_K1Z_XMM_XMMM128, VexShape::kDstSrcRm, true, false},
    {Code::EVEX_VPSUBSW_XMM_K1Z_XMM_XMMM128, VexShape::kDstSrcRm, true, false},
    {Code::EVEX_VPSUBUSB_XMM_K1Z_XMM_XMMM128, VexShape::kDstSrcRm, true, false},
    {Code::EVEX_VPSUBUSW_XMM_K1Z_XMM_XMMM128, VexShape::kDstSrcRm, true, false},
    {Code::EVEX_VPSUBW_XMM_K1Z_XMM_XMMM128, VexShape::kDstSrcRm, true, false},
    {Code::EVEX_VPUNPCKHBW_XMM_K1Z_XMM_XMMM128, VexShape::kDstSrcRm, true, false},
    {Code::EVEX_VPUNPCKHDQ_XMM_K1Z_XMM_XMMM128B32, VexShape::kDstSrcRm, true, true},
    {Code::EVEX_VPUNPCKHWD_XMM_K1Z_XMM_XMMM128, VexShape::kDstSrcRm, true, false},
    {Code::EVEX_VPUNPCKLBW_XMM_K1Z_XMM_XMMM128, VexShape::kDstSrcRm, true, false},
    {Code::EVEX_VPUNPCKLDQ_XMM_K1Z_XMM_XMMM128B32, VexShape::kDstSrcRm, true, true},
    {Code::EVEX_VPUNPCKLWD_XMM_K1Z_XMM_XMMM128, VexShape::kDstSrcRm, true, false},
    {Code::EVEX_VPXORD_XMM_K1Z_XMM_XMMM128B32, VexShape::kDstSrcRm, true, true},
    {Code::EVEX_VPXORQ_XMM_K1Z_XMM_XMMM128B64, VexShape::kDstSrcRm, true, true},
    {Code::EVEX_VSHUFPD_XMM_K1Z_XMM_XMMM128B64_IMM8, VexShape::kDstSrcRmImm, true, true},
    {Code::EVEX_VSHUFPS_XMM_K1Z_XMM_XMMM128B32_IMM8, VexShape::kDstSrcRmImm, true, true},
    {Code::EVEX_VUNPCKHPD_XMM_K1Z_XMM_XMMM128B64, VexShape::kDstSrcRm, true, true},
    {Code::EVEX_VUNPCKHPS_XMM_K1Z_XMM_XMMM128B32, VexShape::kDstSrcRm, true, true},
    {Code::EVEX_VUNPCKLPD_XMM_K1Z_XMM_XMMM128B64, VexShape::kDstSrcRm, true, true},
    {Code::EVEX_VUNPCKLPS_XMM_K1Z_XMM_XMMM128B32, VexShape::kDstSrcRm, true, true},
    {Code::VEX_VADDPD_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VADDPS_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VADDSD_XMM_XMM_XMMM64, VexShape::kDstSrcRm64, false, false},
    {Code::VEX_VADDSS_XMM_XMM_XMMM32, VexShape::kDstSrcRm32, false, false},
    {Code::VEX_VANDNPD_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VANDNPS_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VANDPD_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VANDPS_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VDIVPD_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VDIVPS_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VDIVSD_XMM_XMM_XMMM64, VexShape::kDstSrcRm64, false, false},
    {Code::VEX_VDIVSS_XMM_XMM_XMMM32, VexShape::kDstSrcRm32, false, false},
    {Code::VEX_VMAXPD_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VMAXPS_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VMAXSD_XMM_XMM_XMMM64, VexShape::kDstSrcRm64, false, false},
    {Code::VEX_VMAXSS_XMM_XMM_XMMM32, VexShape::kDstSrcRm32, false, false},
    {Code::VEX_VMINPD_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VMINPS_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VMINSD_XMM_XMM_XMMM64, VexShape::kDstSrcRm64, false, false},
    {Code::VEX_VMINSS_XMM_XMM_XMMM32, VexShape::kDstSrcRm32, false, false},
    {Code::VEX_VMOVAPD_XMMM128_XMM, VexShape::kRmDst, false, false},
    {Code::VEX_VMOVAPD_XMM_XMMM128, VexShape::kDstRm, false, false},
    {Code::VEX_VMOVAPS_XMMM128_XMM, VexShape::kRmDst, false, false},
    {Code::VEX_VMOVAPS_XMM_XMMM128, VexShape::kDstRm, false, false},
    {Code::VEX_VMOVDDUP_XMM_XMMM64, VexShape::kDstRm64, false, false},
    {Code::VEX_VMOVDQA_XMMM128_XMM, VexShape::kRmDst, false, false},
    {Code::VEX_VMOVDQA_XMM_XMMM128, VexShape::kDstRm, false, false},
    {Code::VEX_VMOVDQU_XMMM128_XMM, VexShape::kRmDst, false, false},
    {Code::VEX_VMOVDQU_XMM_XMMM128, VexShape::kDstRm, false, false},
    {Code::VEX_VMOVHLPS_XMM_XMM_XMM, VexShape::kDstSrcSrc, false, false},
    {Code::VEX_VMOVLHPS_XMM_XMM_XMM, VexShape::kDstSrcSrc, false, false},
    {Code::VEX_VMOVQ_XMM_XMMM64, VexShape::kDstRm64, false, false},
    {Code::VEX_VMOVSD_XMM_XMM_XMM, VexShape::kDstSrcSrc, false, false},
    {Code::VEX_VMOVSHDUP_XMM_XMMM128, VexShape::kDstRm, false, false},
    {Code::VEX_VMOVSLDUP_XMM_XMMM128, VexShape::kDstRm, false, false},
    {Code::VEX_VMOVSS_XMM_XMM_XMM, VexShape::kDstSrcSrc, false, false},
    {Code::VEX_VMOVUPD_XMMM128_XMM, VexShape::kRmDst, false, false},
    {Code::VEX_VMOVUPD_XMM_XMMM128, VexShape::kDstRm, false, false},
    {Code::VEX_VMOVUPS_XMMM128_XMM, VexShape::kRmDst, false, false},
    {Code::VEX_VMOVUPS_XMM_XMMM128, VexShape::kDstRm, false, false},
    {Code::VEX_VMULPD_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VMULPS_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VMULSD_XMM_XMM_XMMM64, VexShape::kDstSrcRm64, false, false},
    {Code::VEX_VMULSS_XMM_XMM_XMMM32, VexShape::kDstSrcRm32, false, false},
    {Code::VEX_VORPD_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VORPS_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPACKSSDW_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPACKSSWB_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPACKUSWB_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPADDB_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPADDD_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPADDQ_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPADDSB_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPADDSW_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPADDUSB_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPADDUSW_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPADDW_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPANDN_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPAND_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPAVGB_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPAVGW_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPCMPEQB_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPCMPEQD_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPCMPEQQ_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPCMPEQW_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPCMPGTB_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPCMPGTD_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPCMPGTQ_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPCMPGTW_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPMULLW_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPOR_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPSLLDQ_XMM_XMM_IMM8, VexShape::kDstSrcImm, false, false},
    {Code::VEX_VPSLLD_XMM_XMM_IMM8, VexShape::kDstSrcImm, false, false},
    {Code::VEX_VPSLLD_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPSLLQ_XMM_XMM_IMM8, VexShape::kDstSrcImm, false, false},
    {Code::VEX_VPSLLQ_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPSLLW_XMM_XMM_IMM8, VexShape::kDstSrcImm, false, false},
    {Code::VEX_VPSLLW_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPSRAD_XMM_XMM_IMM8, VexShape::kDstSrcImm, false, false},
    {Code::VEX_VPSRAD_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPSRAW_XMM_XMM_IMM8, VexShape::kDstSrcImm, false, false},
    {Code::VEX_VPSRAW_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPSRLDQ_XMM_XMM_IMM8, VexShape::kDstSrcImm, false, false},
    {Code::VEX_VPSRLD_XMM_XMM_IMM8, VexShape::kDstSrcImm, false, false},
    {Code::VEX_VPSRLD_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPSRLQ_XMM_XMM_IMM8, VexShape::kDstSrcImm, false, false},
    {Code::VEX_VPSRLQ_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPSRLW_XMM_XMM_IMM8, VexShape::kDstSrcImm, false, false},
    {Code::VEX_VPSRLW_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPSUBB_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPSUBD_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPSUBQ_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPSUBSB_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPSUBSW_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPSUBUSB_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPSUBUSW_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPSUBW_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPUNPCKHBW_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPUNPCKHDQ_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPUNPCKHWD_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPUNPCKLBW_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPUNPCKLDQ_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPUNPCKLWD_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VPXOR_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VSHUFPD_XMM_XMM_XMMM128_IMM8, VexShape::kDstSrcRmImm, false, false},
    {Code::VEX_VSHUFPS_XMM_XMM_XMMM128_IMM8, VexShape::kDstSrcRmImm, false, false},
    {Code::VEX_VSQRTSD_XMM_XMM_XMMM64, VexShape::kDstSrcRm64, false, false},
    {Code::VEX_VSQRTSS_XMM_XMM_XMMM32, VexShape::kDstSrcRm32, false, false},
    {Code::VEX_VSUBPD_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VSUBPS_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VSUBSD_XMM_XMM_XMMM64, VexShape::kDstSrcRm64, false, false},
    {Code::VEX_VSUBSS_XMM_XMM_XMMM32, VexShape::kDstSrcRm32, false, false},
    {Code::VEX_VUNPCKHPD_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VUNPCKHPS_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VUNPCKLPD_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VUNPCKLPS_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VXORPD_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
    {Code::VEX_VXORPS_XMM_XMM_XMMM128, VexShape::kDstSrcRm, false, false},
}};

[[nodiscard]] std::optional<Instruction> gen_vex_simd(Ctx& c) {
  const fuzz::HostCaps& caps = fuzz::host_caps();
  if (!caps.avx || !caps.avx2) return std::nullopt;
  const bool evex_ok = caps.avx512f && caps.avx512vl && caps.avx512bw && caps.avx512dq;

  // None of these write EFLAGS. The VEX forms of COMISS and friends do, but they land in the
  // kDstRm shapes below and are handled by the flags_mask assignment after the pick.
  c.flags_mask = 0;

  // A host without AVX-512 leaves roughly a third of the table unusable, so retry rather than
  // discard the iteration outright.
  const VexEntry* entry = nullptr;
  for (int attempt = 0; attempt < 8 && entry == nullptr; ++attempt) {
    const VexEntry& candidate =
        kVexSimd[static_cast<std::size_t>(rand_int(c.rng, 0, static_cast<int>(kVexSimd.size()) - 1))];
    if (candidate.evex && !evex_ok) continue;
    entry = &candidate;
  }
  if (entry == nullptr) return std::nullopt;

  const Code code = entry->code;
  const int d = pick_xmm_index(c.rng);
  const int s1 = pick_xmm_index(c.rng);
  const int s2 = pick_xmm_index(c.rng);
  const bool use_mem = rand_int(c.rng, 0, 3) == 0;
  const bool broadcast = use_mem && entry->broadcast && rand_int(c.rng, 0, 2) == 0;

  // Set on the instruction rather than the operand: EVEX.b is a prefix bit, and iced picks the
  // element size out of the code's own memory size once it is set.
  const auto finish = [broadcast](Instruction instr) {
    if (broadcast) instr.set_is_broadcast(true);
    return instr;
  };

  switch (entry->shape) {
    case VexShape::kDstSrcRm:
    case VexShape::kDstSrcRm64:
    case VexShape::kDstSrcRm32:
      if (use_mem) {
        c.touches_memory = true;
        return finish(InstructionFactory::with3(code, xmm_of(d), xmm_of(s1), mem_operand(random_disp8(c.rng))));
      }
      return InstructionFactory::with3(code, xmm_of(d), xmm_of(s1), xmm_of(s2));

    case VexShape::kDstRm:
    case VexShape::kDstRm64:
      if (use_mem) {
        c.touches_memory = true;
        return finish(InstructionFactory::with2(code, xmm_of(d), mem_operand(random_disp8(c.rng))));
      }
      return InstructionFactory::with2(code, xmm_of(d), xmm_of(s1));

    case VexShape::kRmDst:
      if (use_mem) {
        c.touches_memory = true;
        return InstructionFactory::with2(code, mem_operand(random_disp8(c.rng)), xmm_of(s1));
      }
      return InstructionFactory::with2(code, xmm_of(d), xmm_of(s1));

    case VexShape::kDstSrcSrc:
      return InstructionFactory::with3(code, xmm_of(d), xmm_of(s1), xmm_of(s2));

    case VexShape::kDstSrcImm: {
      const auto imm = static_cast<std::int32_t>(random_imm(c.rng, 8));
      auto instr = InstructionFactory::with3(code, xmm_of(d), xmm_of(s1), imm);
      set_imm8(instr, 2, static_cast<std::uint64_t>(imm));
      return instr;
    }

    case VexShape::kDstSrcRmImm: {
      const auto imm = static_cast<std::int32_t>(random_imm(c.rng, 8));
      if (use_mem) {
        c.touches_memory = true;
        auto instr = InstructionFactory::with4(code, xmm_of(d), xmm_of(s1), mem_operand(random_disp8(c.rng)), imm);
        set_imm8(instr, 3, static_cast<std::uint64_t>(imm));
        return finish(instr);
      }
      auto instr = InstructionFactory::with4(code, xmm_of(d), xmm_of(s1), xmm_of(s2), imm);
      set_imm8(instr, 3, static_cast<std::uint64_t>(imm));
      return instr;
    }

    case VexShape::kDstRmImm: {
      const auto imm = static_cast<std::int32_t>(random_imm(c.rng, 8));
      if (use_mem) {
        c.touches_memory = true;
        auto instr = InstructionFactory::with3(code, xmm_of(d), mem_operand(random_disp8(c.rng)), imm);
        set_imm8(instr, 2, static_cast<std::uint64_t>(imm));
        return instr;
      }
      auto instr = InstructionFactory::with3(code, xmm_of(d), xmm_of(s1), imm);
      set_imm8(instr, 2, static_cast<std::uint64_t>(imm));
      return instr;
    }
  }
  return std::nullopt;
}

// ------------------------------------------------- exchange / double-shift

// The read-modify-write trio. XCHG against memory carries an implicit LOCK, and all three have a
// second write-back that a plain ALU handler doesn't, which is exactly the part worth comparing.
[[nodiscard]] std::optional<Instruction> gen_xchg(Ctx& c) {
  static constexpr std::array<Code, 4> kRmR = {Code::XCHG_RM8_R8, Code::XCHG_RM16_R16,
                                                Code::XCHG_RM32_R32, Code::XCHG_RM64_R64};
  static constexpr std::array<Code, 3> kShort = {Code::XCHG_R16_AX, Code::XCHG_R32_EAX,
                                                  Code::XCHG_R64_RAX};
  c.flags_mask = 0;  // XCHG defines no flags
  if (rand_int(c.rng, 0, 3) == 0) {
    // The 0x90+r short form. Register 0 is excluded because `xchg rax,rax` is NOP's own encoding,
    // which is a decode question rather than an exchange one.
    const Width w = width16_32_64(c.rng);
    const int r = rand_int(c.rng, 1, 15);
    const Register acc = w == Width::W16 ? Register::AX : w == Width::W32 ? Register::EAX : Register::RAX;
    return InstructionFactory::with2(kShort[static_cast<std::size_t>(widx16_32_64(w))], reg_of(w, r), acc);
  }
  const Width w = static_cast<Width>(rand_int(c.rng, 0, 3));
  const Code code = kRmR[static_cast<std::size_t>(w)];
  const int s = pick_reg_index(c.rng);
  if (rand_int(c.rng, 0, 2) == 0) {
    c.touches_memory = true;
    return InstructionFactory::with2(code, mem_operand(random_disp8(c.rng)), reg_of(w, s));
  }
  return InstructionFactory::with2(code, reg_of(w, pick_reg_index(c.rng)), reg_of(w, s));
}

[[nodiscard]] std::optional<Instruction> gen_xadd_cmpxchg(Ctx& c) {
  static constexpr std::array<Code, 4> kXadd = {Code::XADD_RM8_R8, Code::XADD_RM16_R16,
                                                 Code::XADD_RM32_R32, Code::XADD_RM64_R64};
  static constexpr std::array<Code, 4> kCmpxchg = {Code::CMPXCHG_RM8_R8, Code::CMPXCHG_RM16_R16,
                                                    Code::CMPXCHG_RM32_R32, Code::CMPXCHG_RM64_R64};
  const Width w = static_cast<Width>(rand_int(c.rng, 0, 3));
  const bool is_xadd = rand_int(c.rng, 0, 1) == 0;
  const Code code = (is_xadd ? kXadd : kCmpxchg)[static_cast<std::size_t>(w)];
  const int s = pick_reg_index(c.rng);
  if (rand_int(c.rng, 0, 2) == 0) {
    c.touches_memory = true;
    return InstructionFactory::with2(code, mem_operand(random_disp8(c.rng)), reg_of(w, s));
  }
  // CMPXCHG compares against the accumulator, and a fully random one never matches. Force the
  // equal case half the time so the ZF-set branch and its accumulator write-back both get hit.
  if (!is_xadd && rand_int(c.rng, 0, 1) == 0) {
    const int d = pick_reg_index(c.rng);
    if (d != 0) {
      const auto shared = random_interesting_u64(c.rng);
      c.force_gpr[0] = shared;
      c.force_gpr[static_cast<std::size_t>(d)] = shared;
    }
    return InstructionFactory::with2(code, reg_of(w, d), reg_of(w, s));
  }
  return InstructionFactory::with2(code, reg_of(w, pick_reg_index(c.rng)), reg_of(w, s));
}

[[nodiscard]] std::optional<Instruction> gen_shld_shrd(Ctx& c) {
  static constexpr std::array<Code, 3> kShldCl = {Code::SHLD_RM16_R16_CL, Code::SHLD_RM32_R32_CL,
                                                   Code::SHLD_RM64_R64_CL};
  static constexpr std::array<Code, 3> kShldImm = {Code::SHLD_RM16_R16_IMM8, Code::SHLD_RM32_R32_IMM8,
                                                    Code::SHLD_RM64_R64_IMM8};
  static constexpr std::array<Code, 3> kShrdCl = {Code::SHRD_RM16_R16_CL, Code::SHRD_RM32_R32_CL,
                                                   Code::SHRD_RM64_R64_CL};
  static constexpr std::array<Code, 3> kShrdImm = {Code::SHRD_RM16_R16_IMM8, Code::SHRD_RM32_R32_IMM8,
                                                    Code::SHRD_RM64_R64_IMM8};
  // OF is only defined for a 1-bit shift and AF is undefined whenever a shift happens, so neither
  // is comparable against a randomized count.
  c.flags_mask &= ~(0x0800ull | 0x0010ull);
  const Width w = width16_32_64(c.rng);
  const auto wi = static_cast<std::size_t>(widx16_32_64(w));
  const bool left = rand_int(c.rng, 0, 1) == 0;
  const bool use_cl = rand_int(c.rng, 0, 1) == 0;
  const int s = pick_reg_index(c.rng);
  const bool use_mem = rand_int(c.rng, 0, 2) == 0;
  const Code code = use_cl ? (left ? kShldCl : kShrdCl)[wi] : (left ? kShldImm : kShrdImm)[wi];

  // A 16-bit double-shift masks its count to 5 bits, so a random one routinely exceeds the operand
  // size -- which Intel leaves explicitly undefined, in the result AND the flags. Comparing that
  // against hardware reports two equally-correct engines as diverging, so keep 16-bit counts in
  // range and let the wider forms stay fully random (their masked counts never exceed the operand).
  const bool narrow = w == Width::W16;
  if (use_cl) {
    if (narrow) { c.force_gpr[1] = static_cast<std::uint64_t>(rand_int(c.rng, 0, 15)); }
    if (use_mem) {
      c.touches_memory = true;
      return InstructionFactory::with3(code, mem_operand(random_disp8(c.rng)), reg_of(w, s), Register::CL);
    }
    return InstructionFactory::with3(code, reg_of(w, pick_reg_index(c.rng)), reg_of(w, s), Register::CL);
  }
  std::int32_t imm = static_cast<std::int32_t>(random_imm(c.rng, 8));
  if (rand_int(c.rng, 0, 4) != 0) imm &= 0x3F;
  if (narrow) imm &= 0x0F;
  if (use_mem) {
    c.touches_memory = true;
    auto instr = InstructionFactory::with3(code, mem_operand(random_disp8(c.rng)), reg_of(w, s), imm);
    set_imm8(instr, 2, static_cast<std::uint64_t>(imm));
    return instr;
  }
  auto instr = InstructionFactory::with3(code, reg_of(w, pick_reg_index(c.rng)), reg_of(w, s), imm);
  set_imm8(instr, 2, static_cast<std::uint64_t>(imm));
  return instr;
}

// ------------------------------------------------------- privileged (ring 0)
//
// These were held out of kFamilies for a long time because seven had no concept of a privilege
// level, so every one of them produced the same uninformative "seven ran it, hardware #GP'd"
// mismatch. Seven models CPL now (it reads the CS selector's low two bits) and the lanes set
// CS to a ring-3 selector to match the hardware lane, so these finally compare something real:
// they are the only coverage the privilege gates have against actual silicon.

// Every one of these deterministically #GPs in our ring-3 sandbox before
// doing anything real -- see project notes (Hardware Lane Safety) for why
// that makes them safe to generate, unlike SYSCALL/SYSENTER/INT-n.
[[nodiscard]] std::optional<Instruction> gen_privileged(Ctx& c) {
  c.flags_mask = 0;  // none of these define/depend on the ALU status flags
  const int pick = rand_int(c.rng, 0, 9);
  switch (pick) {
    case 0: return InstructionFactory::with(Code::CLI);
    case 1: return InstructionFactory::with(Code::STI);
    case 2: return InstructionFactory::with(Code::HLT);
    case 3: return InstructionFactory::with(Code::CLTS);
    case 4: return InstructionFactory::with(Code::INVD);
    case 5: return InstructionFactory::with(Code::WBINVD);
    case 6: return InstructionFactory::with(Code::SWAPGS);
    case 7: return InstructionFactory::with(Code::XSETBV);
    case 8: return InstructionFactory::with(Code::RDMSR);
    case 9: return InstructionFactory::with(Code::WRMSR);
    default: break;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<Instruction> gen_privileged_movcrdr(Ctx& c) {
  c.flags_mask = 0;
  static constexpr std::array<Register, 5> kCr = {Register::CR0, Register::CR2, Register::CR3, Register::CR4, Register::CR8};
  static constexpr std::array<Register, 8> kDr = {Register::DR0, Register::DR1, Register::DR2, Register::DR3,
                                                    Register::DR4, Register::DR5, Register::DR6, Register::DR7};
  const bool is_cr = rand_int(c.rng, 0, 1) == 0;
  const bool read_dir = rand_int(c.rng, 0, 1) == 0;  // gpr <- CR/DR, vs CR/DR <- gpr
  const int gp = pick_reg_index(c.rng);
  if (is_cr) {
    const Register crreg = kCr[static_cast<std::size_t>(rand_int(c.rng, 0, 4))];
    return read_dir ? InstructionFactory::with2(Code::MOV_R64_CR, reg_of(Width::W64, gp), crreg)
                     : InstructionFactory::with2(Code::MOV_CR_R64, crreg, reg_of(Width::W64, gp));
  }
  const Register drreg = kDr[static_cast<std::size_t>(rand_int(c.rng, 0, 7))];
  return read_dir ? InstructionFactory::with2(Code::MOV_R64_DR, reg_of(Width::W64, gp), drreg)
                   : InstructionFactory::with2(Code::MOV_DR_R64, drreg, reg_of(Width::W64, gp));
}

// ------------------------------------------------------------------- x87
//
// Two things about the FPU environment are deliberately pinned rather than randomized:
//
//   * All six exception masks stay SET (FCW[5:0] = 0x3F). An unmasked x87 exception does not fault
//     on the instruction that raised it -- the #MF is deferred to the next non-control x87
//     instruction -- so in a single-step harness it would either never fire at all or fire inside a
//     completely unrelated later test case, since the hardware lane's victim process is shared.
//     Masked is also the only mode in which every operation still produces a defined result to
//     compare instead of leaving its destination untouched.
//   * Precision control stays at extended (FCW[9:8] = 3). seven has no precision-control handling
//     anywhere, so randomizing it would turn every arithmetic case into the same single known
//     divergence rather than finding anything new.
//
// Rounding control IS randomized across all four modes -- that path is new and nothing else
// validates it against silicon.
//
// Left out on purpose, see project notes: FNSTENV/FNSAVE (seven documents FIP/FCS/FDP/FDS as
// permanently zero, so the stored image differs from hardware's on every single case and says
// nothing new), FLDENV/FRSTOR (they load a guest-controlled control word out of the random scratch
// page, which can unmask an exception mid-run and defer an #MF into a later test case), FXSAVE/
// FXRSTOR (a 512-byte image does not fit the 256-byte compared window), FBLD/FBSTP (random bytes
// are not valid packed BCD and the result of feeding invalid digits is undefined), and FICOM/
// FICOMP (absent from seven's handled-code table entirely -- an unimplemented opcode, not a
// semantics question a fuzzer answers).
//
// One shape below IS generated but is not comparable when it happens to come up, and there is no
// way to know that before running it: FPREM/FPREM1 with an exponent difference of 64 or more reduce
// by an implementation-defined number of bits (the SDM only pins it to "between 32 and 63") and
// report the result as incomplete via C2. Two engines choosing differently there are both correct,
// so read a C2-set FPREM/FPREM1 finding as noise unless something besides ST(0) also differs.

constexpr std::array<Register, 8> kRegsSt = {
    Register::ST0, Register::ST1, Register::ST2, Register::ST3,
    Register::ST4, Register::ST5, Register::ST6, Register::ST7,
};

[[nodiscard]] Register st_of(int idx) { return kRegsSt[static_cast<std::size_t>(idx)]; }

struct X87Raw {
  std::uint16_t signexp;
  std::uint64_t signif;
};

// Includes the four encodings that exist only in this format -- pseudo-NaN, pseudo-infinity,
// unnormal, pseudo-denormal -- which no f32/f64 operand pool can reach, and which hardware treats
// as a class of their own rather than as NaNs or normals.
constexpr std::array<X87Raw, 24> kX87Pool = {{
    {0x0000, 0x0000000000000000ull},  // +0
    {0x8000, 0x0000000000000000ull},  // -0
    {0x3FFF, 0x8000000000000000ull},  // +1
    {0xBFFF, 0x8000000000000000ull},  // -1
    {0x4000, 0x8000000000000000ull},  // +2
    {0x4000, 0xC000000000000000ull},  // +3
    {0x4002, 0xA000000000000000ull},  // +10
    {0x400B, 0xB400000000000000ull},  // +5760, an exact integer FRNDINT/FIST round-trip
    {0x7FFF, 0x8000000000000000ull},  // +inf
    {0xFFFF, 0x8000000000000000ull},  // -inf
    {0x7FFF, 0xC000000000000000ull},  // QNaN
    {0xFFFF, 0xC000000000000000ull},  // the real indefinite, i.e. the masked-#IA result itself
    {0x7FFF, 0x8000000000000001ull},  // SNaN
    {0x7FFF, 0x4000000000000000ull},  // pseudo-NaN, integer bit clear
    {0x7FFF, 0x0000000000000000ull},  // pseudo-infinity
    {0x4000, 0x4000000000000000ull},  // unnormal
    {0x0000, 0x8000000000000000ull},  // pseudo-denormal, integer bit set at exponent 0
    {0x0000, 0x0000000000000001ull},  // smallest denormal
    {0x0000, 0x7FFFFFFFFFFFFFFFull},  // largest denormal
    {0x0001, 0x8000000000000000ull},  // smallest normal
    {0x7FFE, 0xFFFFFFFFFFFFFFFFull},  // largest normal
    {0x3FFF, 0xC90FDAA22168C235ull},  // pi/2, where the trig argument reduction turns over
    {0x403E, 0x8000000000000000ull},  // 2^63, where the trig operand range check starts rejecting
    {0x4040, 0x8000000000000000ull},  // 2^65, comfortably past it
}};

[[nodiscard]] X87Raw random_x87_value(std::mt19937_64& rng) {
  if (rand_int(rng, 0, 99) < 15) {
    std::uniform_int_distribution<std::uint64_t> full(0, UINT64_MAX);
    return X87Raw{static_cast<std::uint16_t>(rand_int(rng, 0, 0xFFFF)), full(rng)};
  }
  X87Raw v = kX87Pool[static_cast<std::size_t>(rand_int(rng, 0, static_cast<int>(kX87Pool.size()) - 1))];
  if (rand_int(rng, 0, 3) == 0) v.signexp ^= 0x8000u;  // same magnitude, other sign
  return v;
}

constexpr std::array<std::uint32_t, 14> kF32Pool = {
    0x00000000u, 0x80000000u, 0x3F800000u, 0xBF800000u, 0x00800000u, 0x007FFFFFu, 0x00000001u,
    0x7F7FFFFFu, 0x7F800000u, 0xFF800000u, 0x7FC00000u, 0x7FA00000u, 0x40490FDBu, 0x5F000000u,
};
constexpr std::array<std::uint64_t, 14> kF64Pool = {
    0x0000000000000000ull, 0x8000000000000000ull, 0x3FF0000000000000ull, 0xBFF0000000000000ull,
    0x0010000000000000ull, 0x000FFFFFFFFFFFFFull, 0x0000000000000001ull, 0x7FEFFFFFFFFFFFFFull,
    0x7FF0000000000000ull, 0xFFF0000000000000ull, 0x7FF8000000000000ull, 0x7FF4000000000000ull,
    0x400921FB54442D18ull, 0x43E0000000000000ull,
};
constexpr std::array<std::uint64_t, 12> kX87IntPool = {
    0ull, 1ull, 2ull, 100ull, 0x7FFFull, 0x8000ull, 0xFFFFull, 0x7FFFFFFFull,
    0x80000000ull, 0x7FFFFFFFFFFFFFFFull, 0x8000000000000000ull, 0xFFFFFFFFFFFFFFFFull,
};

void plant_bytes(Ctx& c, std::int8_t disp, const std::uint8_t* src, std::size_t size) {
  Ctx::ForcedBytes fb{};
  fb.offset = static_cast<std::size_t>(disp);
  fb.size = size;
  for (std::size_t i = 0; i < size && i < fb.value.size(); ++i) fb.value[i] = src[i];
  c.force_data_bytes = fb;
}

void plant_int(Ctx& c, std::int8_t disp, std::uint64_t value, std::size_t size) {
  std::array<std::uint8_t, 10> bytes{};
  for (std::size_t i = 0; i < size; ++i) bytes[i] = static_cast<std::uint8_t>(value >> (8 * i));
  plant_bytes(c, disp, bytes.data(), size);
}

// A quarter of the time nothing is planted and the operand keeps the random scratch bytes, which is
// its own kind of coverage: for m80 in particular that is how the unnormal and pseudo-NaN encodings
// come up as memory sources rather than only as register contents.
void plant_fp_operand(Ctx& c, std::int8_t disp, int size) {
  if (rand_int(c.rng, 0, 3) == 0) return;
  if (size == 4) {
    plant_int(c, disp, kF32Pool[static_cast<std::size_t>(rand_int(c.rng, 0, static_cast<int>(kF32Pool.size()) - 1))], 4);
  } else if (size == 8) {
    plant_int(c, disp, kF64Pool[static_cast<std::size_t>(rand_int(c.rng, 0, static_cast<int>(kF64Pool.size()) - 1))], 8);
  } else {
    const X87Raw v = random_x87_value(c.rng);
    std::array<std::uint8_t, 10> bytes{};
    for (int i = 0; i < 8; ++i) bytes[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(v.signif >> (8 * i));
    bytes[8] = static_cast<std::uint8_t>(v.signexp);
    bytes[9] = static_cast<std::uint8_t>(v.signexp >> 8);
    plant_bytes(c, disp, bytes.data(), 10);
  }
}

void plant_int_operand(Ctx& c, std::int8_t disp, int size) {
  if (rand_int(c.rng, 0, 3) == 0) return;
  plant_int(c, disp, kX87IntPool[static_cast<std::size_t>(rand_int(c.rng, 0, static_cast<int>(kX87IntPool.size()) - 1))],
            static_cast<std::size_t>(size));
}

// Every x87 family funnels through this so nothing forgets to ask next() for a populated stack.
void x87_setup(Ctx& c, int needs) {
  c.uses_x87 = true;
  if (needs > c.x87_needs) c.x87_needs = needs;
}

struct X87MemForm {
  Code code;
  int size;
};

constexpr std::array<X87MemForm, 6> kX87Load = {{
    {Code::FLD_M32FP, 4}, {Code::FLD_M64FP, 8}, {Code::FLD_M80FP, 10},
    {Code::FILD_M16INT, 2}, {Code::FILD_M32INT, 4}, {Code::FILD_M64INT, 8},
}};

[[nodiscard]] std::optional<Instruction> gen_x87_load(Ctx& c) {
  x87_setup(c, 0);
  c.touches_memory = true;
  const X87MemForm f = kX87Load[static_cast<std::size_t>(rand_int(c.rng, 0, static_cast<int>(kX87Load.size()) - 1))];
  const std::int8_t disp = random_disp8(c.rng);
  if (f.code == Code::FILD_M16INT || f.code == Code::FILD_M32INT || f.code == Code::FILD_M64INT) {
    plant_int_operand(c, disp, f.size);
  } else {
    plant_fp_operand(c, disp, f.size);
  }
  return InstructionFactory::with1(f.code, mem_operand(disp));
}

constexpr std::array<X87MemForm, 11> kX87Store = {{
    {Code::FST_M32FP, 4}, {Code::FST_M64FP, 8},
    {Code::FSTP_M32FP, 4}, {Code::FSTP_M64FP, 8}, {Code::FSTP_M80FP, 10},
    {Code::FIST_M16INT, 2}, {Code::FIST_M32INT, 4},
    {Code::FISTP_M16INT, 2}, {Code::FISTP_M32INT, 4}, {Code::FISTP_M64INT, 8},
    {Code::FISTTP_M32INT, 4},
}};

[[nodiscard]] std::optional<Instruction> gen_x87_store(Ctx& c) {
  x87_setup(c, 1);
  c.touches_memory = true;
  const X87MemForm f = kX87Store[static_cast<std::size_t>(rand_int(c.rng, 0, static_cast<int>(kX87Store.size()) - 1))];
  return InstructionFactory::with1(f.code, mem_operand(random_disp8(c.rng)));
}

// The pop forms name ST(i) as the destination and ST(0) as the source, the non-pop ST-ST forms come
// in both directions, and the memory forms always target ST(0).
struct X87ArithGroup {
  Code st0_sti;
  Code sti_st0;
  Code stip_st0;
  Code m32;
  Code m64;
  Code m16int;
  Code m32int;
};

constexpr std::array<X87ArithGroup, 6> kX87Arith = {{
    {Code::FADD_ST0_STI, Code::FADD_STI_ST0, Code::FADDP_STI_ST0, Code::FADD_M32FP, Code::FADD_M64FP,
     Code::FIADD_M16INT, Code::FIADD_M32INT},
    {Code::FMUL_ST0_STI, Code::FMUL_STI_ST0, Code::FMULP_STI_ST0, Code::FMUL_M32FP, Code::FMUL_M64FP,
     Code::FIMUL_M16INT, Code::FIMUL_M32INT},
    {Code::FSUB_ST0_STI, Code::FSUB_STI_ST0, Code::FSUBP_STI_ST0, Code::FSUB_M32FP, Code::FSUB_M64FP,
     Code::FISUB_M16INT, Code::FISUB_M32INT},
    {Code::FSUBR_ST0_STI, Code::FSUBR_STI_ST0, Code::FSUBRP_STI_ST0, Code::FSUBR_M32FP, Code::FSUBR_M64FP,
     Code::FISUBR_M16INT, Code::FISUBR_M32INT},
    {Code::FDIV_ST0_STI, Code::FDIV_STI_ST0, Code::FDIVP_STI_ST0, Code::FDIV_M32FP, Code::FDIV_M64FP,
     Code::FIDIV_M16INT, Code::FIDIV_M32INT},
    {Code::FDIVR_ST0_STI, Code::FDIVR_STI_ST0, Code::FDIVRP_STI_ST0, Code::FDIVR_M32FP, Code::FDIVR_M64FP,
     Code::FIDIVR_M16INT, Code::FIDIVR_M32INT},
}};

[[nodiscard]] std::optional<Instruction> gen_x87_arith(Ctx& c) {
  const X87ArithGroup& g = kX87Arith[static_cast<std::size_t>(rand_int(c.rng, 0, static_cast<int>(kX87Arith.size()) - 1))];
  const int form = rand_int(c.rng, 0, 6);
  if (form <= 2) {
    const int i = rand_int(c.rng, 0, 7);
    x87_setup(c, i + 1);
    if (form == 0) return InstructionFactory::with2(g.st0_sti, Register::ST0, st_of(i));
    if (form == 1) return InstructionFactory::with2(g.sti_st0, st_of(i), Register::ST0);
    return InstructionFactory::with2(g.stip_st0, st_of(i), Register::ST0);
  }
  x87_setup(c, 1);
  c.touches_memory = true;
  const std::int8_t disp = random_disp8(c.rng);
  switch (form) {
    case 3: plant_fp_operand(c, disp, 4); return InstructionFactory::with1(g.m32, mem_operand(disp));
    case 4: plant_fp_operand(c, disp, 8); return InstructionFactory::with1(g.m64, mem_operand(disp));
    case 5: plant_int_operand(c, disp, 2); return InstructionFactory::with1(g.m16int, mem_operand(disp));
    default: plant_int_operand(c, disp, 4); return InstructionFactory::with1(g.m32int, mem_operand(disp));
  }
}

// Exactly-specified unary operations only. The transcendentals get their own family below because
// bit-exactness against Intel's microcode is a different question from correctness.
constexpr std::array<Code, 4> kX87Unary1 = {Code::FSQRT, Code::FRNDINT, Code::FABS, Code::FCHS};
constexpr std::array<Code, 4> kX87Unary2 = {Code::FSCALE, Code::FPREM, Code::FPREM1, Code::FXTRACT};

[[nodiscard]] std::optional<Instruction> gen_x87_unary(Ctx& c) {
  // FXTRACT reads one register but pushes a second result, so it wants a slot free as well.
  if (rand_int(c.rng, 0, 1) == 0) {
    x87_setup(c, 1);
    return InstructionFactory::with(kX87Unary1[static_cast<std::size_t>(rand_int(c.rng, 0, 3))]);
  }
  const Code code = kX87Unary2[static_cast<std::size_t>(rand_int(c.rng, 0, 3))];
  x87_setup(c, code == Code::FXTRACT ? 1 : 2);
  return InstructionFactory::with(code);
}

constexpr std::array<Code, 8> kX87Transcendental = {
    Code::FSIN, Code::FCOS, Code::FSINCOS, Code::FPTAN,
    Code::F2XM1, Code::FYL2X, Code::FYL2XP1, Code::FPATAN,
};

[[nodiscard]] std::optional<Instruction> gen_x87_transcendental(Ctx& c) {
  const Code code = kX87Transcendental[static_cast<std::size_t>(rand_int(c.rng, 0, 7))];
  const bool two_operand = code == Code::FYL2X || code == Code::FYL2XP1 || code == Code::FPATAN;
  x87_setup(c, two_operand ? 2 : 1);
  return InstructionFactory::with(code);
}

[[nodiscard]] std::optional<Instruction> gen_x87_compare(Ctx& c) {
  const int form = rand_int(c.rng, 0, 9);
  if (form == 0) {
    x87_setup(c, 1);
    return InstructionFactory::with(Code::FTST);
  }
  if (form == 1) {
    x87_setup(c, 1);
    return InstructionFactory::with(Code::FXAM);
  }
  if (form == 2) {
    x87_setup(c, 2);
    return InstructionFactory::with(rand_int(c.rng, 0, 1) == 0 ? Code::FCOMPP : Code::FUCOMPP);
  }
  if (form <= 4) {
    x87_setup(c, 1);
    c.touches_memory = true;
    const bool m64 = rand_int(c.rng, 0, 1) == 0;
    const bool pop = rand_int(c.rng, 0, 1) == 0;
    const std::int8_t disp = random_disp8(c.rng);
    plant_fp_operand(c, disp, m64 ? 8 : 4);
    const Code code = m64 ? (pop ? Code::FCOMP_M64FP : Code::FCOM_M64FP)
                          : (pop ? Code::FCOMP_M32FP : Code::FCOM_M32FP);
    return InstructionFactory::with1(code, mem_operand(disp));
  }
  // The DCD0/DCD8/DED0 entries are the undocumented alias encodings of the same three operations;
  // hardware runs them, so they are worth putting through the decoder as well.
  static constexpr std::array<Code, 11> kStSt = {
      Code::FCOM_ST0_STI,   Code::FCOMP_ST0_STI,  Code::FUCOM_ST0_STI,  Code::FUCOMP_ST0_STI,
      Code::FCOMI_ST0_STI,  Code::FCOMIP_ST0_STI, Code::FUCOMI_ST0_STI, Code::FUCOMIP_ST0_STI,
      Code::FCOM_ST0_STI_DCD0, Code::FCOMP_ST0_STI_DCD8, Code::FCOMP_ST0_STI_DED0,
  };
  const int i = rand_int(c.rng, 0, 7);
  x87_setup(c, i + 1);
  return InstructionFactory::with2(kStSt[static_cast<std::size_t>(rand_int(c.rng, 0, 10))], Register::ST0, st_of(i));
}

[[nodiscard]] std::optional<Instruction> gen_x87_move(Ctx& c) {
  const int form = rand_int(c.rng, 0, 5);
  const int i = rand_int(c.rng, 0, 7);
  switch (form) {
    case 0:
      x87_setup(c, i + 1);
      return InstructionFactory::with1(Code::FLD_STI, st_of(i));
    case 1: {
      static constexpr std::array<Code, 5> kStore = {Code::FST_STI, Code::FSTP_STI, Code::FSTP_STI_DFD0,
                                                      Code::FSTP_STI_DFD8, Code::FSTPNCE_STI};
      x87_setup(c, 1);
      return InstructionFactory::with1(kStore[static_cast<std::size_t>(rand_int(c.rng, 0, 4))], st_of(i));
    }
    case 2: {
      static constexpr std::array<Code, 3> kXchg = {Code::FXCH_ST0_STI, Code::FXCH_ST0_STI_DDC8,
                                                     Code::FXCH_ST0_STI_DFC8};
      x87_setup(c, i + 1);
      return InstructionFactory::with2(kXchg[static_cast<std::size_t>(rand_int(c.rng, 0, 2))], Register::ST0, st_of(i));
    }
    case 3:
      x87_setup(c, 0);
      return InstructionFactory::with1(rand_int(c.rng, 0, 1) == 0 ? Code::FFREE_STI : Code::FFREEP_STI, st_of(i));
    case 4: {
      static constexpr std::array<Code, 8> kConst = {Code::FLD1, Code::FLDZ,   Code::FLDPI,  Code::FLDL2T,
                                                      Code::FLDL2E, Code::FLDLG2, Code::FLDLN2, Code::FNOP};
      x87_setup(c, 0);
      return InstructionFactory::with(kConst[static_cast<std::size_t>(rand_int(c.rng, 0, 7))]);
    }
    default: {
      static constexpr std::array<Code, 8> kCmov = {
          Code::FCMOVB_ST0_STI,  Code::FCMOVE_ST0_STI,  Code::FCMOVBE_ST0_STI, Code::FCMOVU_ST0_STI,
          Code::FCMOVNB_ST0_STI, Code::FCMOVNE_ST0_STI, Code::FCMOVNBE_ST0_STI, Code::FCMOVNU_ST0_STI};
      x87_setup(c, i + 1);
      return InstructionFactory::with2(kCmov[static_cast<std::size_t>(rand_int(c.rng, 0, 7))], Register::ST0, st_of(i));
    }
  }
}

[[nodiscard]] std::optional<Instruction> gen_x87_control(Ctx& c) {
  const int form = rand_int(c.rng, 0, 6);
  x87_setup(c, 0);
  switch (form) {
    case 0:
      return InstructionFactory::with(Code::FNSTSW_AX);
    case 1:
      c.touches_memory = true;
      return InstructionFactory::with1(Code::FNSTSW_M2BYTE, mem_operand(random_disp8(c.rng)));
    case 2:
      c.touches_memory = true;
      return InstructionFactory::with1(Code::FNSTCW_M2BYTE, mem_operand(random_disp8(c.rng)));
    case 3: {
      // The loaded word is planted rather than left random for the same reason FLDENV is out
      // entirely: a random 16 bits unmasks exceptions, and an unmasked x87 exception in this
      // harness surfaces inside some later test case instead of this one. Bit 12 (infinity
      // control) is architecturally ignored but still stored, so it rides along to check that both
      // engines keep it rather than normalizing it away.
      c.touches_memory = true;
      const std::int8_t disp = random_disp8(c.rng);
      const auto cw = static_cast<std::uint16_t>(0x037Fu | (static_cast<unsigned>(rand_int(c.rng, 0, 3)) << 10) |
                                                  (rand_int(c.rng, 0, 3) == 0 ? 0x1000u : 0u));
      plant_int(c, disp, cw, 2);
      return InstructionFactory::with1(Code::FLDCW_M2BYTE, mem_operand(disp));
    }
    case 4:
      return InstructionFactory::with(Code::FNCLEX);
    case 5:
      return InstructionFactory::with(Code::FNINIT);
    default:
      return InstructionFactory::with(rand_int(c.rng, 0, 1) == 0 ? Code::FINCSTP : Code::FDECSTP);
  }
}

// -------------------------------------------------------------- dispatch

using GenFn = std::optional<Instruction> (*)(Ctx&);
constexpr std::array<GenFn, 47> kFamilies = {
    gen_alu, gen_test, gen_unary, gen_shift, gen_mov, gen_movx, gen_movsxd,
    gen_pushpop, gen_lea, gen_jcc, gen_jmp, gen_call, gen_ret, gen_setcc,
    gen_cmovcc, gen_bt, gen_rmsrc, gen_bswap,
    gen_loop, gen_xchg, gen_xadd_cmpxchg, gen_shld_shrd, gen_movbe_crc32_nop, gen_moffs,
    gen_string_ops,
    gen_bmi,
    gen_muldiv, gen_imul_multi,
    gen_simd_shuffle, gen_simd_logic, gen_simd_pack, gen_simd_shift, gen_simd_fp,        gen_vex_simd,
    gen_sse_arith, gen_packed_int, gen_sse_move,
    gen_privileged, gen_privileged_movcrdr,
    gen_x87_load, gen_x87_store, gen_x87_arith, gen_x87_unary, gen_x87_compare, gen_x87_move,
    gen_x87_control,
    gen_x87_transcendental,
};

[[nodiscard]] std::string hex_dump(const std::vector<std::uint8_t>& b) {
  static constexpr char kHex[] = "0123456789ABCDEF";
  std::string s;
  s.reserve(b.size() * 3);
  for (std::size_t i = 0; i < b.size(); ++i) {
    if (i != 0) s.push_back(' ');
    s.push_back(kHex[b[i] >> 4]);
    s.push_back(kHex[b[i] & 0xF]);
  }
  return s;
}

}  // namespace

TestCase InstructionGenerator::next() {
  for (int attempt = 0; attempt < 200; ++attempt) {
    try {
      Ctx c{rng_, false};
      const GenFn fn = kFamilies[static_cast<std::size_t>(rand_int(rng_, 0, static_cast<int>(kFamilies.size()) - 1))];
      const std::optional<Instruction> instr = fn(c);
      if (!instr.has_value()) { ++discarded_; continue; }

      Encoder enc(64);
      const auto res = enc.encode(*instr, kCodeBase);
      if (!res) { ++discarded_; continue; }
      std::vector<std::uint8_t> bytes = enc.take_buffer();
      if (bytes.empty()) { ++discarded_; continue; }

      TestCase tc;
      tc.bytes = bytes;
      tc.touches_memory = c.touches_memory;
      tc.flags_mask = c.flags_mask;
      tc.text = hex_dump(bytes);

      for (int i = 0; i < 16; ++i) tc.initial.gpr[static_cast<std::size_t>(i)] = random_interesting_u64(rng_);
      tc.initial.gpr[4] = kStackTop;  // RSP always valid
      if (tc.touches_memory) tc.initial.gpr[7] = kDataBase;  // RDI always the scratch base
      for (int i = 0; i < 16; ++i) {
        if (c.force_gpr[static_cast<std::size_t>(i)].has_value()) {
          tc.initial.gpr[static_cast<std::size_t>(i)] = *c.force_gpr[static_cast<std::size_t>(i)];
        }
      }
      for (int i = 0; i < 16; ++i) {
        tc.initial.xmm_lo[static_cast<std::size_t>(i)] = random_interesting_u64(rng_);
        tc.initial.xmm_hi[static_cast<std::size_t>(i)] = random_interesting_u64(rng_);
      }

      if (c.uses_x87) {
        X87State& x = tc.initial.x87;
        // Exception masks stay set and PC stays at extended; only RC moves. See the x87 section's
        // header comment for both reasons.
        x.control_word = static_cast<std::uint16_t>(0x037Fu | (static_cast<unsigned>(rand_int(rng_, 0, 3)) << 10));
        const int top = rand_int(rng_, 0, 7);
        // The stack normally holds at least what the instruction reads, so most cases exercise the
        // operation itself. The rest deliberately come up short: a masked stack underflow has its
        // own fully defined behaviour (indefinite QNaN, C1 cleared, SF set) that is just as much
        // part of the spec as the arithmetic is.
        const int depth = rand_int(rng_, 0, 99) < 10
                              ? rand_int(rng_, 0, 8)
                              : c.x87_needs + rand_int(rng_, 0, 8 - c.x87_needs);
        auto sw = static_cast<std::uint16_t>(static_cast<unsigned>(top) << 11);
        if (rand_int(rng_, 0, 3) == 0) {
          // Stale condition codes and sticky exception flags, so "does this clear what it does not
          // define" gets exercised alongside "does it set what it does". ES and B are left alone:
          // they describe an exception this corpus deliberately never lets happen.
          static constexpr std::uint16_t kStaleBits[] = {0x0001, 0x0002, 0x0004, 0x0008, 0x0010,
                                                          0x0020, 0x0040, 0x0100, 0x0200, 0x0400, 0x4000};
          for (const auto bit : kStaleBits) {
            if (rand_int(rng_, 0, 5) == 0) sw |= bit;
          }
        }
        x.status_word = sw;
        std::uint16_t tw = 0xFFFF;  // every slot empty, then punch in the live ones
        for (int i = 0; i < depth; ++i) {
          const X87Raw v = random_x87_value(rng_);
          x.signexp[static_cast<std::size_t>(i)] = v.signexp;
          x.signif[static_cast<std::size_t>(i)] = v.signif;
          // The tag word is indexed by physical register while the values above are ST-relative,
          // and the tag itself has to be the one hardware would derive from the value, since that
          // is exactly what FXRSTOR does with the abridged byte the hardware lane hands it.
          const int phys = (top + i) & 0x7;
          tw = static_cast<std::uint16_t>(tw & ~(0x3u << (2 * phys)));
          tw = static_cast<std::uint16_t>(tw | (static_cast<unsigned>(x87_classify_tag(v.signexp, v.signif))
                                                << (2 * phys)));
        }
        x.tag_word = tw;
      }

      std::uint64_t fl = 0x2;  // reserved bit 1
      static constexpr std::uint64_t kRandFlags[] = {0x1, 0x4, 0x10, 0x40, 0x80, 0x400, 0x800};
      for (const auto bitv : kRandFlags) {
        if (rand_int(rng_, 0, 1) == 0) fl |= bitv;
      }
      fl |= 0x200;  // IF=1 by harness convention; excluded from comparison regardless
      tc.initial.rflags = fl;

      for (auto& b : tc.data_seed) b = static_cast<std::uint8_t>(rand_int(rng_, 0, 255));

      if (c.force_data_qword.has_value() && c.force_data_qword->offset + 8 <= tc.data_seed.size()) {
        for (std::size_t b = 0; b < 8; ++b) {
          tc.data_seed[c.force_data_qword->offset + b] =
              static_cast<std::uint8_t>(c.force_data_qword->value >> (8 * b));
        }
      }

      if (c.force_data_bytes.has_value() &&
          c.force_data_bytes->offset + c.force_data_bytes->size <= tc.data_seed.size()) {
        for (std::size_t b = 0; b < c.force_data_bytes->size; ++b) {
          tc.data_seed[c.force_data_bytes->offset + b] = c.force_data_bytes->value[b];
        }
      }

      if (c.bsf_bsr_dest.has_value()) {
        std::uint64_t src_val = 0;
        bool have_src = false;
        if (c.bsf_bsr_src_reg.has_value()) {
          src_val = tc.initial.gpr[static_cast<std::size_t>(*c.bsf_bsr_src_reg)];
          have_src = true;
        } else if (c.bsf_bsr_src_mem_disp.has_value()) {
          const auto off = static_cast<std::size_t>(*c.bsf_bsr_src_mem_disp);
          if (off + static_cast<std::size_t>(c.bsf_bsr_width_bytes) <= tc.data_seed.size()) {
            for (int b = 0; b < c.bsf_bsr_width_bytes; ++b) {
              src_val |= static_cast<std::uint64_t>(tc.data_seed[off + static_cast<std::size_t>(b)])
                         << (8 * b);
            }
            have_src = true;
          }
        }
        if (have_src && c.bsf_bsr_width_bytes > 0 && c.bsf_bsr_width_bytes < 8) {
          src_val &= (std::uint64_t{1} << (8 * c.bsf_bsr_width_bytes)) - 1;
        }
        if (have_src && src_val == 0) {
          tc.gpr_compare_mask &= ~(1u << *c.bsf_bsr_dest);
        }
      }

      return tc;
    } catch (...) {
      ++discarded_;
      continue;
    }
  }
  // Unreachable in practice — a bare NOP keeps the pipeline alive if it ever happens.
  TestCase tc;
  tc.bytes = {0x90};
  tc.text = "90";
  tc.initial.gpr[4] = kStackTop;
  tc.initial.rflags = 0x202;
  return tc;
}

}  // namespace sf
