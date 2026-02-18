use openvm_instructions::{LocalOpcode, SystemOpcode, VmOpcode, instruction::Instruction, riscv};
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_womir_transpiler::{
    AllocateFrameOpcode, BaseAlu64Opcode, BaseAluOpcode, ConstOpcodes, CopyIntoFrameOpcode,
    Eq64Opcode, EqOpcode, HintStoreOpcode, JaafOpcode, JumpOpcode, LessThan64Opcode,
    LessThanOpcode, MulOpcode, Phantom, Shift64Opcode, ShiftOpcode,
};

use openvm_rv32im_transpiler::Rv32LoadStoreOpcode as LoadStoreOpcode;

use crate::womir_translation::{ERROR_ABORT_CODE, ERROR_CODE_OFFSET};

/// Immediate in the format expected by the ALU adapter.
#[derive(Debug, Clone, Copy)]
pub struct AluImm(u32);

impl From<i16> for AluImm {
    fn from(value: i16) -> Self {
        // ALU adapter expects the 16 bits value in the lower 2 bytes,
        // the sign extension on the 3rd byte, and the 4th byte to
        // be zeroed.
        let value = value as i32 as u32 & 0xff_ff_ff;
        AluImm(value)
    }
}

impl TryFrom<u32> for AluImm {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        i16::try_from(value as i32)
            .map(AluImm::from)
            .map_err(|_| ())
    }
}

pub fn instr_r<F: PrimeField32>(
    opcode: usize,
    rd: usize,
    rs1: usize,
    rs2: usize,
) -> Instruction<F> {
    Instruction::new(
        VmOpcode::from_usize(opcode),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rd),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs1),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs2),
        F::ONE,
        F::ONE,
        F::ZERO,
        F::ZERO,
    )
}

pub fn instr_i<F: PrimeField32>(
    opcode: usize,
    rd: usize,
    rs1: usize,
    imm: impl Into<AluImm>,
) -> Instruction<F> {
    let imm: AluImm = imm.into();
    Instruction::new(
        VmOpcode::from_usize(opcode),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rd),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs1),
        F::from_canonical_u32(imm.0),
        F::ONE,
        F::ZERO,
        F::ZERO,
        F::ZERO,
    )
}

pub fn add<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(BaseAluOpcode::ADD.global_opcode().as_usize(), rd, rs1, rs2)
}

#[allow(dead_code)]
pub fn add_imm<F: PrimeField32>(rd: usize, rs1: usize, imm: impl Into<AluImm>) -> Instruction<F> {
    instr_i(BaseAluOpcode::ADD.global_opcode().as_usize(), rd, rs1, imm)
}

pub fn sub<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(BaseAluOpcode::SUB.global_opcode().as_usize(), rd, rs1, rs2)
}

#[cfg(test)]
pub fn mul<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(MulOpcode::MUL.global_opcode().as_usize(), rd, rs1, rs2)
}

pub fn mul_imm<F: PrimeField32>(rd: usize, rs1: usize, imm: impl Into<AluImm>) -> Instruction<F> {
    instr_i(MulOpcode::MUL.global_opcode().as_usize(), rd, rs1, imm)
}

#[cfg(test)]
pub fn mul_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    use openvm_womir_transpiler::Mul64Opcode;
    instr_r(Mul64Opcode::MUL.global_opcode().as_usize(), rd, rs1, rs2)
}

#[cfg(test)]
pub fn mul_imm_64<F: PrimeField32>(rd: usize, rs1: usize, imm: AluImm) -> Instruction<F> {
    use openvm_womir_transpiler::Mul64Opcode;
    instr_i(Mul64Opcode::MUL.global_opcode().as_usize(), rd, rs1, imm)
}

#[cfg(test)]
pub fn div<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    use openvm_womir_transpiler::DivRemOpcode;
    instr_r(DivRemOpcode::DIV.global_opcode().as_usize(), rd, rs1, rs2)
}

#[cfg(test)]
pub fn divu<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    use openvm_womir_transpiler::DivRemOpcode;
    instr_r(DivRemOpcode::DIVU.global_opcode().as_usize(), rd, rs1, rs2)
}

#[cfg(test)]
pub fn rems<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    use openvm_womir_transpiler::DivRemOpcode;
    instr_r(DivRemOpcode::REM.global_opcode().as_usize(), rd, rs1, rs2)
}

#[cfg(test)]
pub fn remu<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    use openvm_womir_transpiler::DivRemOpcode;
    instr_r(DivRemOpcode::REMU.global_opcode().as_usize(), rd, rs1, rs2)
}

#[cfg(test)]
pub fn div_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    use openvm_womir_transpiler::DivRem64Opcode;
    instr_r(DivRem64Opcode::DIV.global_opcode().as_usize(), rd, rs1, rs2)
}

#[cfg(test)]
pub fn divu_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    use openvm_womir_transpiler::DivRem64Opcode;
    instr_r(
        DivRem64Opcode::DIVU.global_opcode().as_usize(),
        rd,
        rs1,
        rs2,
    )
}

#[cfg(test)]
pub fn rems_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    use openvm_womir_transpiler::DivRem64Opcode;
    instr_r(DivRem64Opcode::REM.global_opcode().as_usize(), rd, rs1, rs2)
}

#[cfg(test)]
pub fn remu_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    use openvm_womir_transpiler::DivRem64Opcode;
    instr_r(
        DivRem64Opcode::REMU.global_opcode().as_usize(),
        rd,
        rs1,
        rs2,
    )
}

#[cfg(test)]
pub fn xor<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(BaseAluOpcode::XOR.global_opcode().as_usize(), rd, rs1, rs2)
}

pub fn or<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(BaseAluOpcode::OR.global_opcode().as_usize(), rd, rs1, rs2)
}

#[cfg(test)]
pub fn and<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(BaseAluOpcode::AND.global_opcode().as_usize(), rd, rs1, rs2)
}

pub fn and_imm<F: PrimeField32>(rd: usize, rs1: usize, imm: impl Into<AluImm>) -> Instruction<F> {
    instr_i(BaseAluOpcode::AND.global_opcode().as_usize(), rd, rs1, imm)
}

pub fn shl<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(ShiftOpcode::SLL.global_opcode().as_usize(), rd, rs1, rs2)
}

pub fn shl_imm<F: PrimeField32>(rd: usize, rs1: usize, imm: impl Into<AluImm>) -> Instruction<F> {
    instr_i(ShiftOpcode::SLL.global_opcode().as_usize(), rd, rs1, imm)
}

pub fn shr_u<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(ShiftOpcode::SRL.global_opcode().as_usize(), rd, rs1, rs2)
}

pub fn shr_u_imm<F: PrimeField32>(rd: usize, rs1: usize, imm: impl Into<AluImm>) -> Instruction<F> {
    instr_i(ShiftOpcode::SRL.global_opcode().as_usize(), rd, rs1, imm)
}

pub fn shr_s_imm<F: PrimeField32>(rd: usize, rs1: usize, imm: impl Into<AluImm>) -> Instruction<F> {
    instr_i(ShiftOpcode::SRA.global_opcode().as_usize(), rd, rs1, imm)
}

pub fn shl_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(Shift64Opcode::SLL.global_opcode().as_usize(), rd, rs1, rs2)
}

pub fn shl_imm_64<F: PrimeField32>(
    rd: usize,
    rs1: usize,
    imm: impl Into<AluImm>,
) -> Instruction<F> {
    instr_i(Shift64Opcode::SLL.global_opcode().as_usize(), rd, rs1, imm)
}

pub fn shr_u_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(Shift64Opcode::SRL.global_opcode().as_usize(), rd, rs1, rs2)
}

pub fn shr_s_imm_64<F: PrimeField32>(
    rd: usize,
    rs1: usize,
    imm: impl Into<AluImm>,
) -> Instruction<F> {
    instr_i(Shift64Opcode::SRA.global_opcode().as_usize(), rd, rs1, imm)
}

pub fn shr_u_imm_64<F: PrimeField32>(
    rd: usize,
    rs1: usize,
    imm: impl Into<AluImm>,
) -> Instruction<F> {
    instr_i(Shift64Opcode::SRL.global_opcode().as_usize(), rd, rs1, imm)
}

pub fn lt_u<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(
        LessThanOpcode::SLTU.global_opcode().as_usize(),
        rd,
        rs1,
        rs2,
    )
}

pub fn lt_u_imm<F: PrimeField32>(rd: usize, rs1: usize, imm: impl Into<AluImm>) -> Instruction<F> {
    instr_i(
        LessThanOpcode::SLTU.global_opcode().as_usize(),
        rd,
        rs1,
        imm,
    )
}

pub fn lt_s<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(LessThanOpcode::SLT.global_opcode().as_usize(), rd, rs1, rs2)
}

pub fn gt_u<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    // lt_u, but swapped
    lt_u(rd, rs2, rs1)
}

pub fn gt_s<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    // lt_s, but swapped
    lt_s(rd, rs2, rs1)
}

pub fn lt_u_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(
        LessThan64Opcode::SLTU.global_opcode().as_usize(),
        rd,
        rs1,
        rs2,
    )
}

pub fn lt_s_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(
        LessThan64Opcode::SLT.global_opcode().as_usize(),
        rd,
        rs1,
        rs2,
    )
}

pub fn gt_u_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    // lt_u, but swapped
    lt_u_64(rd, rs2, rs1)
}

pub fn gt_s_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    // lt_s, but swapped
    lt_s_64(rd, rs2, rs1)
}

pub fn eq<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(EqOpcode::EQ.global_opcode().as_usize(), rd, rs1, rs2)
}

pub fn eq_imm<F: PrimeField32>(rd: usize, rs1: usize, imm: impl Into<AluImm>) -> Instruction<F> {
    instr_i(EqOpcode::EQ.global_opcode().as_usize(), rd, rs1, imm)
}

pub fn eq_imm_64<F: PrimeField32>(rd: usize, rs1: usize, imm: impl Into<AluImm>) -> Instruction<F> {
    instr_i(Eq64Opcode::EQ.global_opcode().as_usize(), rd, rs1, imm)
}

pub fn neq_imm<F: PrimeField32>(rd: usize, rs1: usize, imm: impl Into<AluImm>) -> Instruction<F> {
    instr_i(EqOpcode::NEQ.global_opcode().as_usize(), rd, rs1, imm)
}

pub fn const_32_imm<F: PrimeField32>(
    target_reg: usize,
    imm_lo: u16,
    imm_hi: u16,
) -> Instruction<F> {
    Instruction::new(
        ConstOpcodes::CONST32.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * target_reg), // a: target_reg
        F::from_canonical_usize(imm_lo as usize),                             // b: low 16 bits
        // of the immediate
        F::from_canonical_usize(imm_hi as usize), // c: high 16 bits
        // of the immediate
        F::ZERO, // d: (not used)
        F::ZERO, // e: (not used)
        F::ONE,  // f: enabled
        F::ZERO, // g: (not used)
    )
}

#[cfg(test)]
pub fn add_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(
        BaseAlu64Opcode::ADD.global_opcode().as_usize(),
        rd,
        rs1,
        rs2,
    )
}

#[cfg(test)]
pub fn add_imm_64<F: PrimeField32>(
    rd: usize,
    rs1: usize,
    imm: impl Into<AluImm>,
) -> Instruction<F> {
    instr_i(
        BaseAlu64Opcode::ADD.global_opcode().as_usize(),
        rd,
        rs1,
        imm,
    )
}

pub fn sub_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(
        BaseAlu64Opcode::SUB.global_opcode().as_usize(),
        rd,
        rs1,
        rs2,
    )
}

#[cfg(test)]
pub fn sub_imm_64<F: PrimeField32>(
    rd: usize,
    rs1: usize,
    imm: impl Into<AluImm>,
) -> Instruction<F> {
    instr_i(
        BaseAlu64Opcode::SUB.global_opcode().as_usize(),
        rd,
        rs1,
        imm,
    )
}

#[cfg(test)]
pub fn xor_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(
        BaseAlu64Opcode::XOR.global_opcode().as_usize(),
        rd,
        rs1,
        rs2,
    )
}

#[cfg(test)]
pub fn xor_imm_64<F: PrimeField32>(
    rd: usize,
    rs1: usize,
    imm: impl Into<AluImm>,
) -> Instruction<F> {
    instr_i(
        BaseAlu64Opcode::XOR.global_opcode().as_usize(),
        rd,
        rs1,
        imm,
    )
}

pub fn or_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(BaseAlu64Opcode::OR.global_opcode().as_usize(), rd, rs1, rs2)
}

#[cfg(test)]
pub fn or_imm_64<F: PrimeField32>(rd: usize, rs1: usize, imm: impl Into<AluImm>) -> Instruction<F> {
    instr_i(BaseAlu64Opcode::OR.global_opcode().as_usize(), rd, rs1, imm)
}

#[cfg(test)]
pub fn and_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(
        BaseAlu64Opcode::AND.global_opcode().as_usize(),
        rd,
        rs1,
        rs2,
    )
}

pub fn and_imm_64<F: PrimeField32>(
    rd: usize,
    rs1: usize,
    imm: impl Into<AluImm>,
) -> Instruction<F> {
    instr_i(
        BaseAlu64Opcode::AND.global_opcode().as_usize(),
        rd,
        rs1,
        imm,
    )
}

/// JAAF instruction: Jump And Activate Frame
/// Sets PC from immediate and FP from register
pub fn jaaf<F: PrimeField32>(to_pc_imm: usize, to_fp_reg: usize) -> Instruction<F> {
    Instruction::new(
        JaafOpcode::JAAF.global_opcode(),
        F::ZERO,                                                             // a: (not used)
        F::ZERO,                                                             // b: (not used)
        F::ZERO,                                                             // c: (not used)
        F::from_canonical_usize(to_pc_imm),                                  // d: to_pc_imm
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * to_fp_reg), // e: to_fp_reg
        F::ONE,                                                              // f: enabled
        F::ZERO, // g: imm sign (0 for positive)
    )
}

/// JAAF_SAVE instruction: Jump And Activate Frame, Save FP
/// Sets PC from immediate, FP from register, and saves current FP
pub fn jaaf_save<F: PrimeField32>(
    save_fp: usize,
    to_pc_imm: usize,
    to_fp_reg: usize,
) -> Instruction<F> {
    Instruction::new(
        JaafOpcode::JAAF_SAVE.global_opcode(),
        F::ZERO,                                                             // a: (not used)
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * save_fp),   // b: save_fp
        F::ZERO,                                                             // c: (not used)
        F::from_canonical_usize(to_pc_imm),                                  // d: to_pc_imm
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * to_fp_reg), // e: to_fp_reg
        F::ONE,                                                              // f: enabled
        F::ZERO, // g: imm sign (0 for positive)
    )
}

/// RET instruction: Return (restore PC and FP from registers)
/// Sets PC and FP from registers
pub fn ret<F: PrimeField32>(to_pc_reg: usize, to_fp_reg: usize) -> Instruction<F> {
    Instruction::new(
        JaafOpcode::RET.global_opcode(),
        F::ZERO,                                                             // a: (not used)
        F::ZERO,                                                             // b: (not used)
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * to_pc_reg), // c: to_pc_reg
        F::ZERO,                                                             // d: (not used)
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * to_fp_reg), // e: to_fp_reg
        F::ONE,                                                              // f: enabled
        F::ZERO,                                                             // g: imm sign
    )
}

/// CALL instruction: Call function (save PC and FP, jump to label)
/// Saves current PC and FP, then sets PC from immediate and FP from register
pub fn call<F: PrimeField32>(
    save_pc: usize,
    save_fp: usize,
    to_pc_imm: usize,
    to_fp_reg: usize,
) -> Instruction<F> {
    Instruction::new(
        JaafOpcode::CALL.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * save_pc), // a: rd1 (save PC here)
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * save_fp), // b: rd2 (save FP here)
        F::ZERO,                                                           // c: rs1 (not used)
        F::from_canonical_usize(to_pc_imm), // d: immediate for PC target
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * to_fp_reg), // e: rs2 (new FP)
        F::ONE,                             // f: enabled
        F::ZERO,                            // g: imm sign (0 for positive)
    )
}

/// CALL_INDIRECT instruction: Call function indirect (save PC and FP, jump to register)
/// Saves current PC and FP, then sets PC and FP from registers
#[allow(dead_code)]
pub fn call_indirect<F: PrimeField32>(
    save_pc: usize,
    save_fp: usize,
    to_pc_reg: usize,
    to_fp_reg: usize,
) -> Instruction<F> {
    Instruction::new(
        JaafOpcode::CALL_INDIRECT.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * save_pc),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * save_fp),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * to_pc_reg),
        F::ZERO, // d: immediate (not used)
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * to_fp_reg),
        F::ONE,  // f: enabled
        F::ZERO, // g: imm sign
    )
}

/// ALLOCATE_FRAME_I instruction: Allocate frame from immediate size and return pointer
/// target_reg receives the allocated pointer, amount_imm is the amount to allocate
pub fn allocate_frame_imm<F: PrimeField32>(target_reg: usize, amount_imm: usize) -> Instruction<F> {
    Instruction::new(
        AllocateFrameOpcode::ALLOCATE_FRAME.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * target_reg), // a: target_reg
        F::from_canonical_usize(amount_imm),                                  // b: amount_imm
        F::ZERO,                                                              // c: amount_reg
        F::ZERO, // d: whether to use the amount immediate (ZERO) or register (ONE)
        F::ZERO, // e: (not used)
        F::ONE,  // f: enabled
        F::ZERO, // g: imm sign
    )
}

/// ALLOCATE_FRAME_V instruction: Allocate frame from register and return pointer
/// target_reg receives the allocated pointer, amount_reg is the register containing the amount to allocate
pub fn allocate_frame_reg<F: PrimeField32>(target_reg: usize, amount_reg: usize) -> Instruction<F> {
    Instruction::new(
        AllocateFrameOpcode::ALLOCATE_FRAME.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * target_reg), // a: target_reg
        F::ZERO,                                                              // b: amount_imm
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * amount_reg), // c: amount_reg
        F::ONE,  // d: whether to use the amount immediate (ZERO) or register (ONE)
        F::ZERO, // e: (not used)
        F::ONE,  // f: enabled
        F::ZERO, // g: imm sign
    )
}

/// COPY_INTO_FRAME instruction: Copy value into frame-relative address
/// dest_fp is the frame pointer, src_value is the value to copy, dest_offset is the offset
/// Writes src_value content to [dest_fp[dest_offset]]
pub fn copy_into_frame<F: PrimeField32>(
    target_reg: usize,
    src_reg: usize,
    target_fp: usize,
) -> Instruction<F> {
    Instruction::new(
        CopyIntoFrameOpcode::COPY_INTO_FRAME.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * target_reg), // a: target_reg
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * src_reg),    // b: register
        // containing value to copy
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * target_fp), // c: future fp to be used as reference for target_reg
        F::ZERO,                                                             // d: (not used)
        F::ZERO,                                                             // e: (not used)
        F::ONE,                                                              // f: enabled
        F::ZERO,                                                             // g: (not used)
    )
}

/// COPY_FROM_FRAME instruction: Copy value from frame-relative address.
/// src_fp is the frame pointer, src_value is the value to copy.
/// Writes [src_fp + src_value] content to [fp + target_reg].
pub fn copy_from_frame<F: PrimeField32>(
    target_reg: usize,
    src_reg: usize,
    src_fp: usize,
) -> Instruction<F> {
    Instruction::new(
        CopyIntoFrameOpcode::COPY_FROM_FRAME.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * target_reg), // a: target_reg
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * src_reg),    // b: register
        // containing value to copy
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * src_fp), // c: other future fp to be used as reference for src_reg
        F::ZERO,                                                          // d: (not used)
        F::ZERO,                                                          // e: (not used)
        F::ONE,                                                           // f: enabled
        F::ZERO,                                                          // g: (not used)
    )
}

/// JUMP instruction: Unconditional jump to immediate PC
pub fn jump<F: PrimeField32>(to_pc_imm: usize) -> Instruction<F> {
    Instruction::new(
        JumpOpcode::JUMP.global_opcode(),
        F::from_canonical_usize(to_pc_imm), // a: to_pc_imm
        F::ZERO,                            // b: (not used)
        F::ZERO,                            // c: (not used)
        F::ZERO,                            // d: (not used)
        F::ZERO,                            // e: (not used)
        F::ONE,                             // f: enabled
        F::ZERO,                            // g: imm sign
    )
}

/// SKIP instruction: Unconditional relative jump to current PC + offset
pub fn skip<F: PrimeField32>(offset_reg: usize) -> Instruction<F> {
    Instruction::new(
        JumpOpcode::SKIP.global_opcode(),
        F::ZERO,                                                              // a: (not used)
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * offset_reg), // b: register with the offset
        F::ZERO,                                                              // c: (not used)
        F::ZERO,                                                              // d: (not used)
        F::ZERO,                                                              // e: (not used)
        F::ONE,                                                               // f: enabled
        F::ZERO,                                                              // g: imm sign
    )
}

/// JUMP_IF instruction: Conditional jump to immediate PC if condition != 0
pub fn jump_if<F: PrimeField32>(condition_reg: usize, to_pc_imm: usize) -> Instruction<F> {
    Instruction::new(
        JumpOpcode::JUMP_IF.global_opcode(),
        F::from_canonical_usize(to_pc_imm), // a: to_pc_imm
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * condition_reg), // b: condition_reg
        F::ZERO,                            // c: (not used)
        F::ZERO,                            // d: (not used)
        F::ZERO,                            // e: (not used)
        F::ONE,                             // f: enabled
        F::ZERO,                            // g: imm sign
    )
}

/// JUMP_IF_ZERO instruction: Conditional jump to immediate PC if condition == 0
pub fn jump_if_zero<F: PrimeField32>(condition_reg: usize, to_pc_imm: usize) -> Instruction<F> {
    Instruction::new(
        JumpOpcode::JUMP_IF_ZERO.global_opcode(),
        F::from_canonical_usize(to_pc_imm), // a: to_pc_imm
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * condition_reg), // b: condition_reg
        F::ZERO,                            // c: (not used)
        F::ZERO,                            // d: (not used)
        F::ZERO,                            // e: (not used)
        F::ONE,                             // f: enabled
        F::ZERO,                            // g: imm sign
    )
}

/// LOADW instruction: Load word from memory
/// rd = MEM[rs1 + imm]
pub fn loadw<F: PrimeField32>(rd: usize, rs1: usize, imm: u32) -> Instruction<F> {
    Instruction::new(
        LoadStoreOpcode::LOADW.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rd), // a: rd
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs1), // b: rs1
        F::from_canonical_u32(imm & 0xFFFF),                          // c: imm (lower 16 bits)
        F::ONE,                                                       // d: register address space
        F::from_canonical_usize(2), // e: memory address space (2 for word)
        F::ONE,                     // f: enabled
        F::from_canonical_u32(imm >> 16), // g: imm (higher 16 bits)
    )
}

/// STOREW instruction: Store word to memory
/// MEM[rs1 + imm] = rs2
pub fn storew<F: PrimeField32>(value: usize, base_address: usize, imm: u32) -> Instruction<F> {
    Instruction::new(
        LoadStoreOpcode::STOREW.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * value), // a: rs2 (data to store)
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * base_address), // b: rs1 (base address)
        F::from_canonical_u32(imm & 0xFFFF), // c: imm (lower 16 bits)
        F::ONE,                              // d: register address space
        F::from_canonical_usize(2),          // e: memory address space (2 for word)
        F::ONE,                              // f: enabled
        F::from_canonical_u32(imm >> 16),    // g: imm (higher 16 bits)
    )
}

/// LOADB: load byte from memory
/// rd = MEM[rs1 + imm] (sign-extended)
pub fn loadb<F: PrimeField32>(rd: usize, rs1: usize, imm: u32) -> Instruction<F> {
    Instruction::new(
        LoadStoreOpcode::LOADB.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rd), // a: rd
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs1), // b: rs1
        F::from_canonical_u32(imm & 0xFFFF),                          // c: imm (lower 16 bits)
        F::ONE,                                                       // d: register address space
        F::from_canonical_usize(2), // e: memory address space (2 for byte)
        F::ONE,                     // f: enabled
        F::from_canonical_u32(imm >> 16), // g: imm (higher 16 bits)
    )
}

/// LOADBU instruction: Load byte unsigned from memory
/// rd = MEM[rs1 + imm] (zero-extended)
pub fn loadbu<F: PrimeField32>(rd: usize, rs1: usize, imm: u32) -> Instruction<F> {
    Instruction::new(
        LoadStoreOpcode::LOADBU.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rd), // a: rd
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs1), // b: rs1
        F::from_canonical_u32(imm & 0xFFFF),                          // c: imm (lower 16 bits)
        F::ONE,                                                       // d: register address space
        F::from_canonical_usize(2), // e: memory address space (2 for byte)
        F::ONE,                     // f: enabled
        F::from_canonical_u32(imm >> 16), // g: imm (higher 16 bits)
    )
}

/// LOADH: load halfword from memory
/// rd = MEM[rs1 + imm] (sign-extended)
#[allow(unused)]
pub fn loadh<F: PrimeField32>(rd: usize, rs1: usize, imm: u32) -> Instruction<F> {
    Instruction::new(
        LoadStoreOpcode::LOADH.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rd), // a: rd
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs1), // b: rs1
        F::from_canonical_u32(imm & 0xFFFF),                          // c: imm (lower 16 bits)
        F::ONE,                                                       // d: register address space
        F::from_canonical_usize(2), // e: memory address space (2 for halfword)
        F::ONE,                     // f: enabled
        F::from_canonical_u32(imm >> 16), // g: imm (higher 16 bits)
    )
}

/// LOADHU instruction: Load halfword unsigned from memory
/// rd = MEM[rs1 + imm] (zero-extended)
pub fn loadhu<F: PrimeField32>(rd: usize, rs1: usize, imm: u32) -> Instruction<F> {
    Instruction::new(
        LoadStoreOpcode::LOADHU.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rd), // a: rd
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs1), // b: rs1
        F::from_canonical_u32(imm & 0xFFFF),                          // c: imm (lower 16 bits)
        F::ONE,                                                       // d: register address space
        F::from_canonical_usize(2), // e: memory address space (2 for halfword)
        F::ONE,                     // f: enabled
        F::from_canonical_u32(imm >> 16), // g: imm (higher 16 bits)
    )
}

/// STOREB instruction: Store byte to memory
/// MEM[rs1 + imm] = rs2 (lowest byte)
pub fn storeb<F: PrimeField32>(rs2: usize, rs1: usize, imm: u32) -> Instruction<F> {
    Instruction::new(
        LoadStoreOpcode::STOREB.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs2), // a: rs2 (data to store)
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs1), // b: rs1 (base address)
        F::from_canonical_u32(imm & 0xFFFF),                           // c: imm (lower 16 bits)
        F::ONE,                                                        // d: register address space
        F::from_canonical_usize(2), // e: memory address space (2 for byte)
        F::ONE,                     // f: enabled
        F::from_canonical_u32(imm >> 16), // g: imm (higher 16 bits)
    )
}

/// STOREH instruction: Store halfword to memory
/// MEM[rs1 + imm] = rs2 (lowest halfword)
pub fn storeh<F: PrimeField32>(rs2: usize, rs1: usize, imm: u32) -> Instruction<F> {
    Instruction::new(
        LoadStoreOpcode::STOREH.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs2), // a: rs2 (data to store)
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs1), // b: rs1 (base address)
        F::from_canonical_u32(imm & 0xFFFF),                           // c: imm (lower 16 bits)
        F::ONE,                                                        // d: register address space
        F::from_canonical_usize(2), // e: memory address space (2 for halfword)
        F::ONE,                     // f: enabled
        F::from_canonical_u32(imm >> 16), // g: imm (higher 16 bits)
    )
}

#[allow(dead_code)]
pub fn reveal<F: PrimeField32>(rs1_data: usize, rd_index: usize) -> Instruction<F> {
    Instruction::new(
        LoadStoreOpcode::STOREW.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs1_data),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rd_index),
        F::ZERO,
        F::ONE,
        F::from_canonical_usize(3),
        F::ONE,
        F::ZERO,
    )
}

#[allow(dead_code)]
pub fn reveal_imm<F: PrimeField32>(
    data_reg: usize,
    output_index_reg: usize,
    output_index_imm: usize,
) -> Instruction<F> {
    Instruction::new(
        LoadStoreOpcode::STOREW.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * data_reg),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * output_index_reg),
        F::from_canonical_usize(output_index_imm),
        F::ONE,
        F::from_canonical_usize(3),
        F::ONE,
        F::ZERO,
    )
}

pub fn trap<F: PrimeField32>(error_code: usize) -> Instruction<F> {
    Instruction::new(
        SystemOpcode::TERMINATE.global_opcode(),
        F::ZERO,
        F::ZERO,
        F::from_canonical_usize(ERROR_CODE_OFFSET as usize + error_code),
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
    )
}

pub fn abort<F: PrimeField32>() -> Instruction<F> {
    Instruction::new(
        SystemOpcode::TERMINATE.global_opcode(),
        F::ZERO,
        F::ZERO,
        F::from_canonical_usize(ERROR_ABORT_CODE as usize),
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
    )
}

#[allow(dead_code)]
pub fn halt<F: PrimeField32>() -> Instruction<F> {
    Instruction::new(
        SystemOpcode::TERMINATE.global_opcode(),
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
    )
}

pub fn prepare_read<F: PrimeField32>() -> Instruction<F> {
    Instruction::new(
        SystemOpcode::PHANTOM.global_opcode(),
        F::ZERO,
        F::ZERO,
        F::from_canonical_u32(Phantom::HintInput as u32),
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
    )
}

pub fn debug_print<F: PrimeField32>(
    mem_ptr_reg: usize,
    amount_reg: usize,
    mem_imm: u16,
) -> Instruction<F> {
    // OpenVM execution splits c into c_hi and c_lo which are passed separately to the trait impl.
    let c = ((mem_imm as usize) << 16) | (Phantom::PrintStr as usize);
    Instruction::from_isize(
        SystemOpcode::PHANTOM.global_opcode(),
        (riscv::RV32_REGISTER_NUM_LIMBS * mem_ptr_reg) as isize,
        (riscv::RV32_REGISTER_NUM_LIMBS * amount_reg) as isize,
        c as isize,
        0,
        0,
    )
}

/// HINT_STOREW: Read one word from hint stream and write to MEM[reg[mem_ptr_reg]]
pub fn hint_storew<F: PrimeField32>(mem_ptr_reg: usize) -> Instruction<F> {
    Instruction::from_isize(
        HintStoreOpcode::HINT_STOREW.global_opcode(),
        0,
        (riscv::RV32_REGISTER_NUM_LIMBS * mem_ptr_reg) as isize,
        0,
        riscv::RV32_REGISTER_AS as isize,
        riscv::RV32_MEMORY_AS as isize,
    )
}

/// HINT_BUFFER: Read num_words words from hint stream and write to MEM[reg[mem_ptr_reg]..]
pub fn hint_buffer<F: PrimeField32>(num_words_reg: usize, mem_ptr_reg: usize) -> Instruction<F> {
    Instruction::from_isize(
        HintStoreOpcode::HINT_BUFFER.global_opcode(),
        (riscv::RV32_REGISTER_NUM_LIMBS * num_words_reg) as isize,
        (riscv::RV32_REGISTER_NUM_LIMBS * mem_ptr_reg) as isize,
        0,
        riscv::RV32_REGISTER_AS as isize,
        riscv::RV32_MEMORY_AS as isize,
    )
}
