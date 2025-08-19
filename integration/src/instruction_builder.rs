use openvm_instructions::{instruction::Instruction, riscv, LocalOpcode, SystemOpcode, VmOpcode};
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_womir_transpiler::{
    AllocateFrameOpcode, BaseAlu64Opcode, BaseAluOpcode, ConstOpcodes, CopyIntoFrameOpcode,
    DivRem64Opcode, DivRemOpcode, Eq64Opcode, EqOpcode, HintStoreOpcode, JaafOpcode, JumpOpcode,
    LessThan64Opcode, LessThanOpcode, LoadStoreOpcode, Mul64Opcode, MulOpcode, Phantom,
    Shift64Opcode, ShiftOpcode,
};

use crate::womir_translation::ERROR_CODE_OFFSET;

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

#[allow(dead_code)]
pub fn instr_i<F: PrimeField32>(opcode: usize, rd: usize, rs1: usize, imm: F) -> Instruction<F> {
    Instruction::new(
        VmOpcode::from_usize(opcode),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rd),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs1),
        imm,
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
pub fn addi<F: PrimeField32>(rd: usize, rs1: usize, imm: F) -> Instruction<F> {
    instr_i(BaseAluOpcode::ADD.global_opcode().as_usize(), rd, rs1, imm)
}

pub fn sub<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(BaseAluOpcode::SUB.global_opcode().as_usize(), rd, rs1, rs2)
}

pub fn mul<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(MulOpcode::MUL.global_opcode().as_usize(), rd, rs1, rs2)
}

pub fn mul_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(Mul64Opcode::MUL.global_opcode().as_usize(), rd, rs1, rs2)
}

pub fn div<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(DivRemOpcode::DIV.global_opcode().as_usize(), rd, rs1, rs2)
}

pub fn divu<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(DivRemOpcode::DIVU.global_opcode().as_usize(), rd, rs1, rs2)
}

pub fn rem<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(DivRemOpcode::REM.global_opcode().as_usize(), rd, rs1, rs2)
}

pub fn remu<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(DivRemOpcode::REMU.global_opcode().as_usize(), rd, rs1, rs2)
}

pub fn div_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(DivRem64Opcode::DIV.global_opcode().as_usize(), rd, rs1, rs2)
}

pub fn divu_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(
        DivRem64Opcode::DIVU.global_opcode().as_usize(),
        rd,
        rs1,
        rs2,
    )
}

pub fn rem_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(DivRem64Opcode::REM.global_opcode().as_usize(), rd, rs1, rs2)
}

pub fn remu_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(
        DivRem64Opcode::REMU.global_opcode().as_usize(),
        rd,
        rs1,
        rs2,
    )
}

pub fn xor<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(BaseAluOpcode::XOR.global_opcode().as_usize(), rd, rs1, rs2)
}

pub fn or<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(BaseAluOpcode::OR.global_opcode().as_usize(), rd, rs1, rs2)
}

pub fn and<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(BaseAluOpcode::AND.global_opcode().as_usize(), rd, rs1, rs2)
}

pub fn andi<F: PrimeField32>(rd: usize, rs1: usize, imm: F) -> Instruction<F> {
    instr_i(BaseAluOpcode::AND.global_opcode().as_usize(), rd, rs1, imm)
}

pub fn shl<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(ShiftOpcode::SLL.global_opcode().as_usize(), rd, rs1, rs2)
}

pub fn shl_imm<F: PrimeField32>(rd: usize, rs1: usize, imm: F) -> Instruction<F> {
    instr_i(ShiftOpcode::SLL.global_opcode().as_usize(), rd, rs1, imm)
}

pub fn shr_u<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(ShiftOpcode::SRL.global_opcode().as_usize(), rd, rs1, rs2)
}

pub fn shr_s<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(ShiftOpcode::SRA.global_opcode().as_usize(), rd, rs1, rs2)
}

pub fn shr_s_imm<F: PrimeField32>(rd: usize, rs1: usize, imm: F) -> Instruction<F> {
    instr_i(ShiftOpcode::SRA.global_opcode().as_usize(), rd, rs1, imm)
}

pub fn shl_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(Shift64Opcode::SLL.global_opcode().as_usize(), rd, rs1, rs2)
}

#[allow(dead_code)]
pub fn shl_imm_64<F: PrimeField32>(rd: usize, rs1: usize, imm: F) -> Instruction<F> {
    instr_i(Shift64Opcode::SLL.global_opcode().as_usize(), rd, rs1, imm)
}

pub fn shr_u_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(Shift64Opcode::SRL.global_opcode().as_usize(), rd, rs1, rs2)
}

pub fn shr_s_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(Shift64Opcode::SRA.global_opcode().as_usize(), rd, rs1, rs2)
}

#[allow(dead_code)]
pub fn shr_s_imm_64<F: PrimeField32>(rd: usize, rs1: usize, imm: F) -> Instruction<F> {
    instr_i(Shift64Opcode::SRA.global_opcode().as_usize(), rd, rs1, imm)
}

pub fn lt_u<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(
        LessThanOpcode::SLTU.global_opcode().as_usize(),
        rd,
        rs1,
        rs2,
    )
}

pub fn lt_u_imm<F: PrimeField32>(rd: usize, rs1: usize, imm: F) -> Instruction<F> {
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

#[allow(dead_code)]
pub fn lt_u_imm_64<F: PrimeField32>(rd: usize, rs1: usize, imm: F) -> Instruction<F> {
    instr_i(
        LessThan64Opcode::SLTU.global_opcode().as_usize(),
        rd,
        rs1,
        imm,
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

pub fn eqi<F: PrimeField32>(rd: usize, rs1: usize, imm: F) -> Instruction<F> {
    instr_i(EqOpcode::EQ.global_opcode().as_usize(), rd, rs1, imm)
}

pub fn neq<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(EqOpcode::NEQ.global_opcode().as_usize(), rd, rs1, rs2)
}

pub fn eq_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(Eq64Opcode::EQ.global_opcode().as_usize(), rd, rs1, rs2)
}

pub fn eqi_64<F: PrimeField32>(rd: usize, rs1: usize, imm: F) -> Instruction<F> {
    instr_i(Eq64Opcode::EQ.global_opcode().as_usize(), rd, rs1, imm)
}

pub fn neq_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(Eq64Opcode::NEQ.global_opcode().as_usize(), rd, rs1, rs2)
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

pub fn add_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(
        BaseAlu64Opcode::ADD.global_opcode().as_usize(),
        rd,
        rs1,
        rs2,
    )
}

#[allow(dead_code)]
pub fn addi_64<F: PrimeField32>(rd: usize, rs1: usize, imm: F) -> Instruction<F> {
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

pub fn xor_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(
        BaseAlu64Opcode::XOR.global_opcode().as_usize(),
        rd,
        rs1,
        rs2,
    )
}

pub fn or_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(BaseAlu64Opcode::OR.global_opcode().as_usize(), rd, rs1, rs2)
}

pub fn and_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(
        BaseAlu64Opcode::AND.global_opcode().as_usize(),
        rd,
        rs1,
        rs2,
    )
}

pub fn andi_64<F: PrimeField32>(rd: usize, rs1: usize, imm: F) -> Instruction<F> {
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
#[allow(dead_code)]
pub fn loadw<F: PrimeField32>(rd: usize, rs1: usize, imm: i32) -> Instruction<F> {
    let imm_unsigned = (imm & 0xFFFF) as usize;
    let imm_sign = if imm < 0 { 1 } else { 0 };

    Instruction::new(
        LoadStoreOpcode::LOADW.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rd), // a: rd
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs1), // b: rs1
        F::from_canonical_usize(imm_unsigned),                        // c: imm (lower 16 bits)
        F::ONE,                                                       // d: register address space
        F::from_canonical_usize(2), // e: memory address space (2 for word)
        F::ONE,                     // f: enabled
        F::from_canonical_usize(imm_sign), // g: imm sign
    )
}

/// STOREW instruction: Store word to memory
/// MEM[rs1 + imm] = rs2
#[allow(dead_code)]
pub fn storew<F: PrimeField32>(rs2: usize, rs1: usize, imm: i32) -> Instruction<F> {
    let imm_unsigned = (imm & 0xFFFF) as usize;
    let imm_sign = if imm < 0 { 1 } else { 0 };

    Instruction::new(
        LoadStoreOpcode::STOREW.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs2), // a: rs2 (data to store)
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs1), // b: rs1 (base address)
        F::from_canonical_usize(imm_unsigned),                         // c: imm (offset)
        F::ONE,                                                        // d: register address space
        F::from_canonical_usize(2), // e: memory address space (2 for word, same as LOADW)
        F::ONE,                     // f: enabled
        F::from_canonical_usize(imm_sign), // g: imm sign
    )
}

/// LOADBU instruction: Load byte unsigned from memory
/// rd = MEM[rs1 + imm] (zero-extended)
#[allow(dead_code)]
pub fn loadbu<F: PrimeField32>(rd: usize, rs1: usize, imm: i32) -> Instruction<F> {
    let imm_unsigned = (imm & 0xFFFF) as usize;
    let imm_sign = if imm < 0 { 1 } else { 0 };

    Instruction::new(
        LoadStoreOpcode::LOADBU.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rd), // a: rd
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs1), // b: rs1
        F::from_canonical_usize(imm_unsigned),                        // c: imm (lower 16 bits)
        F::ONE,                                                       // d: register address space
        F::from_canonical_usize(2), // e: memory address space (2 for byte, same as word)
        F::ONE,                     // f: enabled
        F::from_canonical_usize(imm_sign), // g: imm sign
    )
}

/// LOADHU instruction: Load halfword unsigned from memory
/// rd = MEM[rs1 + imm] (zero-extended)
#[allow(dead_code)]
pub fn loadhu<F: PrimeField32>(rd: usize, rs1: usize, imm: i32) -> Instruction<F> {
    let imm_unsigned = (imm & 0xFFFF) as usize;
    let imm_sign = if imm < 0 { 1 } else { 0 };

    Instruction::new(
        LoadStoreOpcode::LOADHU.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rd), // a: rd
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs1), // b: rs1
        F::from_canonical_usize(imm_unsigned),                        // c: imm (lower 16 bits)
        F::ONE,                                                       // d: register address space
        F::from_canonical_usize(2), // e: memory address space (2 for halfword, same as word)
        F::ONE,                     // f: enabled
        F::from_canonical_usize(imm_sign), // g: imm sign
    )
}

/// STOREB instruction: Store byte to memory
/// MEM[rs1 + imm] = rs2 (lowest byte)
#[allow(dead_code)]
pub fn storeb<F: PrimeField32>(rs2: usize, rs1: usize, imm: i32) -> Instruction<F> {
    let imm_unsigned = (imm & 0xFFFF) as usize;
    let imm_sign = if imm < 0 { 1 } else { 0 };

    Instruction::new(
        LoadStoreOpcode::STOREB.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs2), // a: rs2 (data to store)
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs1), // b: rs1 (base address)
        F::from_canonical_usize(imm_unsigned),                         // c: imm (offset)
        F::ONE,                                                        // d: register address space
        F::from_canonical_usize(2), // e: memory address space (2 for byte, same as word)
        F::ONE,                     // f: enabled
        F::from_canonical_usize(imm_sign), // g: imm sign
    )
}

/// STOREH instruction: Store halfword to memory
/// MEM[rs1 + imm] = rs2 (lowest halfword)
#[allow(dead_code)]
pub fn storeh<F: PrimeField32>(rs2: usize, rs1: usize, imm: i32) -> Instruction<F> {
    let imm_unsigned = (imm & 0xFFFF) as usize;
    let imm_sign = if imm < 0 { 1 } else { 0 };

    Instruction::new(
        LoadStoreOpcode::STOREH.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs2), // a: rs2 (data to store)
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs1), // b: rs1 (base address)
        F::from_canonical_usize(imm_unsigned),                         // c: imm (offset)
        F::ONE,                                                        // d: register address space
        F::from_canonical_usize(2), // e: memory address space (2 for halfword, same as word)
        F::ONE,                     // f: enabled
        F::from_canonical_usize(imm_sign), // g: imm sign
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

#[allow(dead_code)]
pub fn pre_read_u32<F: PrimeField32>() -> Instruction<F> {
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

#[allow(dead_code)]
pub fn read_u32<F: PrimeField32>(rd: usize) -> Instruction<F> {
    Instruction::from_isize(
        HintStoreOpcode::HINT_STOREW.global_opcode(),
        (riscv::RV32_REGISTER_NUM_LIMBS * rd) as isize,
        0,
        0,
        1,
        0,
    )
}
