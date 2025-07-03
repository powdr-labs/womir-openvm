use openvm_instructions::{instruction::Instruction, riscv, LocalOpcode, SystemOpcode, VmOpcode};
use openvm_rv32im_transpiler::{Rv32JalLuiOpcode, Rv32LoadStoreOpcode};
use openvm_rv32im_wom_transpiler::{
    BaseAluOpcode as BaseAluOpcodeWom, Rv32AllocateFrameOpcode, Rv32CopyIntoFrameOpcode,
    Rv32JaafOpcode, Rv32JumpOpcode,
};
use openvm_stark_backend::p3_field::PrimeField32;

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
    imm: usize,
) -> Instruction<F> {
    Instruction::new(
        VmOpcode::from_usize(opcode),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rd),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs1),
        F::from_canonical_usize(imm),
        F::ONE,
        F::ZERO,
        F::ZERO,
        F::ZERO,
    )
}

pub fn add<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(
        BaseAluOpcodeWom::ADD.global_opcode().as_usize(),
        rd,
        rs1,
        rs2,
    )
}

pub fn addi<F: PrimeField32>(rd: usize, rs1: usize, imm: usize) -> Instruction<F> {
    instr_i(
        BaseAluOpcodeWom::ADD.global_opcode().as_usize(),
        rd,
        rs1,
        imm,
    )
}

/// JAAF instruction: Jump And Activate Frame
/// Sets PC from immediate and FP from register
pub fn jaaf<F: PrimeField32>(to_pc_imm: usize, to_fp_reg: usize) -> Instruction<F> {
    Instruction::new(
        Rv32JaafOpcode::JAAF.global_opcode(),
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
        Rv32JaafOpcode::JAAF_SAVE.global_opcode(),
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
        Rv32JaafOpcode::RET.global_opcode(),
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
        Rv32JaafOpcode::CALL.global_opcode(),
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
pub fn call_indirect<F: PrimeField32>(
    save_pc: usize,
    save_fp: usize,
    to_pc_reg: usize,
    to_fp_reg: usize,
) -> Instruction<F> {
    Instruction::new(
        Rv32JaafOpcode::CALL_INDIRECT.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * save_pc), // a: rd1 (save PC here)
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * save_fp), // b: rd2 (save FP here)
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * to_pc_reg), // c: rs1 (PC source)
        F::ZERO, // d: immediate (not used)
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * to_fp_reg), // e: rs2 (FP source)
        F::ONE,  // f: enabled
        F::ZERO, // g: imm sign
    )
}

/// ALLOCATE_FRAME instruction: Allocate frame and return pointer
/// target_reg receives the allocated pointer, amount_imm is the amount to allocate
pub fn allocate_frame<F: PrimeField32>(target_reg: usize, amount_imm: usize) -> Instruction<F> {
    Instruction::new(
        Rv32AllocateFrameOpcode::ALLOCATE_FRAME.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * target_reg), // a: target_reg
        F::from_canonical_usize(amount_imm),                                  // b: amount_imm
        F::ZERO,                                                              // c: (not used)
        F::ZERO,                                                              // d: (not used)
        F::ZERO,                                                              // e: (not used)
        F::ONE,                                                               // f: enabled
        F::ZERO,                                                              // g: imm sign
    )
}

/// COPY_INTO_FRAME instruction: Copy value into frame-relative address
/// rd is the offset, rs1 is the value to copy, rs2 is the frame pointer
/// Writes rs1 content to [rs2[rd]]
pub fn copy_into_frame<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    Instruction::new(
        Rv32CopyIntoFrameOpcode::COPY_INTO_FRAME.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs2), // a: rs2 (frame pointer)
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs1), // b: rs1 (value to copy)
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rd), // c: rd (offset, target address)
        F::ZERO,                                                      // d: (not used)
        F::ZERO,                                                      // e: (not used)
        F::ONE,                                                       // f: enabled
        F::ZERO,                                                      // g: imm sign
    )
}

/// JUMP instruction: Unconditional jump to immediate PC
pub fn jump<F: PrimeField32>(to_pc_imm: usize) -> Instruction<F> {
    Instruction::new(
        Rv32JumpOpcode::JUMP.global_opcode(),
        F::from_canonical_usize(to_pc_imm), // a: to_pc_imm
        F::ZERO,                            // b: (not used)
        F::ZERO,                            // c: (not used)
        F::ZERO,                            // d: (not used)
        F::ZERO,                            // e: (not used)
        F::ONE,                             // f: enabled
        F::ZERO,                            // g: imm sign
    )
}

/// JUMP_IF instruction: Conditional jump to immediate PC if condition != 0
pub fn jump_if<F: PrimeField32>(condition_reg: usize, to_pc_imm: usize) -> Instruction<F> {
    Instruction::new(
        Rv32JumpOpcode::JUMP_IF.global_opcode(),
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
        Rv32JumpOpcode::JUMP_IF_ZERO.global_opcode(),
        F::from_canonical_usize(to_pc_imm), // a: to_pc_imm
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * condition_reg), // b: condition_reg
        F::ZERO,                            // c: (not used)
        F::ZERO,                            // d: (not used)
        F::ZERO,                            // e: (not used)
        F::ONE,                             // f: enabled
        F::ZERO,                            // g: imm sign
    )
}
