// Temporarily here for reference while we migrate to WOM.

use openvm_instructions::{instruction::Instruction, riscv, LocalOpcode, SystemOpcode, VmOpcode};
use openvm_rv32im_transpiler::{BaseAluOpcode, Rv32JalLuiOpcode, Rv32LoadStoreOpcode};
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
    instr_r(BaseAluOpcode::ADD.global_opcode().as_usize(), rd, rs1, rs2)
}

pub fn addi<F: PrimeField32>(rd: usize, rs1: usize, imm: usize) -> Instruction<F> {
    instr_i(BaseAluOpcode::ADD.global_opcode().as_usize(), rd, rs1, imm)
}

pub fn lui<F: PrimeField32>(rd: usize, imm: usize) -> Instruction<F> {
    Instruction::new(
        Rv32JalLuiOpcode::LUI.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rd),
        F::ZERO,
        F::from_canonical_usize(imm),
        F::ONE,
        F::ZERO,
        F::ONE,
        F::ZERO,
    )
}

pub fn reveal<F: PrimeField32>(rs1_data: usize, rd_index: usize) -> Instruction<F> {
    Instruction::new(
        Rv32LoadStoreOpcode::STOREW.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs1_data),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rd_index),
        F::ZERO,
        F::ONE,
        F::from_canonical_usize(3),
        F::ONE,
        F::ZERO,
    )
}

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
