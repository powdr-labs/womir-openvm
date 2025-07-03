use openvm_instructions::{instruction::Instruction, riscv, LocalOpcode, SystemOpcode, VmOpcode};
use openvm_rv32im_wom_transpiler::BaseAluOpcode as BaseAluOpcodeWom;
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
