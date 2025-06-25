use openvm_instructions::{instruction::Instruction, riscv, VmOpcode};
use openvm_rv32im_transpiler::BaseAluOpcode;
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

pub fn add<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(BaseAluOpcode::ADD as usize, rd, rs1, rs2)
}

pub fn add_wom<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(BaseAluOpcodeWom::ADD as usize, rd, rs1, rs2)
}
