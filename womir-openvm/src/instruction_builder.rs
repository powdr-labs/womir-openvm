use openvm_instructions::{instruction::Instruction, riscv, LocalOpcode, SystemOpcode, VmOpcode};
use openvm_rv32im_transpiler::{BaseAluOpcode, Rv32JalLuiOpcode, Rv32LoadStoreOpcode};
use openvm_rv32im_wom_transpiler::{BaseAluOpcode as BaseAluOpcodeWom, Rv32JaafOpcode};
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

pub fn add_wom<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(
        BaseAluOpcodeWom::ADD.global_opcode().as_usize(),
        rd,
        rs1,
        rs2,
    )
}

pub fn addi_wom<F: PrimeField32>(rd: usize, rs1: usize, imm: usize) -> Instruction<F> {
    instr_i(
        BaseAluOpcodeWom::ADD.global_opcode().as_usize(),
        rd,
        rs1,
        imm,
    )
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

/// JAAF instruction: Jump And Activate Frame
/// rd1: where to save current pc (if needed, use 0 for JAAF)
/// rd2: where to save current fp (if needed, use 0 for JAAF)  
/// rs1: target pc from register (if needed, use 0 for JAAF)
/// imm: target pc from immediate
/// rs2: target fp register
pub fn jaaf<F: PrimeField32>(
    rd1: usize,
    rd2: usize,
    _rs1: usize,
    imm: usize,
    rs2: usize,
) -> Instruction<F> {
    Instruction::new(
        Rv32JaafOpcode::JAAF.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rd1), // a: rd1
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rd2), // b: rd2
        F::from_canonical_usize(imm),                                  // c: immediate for PC target
        F::ONE, // d: address space (1 for registers)
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs2), // e: rs2
        F::ONE, // f: enabled
        F::ZERO, // g: imm sign (0 for positive)
    )
}

/// JAAF_SAVE instruction: Jump And Activate Frame, Save FP
/// Same as JAAF but saves current FP to rd2
pub fn jaaf_save<F: PrimeField32>(
    rd1: usize,
    rd2: usize,
    _rs1: usize,
    imm: usize,
    rs2: usize,
) -> Instruction<F> {
    Instruction::new(
        Rv32JaafOpcode::JAAF_SAVE.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rd1), // a: rd1
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rd2), // b: rd2
        F::from_canonical_usize(imm),                                  // c: immediate for PC target
        F::ONE, // d: address space (1 for registers)
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs2), // e: rs2
        F::ONE, // f: enabled
        F::ZERO, // g: imm sign (0 for positive)
    )
}

/// RET instruction: Return (restore PC and FP from registers)
/// rs1: register containing return PC
/// rs2: register containing saved FP
pub fn ret<F: PrimeField32>(
    rd1: usize,
    rd2: usize,
    rs1: usize,
    _imm: usize,
    rs2: usize,
) -> Instruction<F> {
    Instruction::new(
        Rv32JaafOpcode::RET.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rd1), // a: rd1
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rd2), // b: rd2
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs1), // c: rs1 (PC source)
        F::ONE, // d: address space (1 for registers)
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs2), // e: rs2 (FP source)
        F::ONE, // f: enabled
        F::ZERO, // g: imm sign
    )
}

/// CALL instruction: Call function (save PC and FP, jump to label)
/// rd1: where to save current PC
/// rd2: where to save current FP
/// imm: target PC (label)
/// rs2: target FP register
pub fn call<F: PrimeField32>(
    rd1: usize,
    rd2: usize,
    _rs1: usize,
    imm: usize,
    rs2: usize,
) -> Instruction<F> {
    Instruction::new(
        Rv32JaafOpcode::CALL.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rd1), // a: rd1 (save PC here)
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rd2), // b: rd2 (save FP here)
        F::from_canonical_usize(imm),                                  // c: immediate for PC target
        F::ONE, // d: address space (1 for registers)
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs2), // e: rs2 (new FP)
        F::ONE, // f: enabled
        F::ZERO, // g: imm sign (0 for positive)
    )
}

/// CALL_INDIRECT instruction: Call function indirect (save PC and FP, jump to register)
/// rd1: where to save current PC
/// rd2: where to save current FP
/// rs1: register containing target PC
/// rs2: target FP register
pub fn call_indirect<F: PrimeField32>(
    rd1: usize,
    rd2: usize,
    rs1: usize,
    _imm: usize,
    rs2: usize,
) -> Instruction<F> {
    Instruction::new(
        Rv32JaafOpcode::CALL_INDIRECT.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rd1), // a: rd1 (save PC here)
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rd2), // b: rd2 (save FP here)
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs1), // c: rs1 (PC source)
        F::ONE, // d: address space (1 for registers)
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs2), // e: rs2 (new FP)
        F::ONE, // f: enabled
        F::ZERO, // g: imm sign
    )
}
