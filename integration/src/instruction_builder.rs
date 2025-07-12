use openvm_instructions::{instruction::Instruction, riscv, LocalOpcode, SystemOpcode, VmOpcode};
use openvm_rv32im_transpiler::Rv32LoadStoreOpcode;
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_womir_transpiler::{
    AllocateFrameOpcode, BaseAluOpcode, ConstOpcodes, CopyIntoFrameOpcode, JaafOpcode, JumpOpcode,
    LessThanOpcode, Phantom, ShiftOpcode, WomSystemOpcodes,
};

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

#[allow(dead_code)]
pub fn addi<F: PrimeField32>(rd: usize, rs1: usize, imm: usize) -> Instruction<F> {
    instr_i(BaseAluOpcode::ADD.global_opcode().as_usize(), rd, rs1, imm)
}

pub fn sub<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(BaseAluOpcode::SUB.global_opcode().as_usize(), rd, rs1, rs2)
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

pub fn shl<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(ShiftOpcode::SLL.global_opcode().as_usize(), rd, rs1, rs2)
}

pub fn shr_u<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(ShiftOpcode::SRL.global_opcode().as_usize(), rd, rs1, rs2)
}

pub fn shr_s<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(ShiftOpcode::SRA.global_opcode().as_usize(), rd, rs1, rs2)
}

pub fn lt_u<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(
        LessThanOpcode::SLTU.global_opcode().as_usize(),
        rd,
        rs1,
        rs2,
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
pub fn allocate_frame_imm<F: PrimeField32>(target_reg: usize, amount_imm: usize) -> Instruction<F> {
    Instruction::new(
        AllocateFrameOpcode::ALLOCATE_FRAME.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * target_reg), // a: target_reg
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * amount_imm), // b: amount_imm
        F::ZERO,                                                              // c: (not used)
        F::ZERO,                                                              // d: (not used)
        F::ZERO,                                                              // e: (not used)
        F::ONE,                                                               // f: enabled
        F::ZERO,                                                              // g: imm sign
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

#[allow(dead_code)]
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

pub fn read_u32<F: PrimeField32>(rd: usize) -> Instruction<F> {
    Instruction::new(
        WomSystemOpcodes::PHANTOM.global_opcode(),
        F::ZERO,
        F::ZERO,
        F::from_canonical_u32(Phantom::HintInput as u32),
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
    )

    /*
    Instruction::new(
                opcode: SystemOpcode::PHANTOM.global_opcode(),
                a,
                b,
                c: F::from_canonical_u32((discriminant.0 as u32) | ((c_upper as u32) << 16)),
                ..Default::default()
            }

        Instruction::phantom(
            F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rd),
            F::ZERO,
            0,
        )
    */
}
