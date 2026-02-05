use openvm_circuit::{arch::*, system::memory::online::TracingMemory};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_instructions::{
    LocalOpcode,
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_IMM_AS, RV32_REGISTER_AS},
};
use openvm_rv32im_transpiler::LessThanOpcode;
use openvm_stark_backend::p3_field::PrimeField32;
use std::borrow::{Borrow, BorrowMut};
use std::mem::size_of;

// Re-use upstream types
pub use openvm_rv32im_circuit::{
    LessThanCoreAir, LessThanCoreCols, LessThanCoreRecord, LessThanFiller,
};

use crate::adapters::{RV32_REGISTER_NUM_LIMBS, imm_to_bytes};

// Core executor that implements FpPreflightExecutor
#[derive(Clone, Copy, derive_new::new)]
pub struct LessThanCoreExecutor<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub offset: usize,
}

// FpPreflightExecutor implementation for LessThan
impl<F, RA, const NUM_LIMBS: usize, const LIMB_BITS: usize> crate::FpPreflightExecutor<F, RA>
    for LessThanCoreExecutor<NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("{:?}", LessThanOpcode::from_usize(opcode - self.offset))
    }

    fn execute_with_fp(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
        fp: u32,
    ) -> Result<Option<u32>, ExecutionError> {
        debug_assert!(LIMB_BITS <= 8);
        let Instruction {
            opcode, a, b, c, e, ..
        } = instruction;

        // Read first operand using FP
        let rs1_addr = b.as_canonical_u32() + fp;
        let (_counter1, rs1) = unsafe {
            state
                .memory
                .read::<u8, NUM_LIMBS, RV32_REGISTER_NUM_LIMBS>(RV32_REGISTER_AS, rs1_addr)
        };

        // Check if second operand is immediate or register
        let e_u32 = e.as_canonical_u32();
        let rs2 = if e_u32 == RV32_IMM_AS {
            // Immediate value - convert and extend
            let c_u32 = c.as_canonical_u32();
            let imm_bytes = imm_to_bytes(c_u32);
            let mut result = [0u8; NUM_LIMBS];
            result[..4.min(NUM_LIMBS)].copy_from_slice(&imm_bytes[..4.min(NUM_LIMBS)]);
            result
        } else {
            // Read from register using FP
            let rs2_addr = c.as_canonical_u32() + fp;
            let (_counter2, rs2_read) = unsafe {
                state
                    .memory
                    .read::<u8, NUM_LIMBS, RV32_REGISTER_NUM_LIMBS>(RV32_REGISTER_AS, rs2_addr)
            };
            rs2_read
        };

        let local_opcode = opcode.local_opcode_idx(self.offset) as u8;
        let (cmp_result, _, _, _) = run_less_than::<NUM_LIMBS, LIMB_BITS>(
            local_opcode == LessThanOpcode::SLT as u8,
            &rs1,
            &rs2,
        );

        let mut output = [0u8; NUM_LIMBS];
        output[0] = cmp_result as u8;

        // Write result using FP
        let rd_addr = a.as_canonical_u32() + fp;
        unsafe {
            state
                .memory
                .write::<u8, NUM_LIMBS, RV32_REGISTER_NUM_LIMBS>(RV32_REGISTER_AS, rd_addr, output);
        }

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        // FP doesn't change for LessThan
        Ok(None)
    }
}

// PreCompute struct for InterpreterExecutor
#[repr(C)]
#[derive(AlignedBytesBorrow, Clone, Copy)]
struct LessThanPreCompute<const NUM_LIMBS: usize> {
    rd: u32,
    rs1: u32,
    rs2: u32,
    is_slt: bool,
}

// Execute function for interpreter mode
unsafe fn execute_less_than<
    F: PrimeField32,
    Ctx: ExecutionCtxTrait,
    const NUM_LIMBS: usize,
    const LIMB_BITS: usize,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, openvm_circuit::system::memory::online::GuestMemory, Ctx>,
) {
    use crate::adapters::{memory_read, memory_write};

    let pre_compute = unsafe { &*(pre_compute as *const LessThanPreCompute<NUM_LIMBS>) };

    // Read operands
    let rs1: [u8; NUM_LIMBS] = memory_read(
        &exec_state.memory,
        openvm_instructions::riscv::RV32_REGISTER_AS,
        pre_compute.rs1,
    );
    let rs2: [u8; NUM_LIMBS] = memory_read(
        &exec_state.memory,
        openvm_instructions::riscv::RV32_REGISTER_AS,
        pre_compute.rs2,
    );

    // Compute comparison
    let (cmp_result, _, _, _) =
        run_less_than::<NUM_LIMBS, LIMB_BITS>(pre_compute.is_slt, &rs1, &rs2);

    let mut output = [0u8; NUM_LIMBS];
    output[0] = cmp_result as u8;

    // Write result
    memory_write(
        &mut exec_state.memory,
        openvm_instructions::riscv::RV32_REGISTER_AS,
        pre_compute.rd,
        output,
    );

    // Increment PC
    let next_pc = exec_state.pc().wrapping_add(DEFAULT_PC_STEP);
    exec_state.set_pc(next_pc);
}

// InterpreterExecutor implementation
impl<F, const NUM_LIMBS: usize, const LIMB_BITS: usize> InterpreterExecutor<F>
    for LessThanCoreExecutor<NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
{
    fn pre_compute_size(&self) -> usize {
        std::mem::size_of::<LessThanPreCompute<NUM_LIMBS>>()
    }

    #[cfg(not(feature = "tco"))]
    fn pre_compute<Ctx>(
        &self,
        _pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let Instruction {
            opcode, a, b, c, ..
        } = *inst;

        let pre_compute = LessThanPreCompute::<NUM_LIMBS> {
            rd: a.as_canonical_u32(),
            rs1: b.as_canonical_u32(),
            rs2: c.as_canonical_u32(),
            is_slt: opcode.local_opcode_idx(self.offset) == (LessThanOpcode::SLT as usize),
        };

        data[..std::mem::size_of::<LessThanPreCompute<NUM_LIMBS>>()].copy_from_slice(unsafe {
            std::slice::from_raw_parts(
                &pre_compute as *const _ as *const u8,
                std::mem::size_of::<LessThanPreCompute<NUM_LIMBS>>(),
            )
        });

        Ok(execute_less_than::<F, Ctx, NUM_LIMBS, LIMB_BITS>)
    }

    #[cfg(feature = "tco")]
    fn handler<Ctx>(
        &self,
        _pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let Instruction {
            opcode, a, b, c, ..
        } = *inst;

        let pre_compute = LessThanPreCompute::<NUM_LIMBS> {
            rd: a.as_canonical_u32(),
            rs1: b.as_canonical_u32(),
            rs2: c.as_canonical_u32(),
            is_slt: opcode.local_opcode_idx(self.offset) == (LessThanOpcode::SLT as usize),
        };

        data[..std::mem::size_of::<LessThanPreCompute<NUM_LIMBS>>()].copy_from_slice(unsafe {
            std::slice::from_raw_parts(
                &pre_compute as *const _ as *const u8,
                std::mem::size_of::<LessThanPreCompute<NUM_LIMBS>>(),
            )
        });

        Ok(Box::new(execute_less_than::<F, Ctx, NUM_LIMBS, LIMB_BITS>))
    }
}

// Metered execution function
unsafe fn execute_less_than_metered<
    F: PrimeField32,
    Ctx: MeteredExecutionCtxTrait,
    const NUM_LIMBS: usize,
    const LIMB_BITS: usize,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, openvm_circuit::system::memory::online::GuestMemory, Ctx>,
) {
    use crate::adapters::{memory_read, memory_write};

    let pre_compute: &E2PreCompute<LessThanPreCompute<NUM_LIMBS>> = unsafe {
        std::slice::from_raw_parts(
            pre_compute,
            size_of::<E2PreCompute<LessThanPreCompute<NUM_LIMBS>>>(),
        )
        .borrow()
    };

    // Track chip height
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);

    let data = &pre_compute.data;

    // Read operands
    let rs1: [u8; NUM_LIMBS] = memory_read(
        &exec_state.memory,
        openvm_instructions::riscv::RV32_REGISTER_AS,
        data.rs1,
    );
    let rs2: [u8; NUM_LIMBS] = memory_read(
        &exec_state.memory,
        openvm_instructions::riscv::RV32_REGISTER_AS,
        data.rs2,
    );

    // Compute comparison
    let (cmp_result, _, _, _) = run_less_than::<NUM_LIMBS, LIMB_BITS>(data.is_slt, &rs1, &rs2);

    let mut output = [0u8; NUM_LIMBS];
    output[0] = cmp_result as u8;

    // Write result
    memory_write(
        &mut exec_state.memory,
        openvm_instructions::riscv::RV32_REGISTER_AS,
        data.rd,
        output,
    );

    // Increment PC
    let next_pc = exec_state.pc().wrapping_add(DEFAULT_PC_STEP);
    exec_state.set_pc(next_pc);
}

// InterpreterMeteredExecutor implementation
impl<F, const NUM_LIMBS: usize, const LIMB_BITS: usize> InterpreterMeteredExecutor<F>
    for LessThanCoreExecutor<NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<LessThanPreCompute<NUM_LIMBS>>>()
    }

    #[cfg(not(feature = "tco"))]
    fn metered_pre_compute<Ctx>(
        &self,
        chip_idx: usize,
        _pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let Instruction {
            opcode, a, b, c, ..
        } = *inst;

        let pre_compute: &mut E2PreCompute<LessThanPreCompute<NUM_LIMBS>> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        pre_compute.data = LessThanPreCompute {
            rd: a.as_canonical_u32(),
            rs1: b.as_canonical_u32(),
            rs2: c.as_canonical_u32(),
            is_slt: opcode.local_opcode_idx(self.offset) == (LessThanOpcode::SLT as usize),
        };

        Ok(execute_less_than_metered::<F, Ctx, NUM_LIMBS, LIMB_BITS>)
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx>(
        &self,
        chip_idx: usize,
        _pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let Instruction {
            opcode, a, b, c, ..
        } = *inst;

        let pre_compute: &mut E2PreCompute<LessThanPreCompute<NUM_LIMBS>> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        pre_compute.data = LessThanPreCompute {
            rd: a.as_canonical_u32(),
            rs1: b.as_canonical_u32(),
            rs2: c.as_canonical_u32(),
            is_slt: opcode.local_opcode_idx(self.offset) == (LessThanOpcode::SLT as usize),
        };

        Ok(Box::new(
            execute_less_than_metered::<F, Ctx, NUM_LIMBS, LIMB_BITS>,
        ))
    }
}

// Returns (cmp_result, diff_idx, x_sign, y_sign)
#[inline(always)]
pub(super) fn run_less_than<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    is_slt: bool,
    x: &[u8; NUM_LIMBS],
    y: &[u8; NUM_LIMBS],
) -> (bool, usize, bool, bool) {
    let x_sign = (x[NUM_LIMBS - 1] >> (LIMB_BITS - 1) == 1) && is_slt;
    let y_sign = (y[NUM_LIMBS - 1] >> (LIMB_BITS - 1) == 1) && is_slt;
    for i in (0..NUM_LIMBS).rev() {
        if x[i] != y[i] {
            return ((x[i] < y[i]) ^ x_sign ^ y_sign, i, x_sign, y_sign);
        }
    }
    (false, NUM_LIMBS, x_sign, y_sign)
}
