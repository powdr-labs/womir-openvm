use crate::memory_config::FpMemory;
use openvm_circuit::arch::*;
use openvm_circuit::system::memory::online::TracingMemory;
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
};
use openvm_stark_backend::p3_field::PrimeField32;
use std::borrow::{Borrow, BorrowMut};
// Minimal executor for CONST32 - no computation needed, just write immediate to register
#[derive(Clone, Copy, derive_new::new)]
pub struct Const32Executor<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub offset: usize,
}

// Helper to decompose u32 into limbs
fn decompose_u32<const NUM_LIMBS: usize, const LIMB_BITS: usize>(value: u32) -> [u8; NUM_LIMBS] {
    let mut result = [0u8; NUM_LIMBS];
    let limb_mask = ((1u32 << LIMB_BITS) - 1) as u8;

    for (i, limb) in result.iter_mut().enumerate().take(NUM_LIMBS.min(4)) {
        *limb = ((value >> (i * LIMB_BITS)) as u8) & limb_mask;
    }

    result
}

// PreCompute struct for CONST32
#[repr(C)]
#[derive(AlignedBytesBorrow, Clone, Copy)]
struct Const32PreCompute {
    target_reg: u32,
    imm_lo: u16,
    imm_hi: u16,
}

impl<const NUM_LIMBS: usize, const LIMB_BITS: usize> Const32Executor<NUM_LIMBS, LIMB_BITS> {
    #[inline(always)]
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        inst: &Instruction<F>,
        data: &mut Const32PreCompute,
    ) {
        let Instruction { a, b, c, .. } = *inst;
        *data = Const32PreCompute {
            target_reg: a.as_canonical_u32(),
            imm_lo: (b.as_canonical_u32() & 0xFFFF) as u16,
            imm_hi: (c.as_canonical_u32() & 0xFFFF) as u16,
        };
    }
}

impl<F, RA, const NUM_LIMBS: usize, const LIMB_BITS: usize> PreflightExecutor<F, RA>
    for Const32Executor<NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
{
    fn get_opcode_name(&self, _opcode: usize) -> String {
        "CONST32".to_string()
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let Instruction { a, b, c, .. } = instruction;

        // Extract immediates (16-bit values in b and c)
        let imm_lo = b.as_canonical_u32() & 0xFFFF;
        let imm_hi = c.as_canonical_u32() & 0xFFFF;

        // Combine to form 32-bit immediate
        let imm = (imm_hi << 16) | imm_lo;

        // Decompose into limbs
        let value = decompose_u32::<NUM_LIMBS, LIMB_BITS>(imm);

        // Write to register at (fp + target_reg)
        let target_addr = a.as_canonical_u32() + state.memory.data().fp();
        unsafe {
            state
                .memory
                .write::<u8, NUM_LIMBS, RV32_REGISTER_NUM_LIMBS>(
                    RV32_REGISTER_AS,
                    target_addr,
                    value,
                );
        }

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        Ok(())
    }
}

// InterpreterExecutor implementation
impl<F, const NUM_LIMBS: usize, const LIMB_BITS: usize> InterpreterExecutor<F>
    for Const32Executor<NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
{
    fn pre_compute_size(&self) -> usize {
        std::mem::size_of::<Const32PreCompute>()
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
        let data: &mut Const32PreCompute = data.borrow_mut();
        self.pre_compute_impl(inst, data);

        Ok(execute_e1_handler::<F, Ctx, NUM_LIMBS, LIMB_BITS>)
    }
}

impl<F, const LIMB_BITS: usize> InterpreterMeteredExecutor<F>
    for Const32Executor<RV32_REGISTER_NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<Const32PreCompute>>()
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
        let data: &mut E2PreCompute<Const32PreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(inst, &mut data.data);

        Ok(execute_e2_handler::<F, Ctx, RV32_REGISTER_NUM_LIMBS, LIMB_BITS>)
    }
}

// Execute function for CONST32
unsafe fn execute_e12_impl<
    F: PrimeField32,
    Ctx: ExecutionCtxTrait,
    const NUM_LIMBS: usize,
    const LIMB_BITS: usize,
>(
    pre_compute: &Const32PreCompute,
    exec_state: &mut VmExecState<F, openvm_circuit::system::memory::online::GuestMemory, Ctx>,
) {
    use crate::adapters::memory_write;

    // Combine immediates
    let imm = (pre_compute.imm_hi as u32) << 16 | (pre_compute.imm_lo as u32);

    // Decompose to limbs
    let value = decompose_u32::<NUM_LIMBS, LIMB_BITS>(imm);

    let fp = exec_state.memory.fp();

    // Write to register (FP=0 in interpreter mode)
    memory_write(
        &mut exec_state.memory,
        RV32_REGISTER_AS,
        pre_compute.target_reg + fp,
        value,
    );

    // Increment PC
    let next_pc = exec_state.pc().wrapping_add(DEFAULT_PC_STEP);
    exec_state.set_pc(next_pc);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const NUM_LIMBS: usize,
    const LIMB_BITS: usize,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, openvm_circuit::system::memory::online::GuestMemory, CTX>,
) {
    unsafe {
        let pre_compute: &Const32PreCompute =
            std::slice::from_raw_parts(pre_compute, size_of::<Const32PreCompute>()).borrow();
        execute_e12_impl::<F, CTX, NUM_LIMBS, LIMB_BITS>(pre_compute, exec_state);
    }
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    const NUM_LIMBS: usize,
    const LIMB_BITS: usize,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, openvm_circuit::system::memory::online::GuestMemory, CTX>,
) {
    unsafe {
        let pre_compute: &E2PreCompute<Const32PreCompute> =
            std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<Const32PreCompute>>())
                .borrow();
        exec_state
            .ctx
            .on_height_change(pre_compute.chip_idx as usize, 1);
        execute_e12_impl::<F, CTX, NUM_LIMBS, LIMB_BITS>(&pre_compute.data, exec_state);
    }
}
