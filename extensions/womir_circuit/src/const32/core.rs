use openvm_circuit::arch::*;
use openvm_circuit::system::memory::{MemoryAuxColsFactory, online::TracingMemory};
use openvm_circuit_primitives::bitwise_op_lookup::NUM_BITWISE_OP_LOOKUP_COLS;
use openvm_circuit_primitives::{
    AlignedBytesBorrow,
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    utils::not,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::LocalOpcode;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
};
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra},
    rap::{BaseAirWithPublicValues, ColumnsAir},
};
use openvm_womir_transpiler::ConstOpcodes;
use struct_reflection::{StructReflection, StructReflectionHelper};
#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct Const32CoreCols<T, const NUM_LIMBS: usize> {
    pub is_valid: T, // Do we still need this?
    target_reg: T,
    imm_lo: [T; NUM_LIMBS], // How to divide the limbs for higher and lower bits, just divided by 2? here should be half number of limbs.
    imm_hi: [T; NUM_LIMBS],
}

#[derive(Copy, Clone, Debug, derive_new::new)]
pub struct Const32CoreAir<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub bus: BitwiseOperationLookupBus,
    offset: usize,
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for Const32CoreAir<NUM_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        Const32CoreCols::<F, NUM_LIMBS>::width()
    }
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> ColumnsAir<F>
    for Const32CoreAir<NUM_LIMBS, LIMB_BITS>
{
    fn columns(&self) -> Option<Vec<String>> {
        Const32CoreCols::<F, NUM_LIMBS>::struct_reflection()
    }
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAirWithPublicValues<F>
    for Const32CoreAir<NUM_LIMBS, LIMB_BITS>
{
}

impl<AB, I, const NUM_LIMBS: usize, const LIMB_BITS: usize> VmCoreAir<AB, I>
    for Const32CoreAir<NUM_LIMBS, LIMB_BITS>
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; 0]; 0]>,
    I::Writes: From<[[AB::Expr; RV32_REGISTER_NUM_LIMBS]; 1]>,
    I::ProcessedInstruction: From<MinimalInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        // let core_cols: &ConstsCoreCols<_> = local_core.borrow();

        // // Need at least one constraint otherwise stark-backend complains.
        // builder.assert_bool(core_cols.is_valid);

        // let opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
        //     self,
        //     AB::Expr::from_canonical_usize(ConstOpcodes::CONST32 as usize),
        // );

        // AdapterAirContext {
        //     to_pc: None,
        //     reads: [].into(),
        //     writes: [[AB::Expr::ZERO; RV32_REGISTER_NUM_LIMBS]].into(),
        //     instruction: MinimalInstruction {
        //         is_valid: core_cols.is_valid.into(),
        //         opcode,
        //     }
        //     .into(),
        // }
        unimplemented!()
    }

    fn start_offset(&self) -> usize {
        ConstOpcodes::CLASS_OFFSET
    }
}

// Minimal executor for CONST32 - no computation needed, just write immediate to register
#[derive(Clone, Copy, derive_new::new)]
pub struct Const32Executor<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub offset: usize,
}

// FpPreflightExecutor implementation for CONST32
impl<F, RA, const NUM_LIMBS: usize, const LIMB_BITS: usize> crate::FpPreflightExecutor<F, RA>
    for Const32Executor<NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    RA: RecordArena<'static, (), ()>,
{
    fn get_opcode_name(&self, _opcode: usize) -> String {
        "CONST32".to_string()
    }

    fn execute_with_fp(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
        fp: u32,
    ) -> Result<Option<u32>, ExecutionError> {
        let Instruction { a, b, c, .. } = instruction;

        // Extract immediates (16-bit values in b and c)
        let imm_lo = b.as_canonical_u32() & 0xFFFF;
        let imm_hi = c.as_canonical_u32() & 0xFFFF;

        // Combine to form 32-bit immediate
        let imm = (imm_hi << 16) | imm_lo;

        // Decompose into limbs
        let value = decompose_u32::<NUM_LIMBS, LIMB_BITS>(imm);

        // Write to register at (fp + target_reg)
        let target_addr = a.as_canonical_u32() + fp;
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

        // FP doesn't change for CONST32
        Ok(None)
    }
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
#[derive(Clone, Copy)]
struct Const32PreCompute {
    target_reg: u32,
    imm_lo: u16,
    imm_hi: u16,
}

// Execute function for CONST32
unsafe fn execute_const32<
    F: PrimeField32,
    Ctx: ExecutionCtxTrait,
    const NUM_LIMBS: usize,
    const LIMB_BITS: usize,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, openvm_circuit::system::memory::online::GuestMemory, Ctx>,
) {
    use crate::adapters::memory_write;

    // Convert raw bytes back to Const32PreCompute struct (unsafe cast)
    let pre_compute = unsafe { &*(pre_compute as *const Const32PreCompute) };

    // Combine immediates
    let imm = (pre_compute.imm_hi as u32) << 16 | (pre_compute.imm_lo as u32);

    // Decompose to limbs
    let value = decompose_u32::<NUM_LIMBS, LIMB_BITS>(imm);

    // Write to register (FP=0 in interpreter mode)
    memory_write(
        &mut exec_state.memory,
        RV32_REGISTER_AS,
        pre_compute.target_reg,
        value,
    );

    // Increment PC
    let next_pc = exec_state.pc().wrapping_add(DEFAULT_PC_STEP);
    exec_state.set_pc(next_pc);
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
        let Instruction { a, b, c, .. } = *inst;

        let pre_compute = Const32PreCompute {
            target_reg: a.as_canonical_u32(),
            imm_lo: (b.as_canonical_u32() & 0xFFFF) as u16,
            imm_hi: (c.as_canonical_u32() & 0xFFFF) as u16,
        };

        data[..std::mem::size_of::<Const32PreCompute>()].copy_from_slice(unsafe {
            std::slice::from_raw_parts(
                &pre_compute as *const _ as *const u8,
                std::mem::size_of::<Const32PreCompute>(),
            )
        });

        Ok(execute_const32::<F, Ctx, NUM_LIMBS, LIMB_BITS>)
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
        let Instruction { a, b, c, .. } = *inst;

        let pre_compute = Const32PreCompute {
            target_reg: a.as_canonical_u32(),
            imm_lo: (b.as_canonical_u32() & 0xFFFF) as u16,
            imm_hi: (c.as_canonical_u32() & 0xFFFF) as u16,
        };

        data[..std::mem::size_of::<Const32PreCompute>()].copy_from_slice(unsafe {
            std::slice::from_raw_parts(
                &pre_compute as *const _ as *const u8,
                std::mem::size_of::<Const32PreCompute>(),
            )
        });

        Ok(Box::new(execute_const32::<F, Ctx, NUM_LIMBS, LIMB_BITS>))
    }
}

// Minimal filler - no trace generation needed yet
#[derive(Clone, derive_new::new)]
pub struct Const32Filler<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub offset: usize,
}

impl<F, const NUM_LIMBS: usize, const LIMB_BITS: usize> TraceFiller<F>
    for Const32Filler<NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
{
    fn fill_trace_row(&self, _mem_helper: &MemoryAuxColsFactory<F>, _row_slice: &mut [F]) {
        // Minimal implementation - no constraints to fill yet
    }
}
