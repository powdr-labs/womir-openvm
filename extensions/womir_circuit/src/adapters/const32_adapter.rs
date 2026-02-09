use openvm_circuit::arch::*;
use openvm_circuit::system::memory::MemoryAddress;
use openvm_circuit::system::memory::offline_checker::MemoryWriteAuxCols;
use openvm_circuit::system::memory::{MemoryAuxColsFactory, offline_checker::MemoryBridge};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::riscv::RV32_REGISTER_AS;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::ColumnsAir,
};
use std::borrow::Borrow;
use struct_reflection::{StructReflection, StructReflectionHelper};

// Cols for CONST32 adapter - minimal structure since we just write immediates
#[repr(C)]
#[derive(AlignedBorrow, Clone, Copy, Debug, StructReflection)]
pub struct Const32AdapterCols<T, const NUM_LIMBS: usize> {
    pub from_state: ExecutionState<T>,
    pub target_reg: T,
    pub writes_aux: MemoryWriteAuxCols<T, NUM_LIMBS>,
}

// AIR for CONST32 adapter
#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Const32AdapterAir<const NUM_LIMBS: usize> {
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
}

impl<F: Field, const NUM_LIMBS: usize> BaseAir<F> for Const32AdapterAir<NUM_LIMBS> {
    fn width(&self) -> usize {
        Const32AdapterCols::<F, NUM_LIMBS>::width()
    }
}

impl<F: Field, const NUM_LIMBS: usize> ColumnsAir<F> for Const32AdapterAir<NUM_LIMBS> {
    fn columns(&self) -> Option<Vec<String>> {
        Const32AdapterCols::<F, NUM_LIMBS>::struct_reflection()
    }
}

impl<AB, const NUM_LIMBS: usize> VmAdapterAir<AB> for Const32AdapterAir<NUM_LIMBS>
where
    AB: InteractionBuilder,
{
    type Interface =
        BasicAdapterInterface<AB::Expr, MinimalInstruction<AB::Expr>, 0, 1, 0, NUM_LIMBS>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local: &Const32AdapterCols<_, NUM_LIMBS> = local.borrow();

        self.memory_bridge
            .write(
                MemoryAddress::new(
                    AB::F::from_canonical_u32(RV32_REGISTER_AS),
                    local.target_reg.clone(),
                ),
                ctx.writes[0].clone(),
                local.from_state.timestamp + AB::F::ONE,
                &local.writes_aux,
            )
            .eval(builder, ctx.instruction.is_valid.clone());
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Const32AdapterCols<_, NUM_LIMBS> = local.borrow();
        cols.target_reg
    }
}

// Executor for CONST32 adapter using FP
pub struct Const32AdapterExecutor<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    _phantom: std::marker::PhantomData<([u8; NUM_LIMBS], [u8; LIMB_BITS])>,
}

impl<const NUM_LIMBS: usize, const LIMB_BITS: usize> Const32AdapterExecutor<NUM_LIMBS, LIMB_BITS> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<const NUM_LIMBS: usize, const LIMB_BITS: usize> Default
    for Const32AdapterExecutor<NUM_LIMBS, LIMB_BITS>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const NUM_LIMBS: usize, const LIMB_BITS: usize> Clone
    for Const32AdapterExecutor<NUM_LIMBS, LIMB_BITS>
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<const NUM_LIMBS: usize, const LIMB_BITS: usize> Copy
    for Const32AdapterExecutor<NUM_LIMBS, LIMB_BITS>
{
}

// Record for adapter (empty for now)
#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct Const32AdapterRecord {}

// Filler for CONST32 adapter
#[derive(Clone, derive_new::new)]
pub struct Const32AdapterFiller<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    _phantom: std::marker::PhantomData<([u8; NUM_LIMBS], [u8; LIMB_BITS])>,
}

impl<F, const NUM_LIMBS: usize, const LIMB_BITS: usize> TraceFiller<F>
    for Const32AdapterFiller<NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
{
    fn fill_trace_row(&self, _mem_helper: &MemoryAuxColsFactory<F>, _row_slice: &mut [F]) {
        // Minimal implementation - no constraints to fill yet
    }
}
