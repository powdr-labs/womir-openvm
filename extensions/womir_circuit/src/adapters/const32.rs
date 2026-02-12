use crate::execution::{ExecutionBridge, ExecutionState};
use openvm_circuit::arch::{BasicAdapterInterface, MinimalInstruction, VmAdapterAir};
use openvm_circuit::system::memory::offline_checker::MemoryBridge;
use openvm_circuit_primitives::AlignedBorrow;
use openvm_circuit_primitives::bitwise_op_lookup::BitwiseOperationLookupBus;
use openvm_stark_backend::interaction::InteractionBuilder;
use openvm_stark_backend::p3_air::AirBuilder;
use openvm_stark_backend::p3_air::{Air, BaseAir};
use openvm_stark_backend::rap::{BaseAirWithPublicValues, ColumnsAir, PartitionedBaseAir};
use std::borrow::Borrow;
use struct_reflection::{StructReflection, StructReflectionHelper};

#[derive(AlignedBorrow, StructReflection)]
pub struct Const32AdapterAirCol<T, const NUM_LIMBS: usize, const CELL_BITS: usize> {
    pub from_state: ExecutionState<T>,
    pub rd_ptr: T,
    pub imm: T,
}

#[derive(derive_new::new)]
pub struct Const32AdapterAir<const NUM_LIMBS: usize, const CELL_BITS: usize> {
    pub bus: BitwiseOperationLookupBus,
    pub offset: usize,
    pub(super) _execution_bridge: ExecutionBridge,
    pub(super) _memory_bridge: MemoryBridge,
}

impl<F, const NUM_LIMBS: usize, const CELL_BITS: usize> BaseAir<F>
    for Const32AdapterAir<NUM_LIMBS, CELL_BITS>
{
    fn width(&self) -> usize {
        Const32AdapterAirCol::<F, NUM_LIMBS, CELL_BITS>::width()
    }
}

impl<F, const NUM_LIMBS: usize, const CELL_BITS: usize> BaseAirWithPublicValues<F>
    for Const32AdapterAir<NUM_LIMBS, CELL_BITS>
{
    fn num_public_values(&self) -> usize {
        unimplemented!()
    }
}

impl<F, const NUM_LIMBS: usize, const CELL_BITS: usize> PartitionedBaseAir<F>
    for Const32AdapterAir<NUM_LIMBS, CELL_BITS>
{
}

impl<F, const NUM_LIMBS: usize, const CELL_BITS: usize> ColumnsAir<F>
    for Const32AdapterAir<NUM_LIMBS, CELL_BITS>
{
    fn columns(&self) -> Option<Vec<String>> {
        Const32AdapterAirCol::<F, NUM_LIMBS, CELL_BITS>::struct_reflection()
    }
}

impl<AB: InteractionBuilder, const NUM_LIMBS: usize, const CELL_BITS: usize> VmAdapterAir<AB>
    for Const32AdapterAir<NUM_LIMBS, CELL_BITS>
{
    type Interface =
        BasicAdapterInterface<AB::Expr, MinimalInstruction<AB::Expr>, 2, 1, NUM_LIMBS, CELL_BITS>;
    fn eval(
        &self,
        _builder: &mut AB,
        _local: &[<AB as AirBuilder>::Var],
        _interface: openvm_circuit::arch::AdapterAirContext<
            <AB as AirBuilder>::Expr,
            Self::Interface,
        >,
    ) {
        unimplemented!()
    }
    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Const32AdapterAirCol<_, NUM_LIMBS, CELL_BITS> = local.borrow();
        cols.from_state.pc
    }
}

impl<AB, const NUM_LIMBS: usize, const CELL_BITS: usize> Air<AB>
    for Const32AdapterAir<NUM_LIMBS, CELL_BITS>
where
    AB: AirBuilder,
{
    fn eval(&self, _builder: &mut AB) {
        unimplemented!()
    }
}
