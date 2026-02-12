use std::borrow::Borrow;

use super::RV32_REGISTER_NUM_LIMBS;
use crate::execution::{ExecutionBridge, ExecutionState};
use openvm_circuit::system::memory::offline_checker::{MemoryBridge, MemoryWriteAuxCols};
use openvm_circuit_primitives::{AlignedBorrow, bitwise_op_lookup::BitwiseOperationLookupBus};
use openvm_stark_backend::{
    p3_air::{Air, AirBuilder, BaseAir},
    p3_matrix::Matrix,
    rap::{BaseAirWithPublicValues, ColumnsAir, PartitionedBaseAir},
};

use struct_reflection::{StructReflection, StructReflectionHelper};

#[derive(AlignedBorrow, StructReflection)]
pub struct Const32AdapterAirCol<T, const NUM_LIMBS: usize, const CELL_BITS: usize> {
    pub is_valid: T,
    pub from_state: ExecutionState<T>,
    pub rd_ptr: T,
    pub imm_limbs: [T; NUM_LIMBS],
    pub write_aux: MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LIMBS>,
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
        0
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

impl<AB, const NUM_LIMBS: usize, const CELL_BITS: usize> Air<AB>
    for Const32AdapterAir<NUM_LIMBS, CELL_BITS>
where
    AB: AirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let cols: &Const32AdapterAirCol<AB::Var, NUM_LIMBS, CELL_BITS> = (*local).borrow();
        builder.assert_bool(cols.is_valid);
    }
}
