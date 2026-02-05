//! Stub AIRs for executors that don't have real constraint implementations yet.
//! These allow the proving path to compile and run without actual constraints.

use openvm_circuit::arch::{
    AdapterAirContext, BasicAdapterInterface, ExecutionBridge, MinimalInstruction, TraceFiller,
    VmChipWrapper, VmCoreAir,
};
use openvm_circuit::system::memory::MemoryAuxColsFactory;
use openvm_circuit::system::memory::offline_checker::MemoryBridge;
use openvm_circuit_primitives::bitwise_op_lookup::BitwiseOperationLookupBus;
use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::Field,
    rap::{BaseAirWithPublicValues, ColumnsAir},
};

use crate::adapters::{
    BaseAluAdapterAir, BaseAluAdapterFiller, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS,
};

/// A stub core AIR that has zero width and no constraints.
/// Used for executors that don't have real proving support yet.
#[derive(Clone, Copy)]
pub struct StubCoreAir {
    pub offset: usize,
}

impl StubCoreAir {
    pub fn new(offset: usize) -> Self {
        Self { offset }
    }
}

impl<F: Field> BaseAir<F> for StubCoreAir {
    fn width(&self) -> usize {
        0 // No core columns
    }
}

impl<F: Field> ColumnsAir<F> for StubCoreAir {
    fn columns(&self) -> Option<Vec<String>> {
        Some(vec![])
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for StubCoreAir {}

// Interface type for BaseAluAdapterAir<4>
type StubInterface<T> = BasicAdapterInterface<
    T,
    MinimalInstruction<T>,
    2,
    1,
    RV32_REGISTER_NUM_LIMBS,
    RV32_REGISTER_NUM_LIMBS,
>;

// Implement VmCoreAir specifically for the BaseAluAdapterInterface
impl<AB> VmCoreAir<AB, StubInterface<AB::Expr>> for StubCoreAir
where
    AB: openvm_stark_backend::interaction::InteractionBuilder,
{
    fn eval(
        &self,
        _builder: &mut AB,
        _local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, StubInterface<AB::Expr>> {
        use openvm_stark_backend::p3_field::FieldAlgebra;

        // Create zero expressions for all interface fields using from_fn
        AdapterAirContext {
            to_pc: None,
            reads: std::array::from_fn(|_| std::array::from_fn(|_| AB::Expr::ZERO)),
            writes: std::array::from_fn(|_| std::array::from_fn(|_| AB::Expr::ZERO)),
            instruction: MinimalInstruction {
                is_valid: AB::Expr::ZERO,
                opcode: AB::Expr::ZERO,
            },
        }
    }

    fn start_offset(&self) -> usize {
        self.offset
    }
}

/// A stub filler that does nothing.
#[derive(Clone)]
pub struct StubFiller<AF> {
    #[allow(dead_code)]
    pub adapter_filler: AF,
}

impl<AF> StubFiller<AF> {
    pub fn new(adapter_filler: AF) -> Self {
        Self { adapter_filler }
    }
}

impl<F, AF> TraceFiller<F> for StubFiller<AF>
where
    F: Send + Sync,
    AF: Send + Sync,
{
    fn fill_trace_row(&self, _mem_helper: &MemoryAuxColsFactory<F>, _row_slice: &mut [F]) {
        // No-op - stub has no columns to fill
    }
}

// Type aliases for stub AIRs
pub type StubAir =
    openvm_circuit::arch::VmAirWrapper<BaseAluAdapterAir<RV32_REGISTER_NUM_LIMBS>, StubCoreAir>;
pub type StubChip<F> =
    VmChipWrapper<F, StubFiller<BaseAluAdapterFiller<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>>>;

/// Creates a stub AIR with the given offset
pub fn make_stub_air(
    exec_bridge: ExecutionBridge,
    memory_bridge: MemoryBridge,
    bitwise_lu: BitwiseOperationLookupBus,
    offset: usize,
) -> StubAir {
    openvm_circuit::arch::VmAirWrapper::new(
        BaseAluAdapterAir::new(exec_bridge, memory_bridge, bitwise_lu),
        StubCoreAir::new(offset),
    )
}
