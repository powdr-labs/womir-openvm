use openvm_circuit::arch::ExecutionBridgeInteractor;
use openvm_circuit::arch::ExecutionBus;
use openvm_circuit::arch::PcIncOrSet;
use openvm_circuit::system::program::ProgramBus;
use openvm_circuit_primitives::AlignedBorrow;
use openvm_stark_backend::interaction::InteractionBuilder;
use openvm_stark_backend::interaction::PermutationCheckBus;
use serde::{Deserialize, Serialize};
use struct_reflection::StructReflection;
use struct_reflection::StructReflectionHelper;

use openvm_circuit::arch::{
    ExecutionBridge as OpenVmExecutionBridge, ExecutionState as OpenVmExecutionState,
};

#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Default, AlignedBorrow, Serialize, Deserialize, StructReflection,
)]
pub struct ExecutionState<T> {
    pub pc: T,
    pub fp: T,
    pub timestamp: T,
}

impl<T> From<ExecutionState<T>> for OpenVmExecutionState<T> {
    fn from(state: ExecutionState<T>) -> Self {
        OpenVmExecutionState {
            pc: state.pc,
            timestamp: state.timestamp,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FpBus {
    pub inner: PermutationCheckBus,
}

#[derive(Copy, Clone, Debug)]
pub struct ExecutionBridge {
    execution_bus: ExecutionBus,
    fp_bus: FpBus,
    program_bus: ProgramBus,
}

impl From<ExecutionBridge> for OpenVmExecutionBridge {
    fn from(bridge: ExecutionBridge) -> Self {
        OpenVmExecutionBridge::new(bridge.execution_bus, bridge.program_bus)
    }
}

impl ExecutionBridge {
    pub fn execute_and_increment_or_set_pc<AB: InteractionBuilder>(
        &self,
        opcode: impl Into<AB::Expr>,
        operands: impl IntoIterator<Item = impl Into<AB::Expr>>,
        from_state: ExecutionState<impl Into<AB::Expr> + Clone>,
        timestamp_change: impl Into<AB::Expr>,
        pc_kind: impl Into<PcIncOrSet<AB::Expr>>,
    ) -> ExecutionBridgeInteractor<AB> {
        // TODO: Handle FP as well
        OpenVmExecutionBridge::from(*self).execute_and_increment_or_set_pc(
            opcode,
            operands,
            from_state.into(),
            timestamp_change,
            pc_kind,
        )
    }
}
