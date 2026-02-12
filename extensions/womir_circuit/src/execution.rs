//! Execution state with frame pointer (fp) support.
use openvm_circuit_primitives::AlignedBorrow;
use serde::{Deserialize, Serialize};
use struct_reflection::StructReflection;
use struct_reflection::StructReflectionHelper;

use openvm_circuit::arch::ExecutionState as OpenVmExecutionState;

/// Like `openvm_circuit::arch::ExecutionState`, but with `fp` added.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Default, AlignedBorrow, Serialize, Deserialize, StructReflection,
)]
pub struct ExecutionState<T> {
    pub pc: T,
    pub fp: T,
    pub timestamp: T,
}

/// Discards `fp` when converting to `OpenVmExecutionState`.
impl<T> From<ExecutionState<T>> for OpenVmExecutionState<T> {
    fn from(state: ExecutionState<T>) -> Self {
        OpenVmExecutionState {
            pc: state.pc,
            timestamp: state.timestamp,
        }
    }
}

impl<T> ExecutionState<T> {
    pub fn new(pc: impl Into<T>, fp: impl Into<T>, timestamp: impl Into<T>) -> Self {
        Self {
            pc: pc.into(),
            fp: fp.into(),
            timestamp: timestamp.into(),
        }
    }

    pub fn map<U: Clone, F: Fn(T) -> U>(self, function: F) -> ExecutionState<U> {
        ExecutionState {
            pc: function(self.pc),
            fp: function(self.fp),
            timestamp: function(self.timestamp),
        }
    }
}
