//! Execution bridge and state tracking with frame pointer (fp) support.
//! The analog of OpenVM's crates/vm/src/arch/execution.rs.
use openvm_circuit::arch::ExecutionBus;
use openvm_circuit::arch::PcIncOrSet;
use openvm_circuit::system::program::ProgramBus;
use openvm_circuit_primitives::AlignedBorrow;
use openvm_stark_backend::interaction::BusIndex;
use openvm_stark_backend::interaction::InteractionBuilder;
use openvm_stark_backend::interaction::PermutationCheckBus;
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

/// Similar to `openvm_circuit::arch::ExecutionBus`, but tracks `fp` instead of PC.
#[derive(Clone, Copy, Debug)]
pub struct FpBus {
    pub inner: PermutationCheckBus,
}

impl FpBus {
    pub const fn new(index: BusIndex) -> Self {
        Self {
            inner: PermutationCheckBus::new(index),
        }
    }

    #[inline(always)]
    pub fn index(&self) -> BusIndex {
        self.inner.index
    }
}

impl FpBus {
    /// Caller must constrain that `enabled` is boolean.
    pub fn execute<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        enabled: impl Into<AB::Expr>,
        prev_state: ExecutionState<impl Into<AB::Expr>>,
        next_state: ExecutionState<impl Into<AB::Expr>>,
    ) {
        // Like execution bus, but for FP and timestamp
        let enabled = enabled.into();
        self.inner.receive(
            builder,
            [prev_state.fp.into(), prev_state.timestamp.into()],
            enabled.clone(),
        );
        self.inner.send(
            builder,
            [next_state.fp.into(), next_state.timestamp.into()],
            enabled,
        );
    }
}

/// Like `openvm_circuit::arch::ExecutionBridge`, but with `fp` support.
#[derive(Copy, Clone, Debug)]
pub struct ExecutionBridge {
    execution_bus: ExecutionBus,
    fp_bus: FpBus,
    program_bus: ProgramBus,
}

pub struct ExecutionBridgeInteractor<AB: InteractionBuilder> {
    execution_bus: ExecutionBus,
    program_bus: ProgramBus,
    fp_bus: FpBus,
    opcode: AB::Expr,
    operands: Vec<AB::Expr>,
    from_state: ExecutionState<AB::Expr>,
    to_state: ExecutionState<AB::Expr>,
}

pub enum FpKeepOrSet<T> {
    /// Keep the frame pointer as is.
    Keep,
    /// Set the frame pointer to the given value.
    Set(T),
}

impl ExecutionBridge {
    pub fn new(execution_bus: ExecutionBus, fp_bus: FpBus, program_bus: ProgramBus) -> Self {
        Self {
            execution_bus,
            fp_bus,
            program_bus,
        }
    }

    pub fn execute_and_increment_or_set_pc<AB: InteractionBuilder>(
        &self,
        opcode: impl Into<AB::Expr>,
        operands: impl IntoIterator<Item = impl Into<AB::Expr>>,
        from_state: ExecutionState<impl Into<AB::Expr> + Clone>,
        timestamp_change: impl Into<AB::Expr>,
        pc_kind: impl Into<PcIncOrSet<AB::Expr>>,
        fp_kind: FpKeepOrSet<impl Into<AB::Expr>>,
    ) -> ExecutionBridgeInteractor<AB> {
        let to_state = ExecutionState {
            pc: match pc_kind.into() {
                PcIncOrSet::Set(to_pc) => to_pc,
                PcIncOrSet::Inc(pc_inc) => from_state.pc.clone().into() + pc_inc,
            },
            fp: match fp_kind {
                FpKeepOrSet::Keep => from_state.fp.clone().into(),
                FpKeepOrSet::Set(fp) => fp.into(),
            },
            timestamp: from_state.timestamp.clone().into() + timestamp_change.into(),
        };
        self.execute(opcode, operands, from_state, to_state)
    }

    pub fn execute<AB: InteractionBuilder>(
        &self,
        opcode: impl Into<AB::Expr>,
        operands: impl IntoIterator<Item = impl Into<AB::Expr>>,
        from_state: ExecutionState<impl Into<AB::Expr> + Clone>,
        to_state: ExecutionState<impl Into<AB::Expr>>,
    ) -> ExecutionBridgeInteractor<AB> {
        ExecutionBridgeInteractor {
            execution_bus: self.execution_bus,
            fp_bus: self.fp_bus,
            program_bus: self.program_bus,
            opcode: opcode.into(),
            operands: operands.into_iter().map(Into::into).collect(),
            from_state: from_state.map(Into::into),
            to_state: to_state.map(Into::into),
        }
    }
}

impl<AB: InteractionBuilder> ExecutionBridgeInteractor<AB> {
    /// Caller must constrain that `enabled` is boolean.
    pub fn eval(self, builder: &mut AB, enabled: impl Into<AB::Expr>) {
        let enabled = enabled.into();

        self.program_bus.lookup_instruction(
            builder,
            self.from_state.pc.clone(),
            self.opcode,
            self.operands,
            enabled.clone(),
        );

        self.fp_bus.execute(
            builder,
            enabled.clone(),
            self.from_state.clone(),
            self.to_state.clone(),
        );

        self.execution_bus.execute(
            builder,
            enabled,
            // Convert to OpenVmExecutionState -> discards fp
            self.from_state.into(),
            self.to_state.into(),
        );
    }
}
