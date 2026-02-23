use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::call::{CallAdapterAir, CallAdapterExecutor},
    call::tracegen::CallFiller,
};

mod core;
pub mod execution;
pub mod tracegen;

pub use self::core::{CallCoreAir, CallCoreCols, CallCoreRecord};
pub use execution::CallExecutor;

pub type Rv32CallExecutor = CallExecutor<CallAdapterExecutor>;
pub type CallAir = VmAirWrapper<CallAdapterAir, CallCoreAir>;
pub type CallChip<F> = VmChipWrapper<F, CallFiller>;
