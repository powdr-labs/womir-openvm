use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::adapters::JumpAdapterAir;

pub mod core;
mod execution;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::JumpChipGpu;

pub use execution::{JumpExecutor, JumpFiller};

pub type JumpAir = VmAirWrapper<JumpAdapterAir, core::JumpCoreAir>;
pub type JumpChip<F> = VmChipWrapper<F, JumpFiller>;
