mod core;

pub use core::*;

use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::adapters::RV32_REGISTER_NUM_LIMBS;
use crate::adapters::{LoadStoreAdapterAir, LoadStoreAdapterExecutor};

mod execution;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;
#[cfg(feature = "aot")]
mod aot;

pub type LoadStoreAir =
    VmAirWrapper<LoadStoreAdapterAir, LoadStoreCoreAir<RV32_REGISTER_NUM_LIMBS>>;
pub type LoadStoreExecutor32 = crate::PreflightExecutorWrapperFp<
    LoadStoreExecutor<LoadStoreAdapterExecutor, RV32_REGISTER_NUM_LIMBS>,
    RV32_REGISTER_NUM_LIMBS,
    { super::adapters::RV32_CELL_BITS },
>;
pub type LoadStoreChip<F> = VmChipWrapper<F, LoadStoreFiller>;
