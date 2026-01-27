use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use crate::adapters::{MultAdapterAir, MultAdapterExecutor, MultAdapterFiller};

mod core;
mod execution;
pub use core::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

pub type MultiplicationAir =
    VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>>;
pub type MultiplicationExecutor32 = crate::PreflightExecutorWrapperFp<
    MultiplicationExecutor<MultAdapterExecutor, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
    RV32_REGISTER_NUM_LIMBS,
    RV32_CELL_BITS,
>;
pub type MultiplicationChip<F> = VmChipWrapper<
    F,
    MultiplicationFiller<MultAdapterFiller, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
