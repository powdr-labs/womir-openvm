use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use crate::adapters::{BaseAluAdapterAir, BaseAluAdapterExecutor, BaseAluAdapterFiller};

mod core;
mod execution;
pub use core::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

// Re-use BaseAluAdapter since shift has the same I/O pattern (read 2, write 1)
pub type ShiftAir = VmAirWrapper<
    BaseAluAdapterAir<RV32_REGISTER_NUM_LIMBS>,
    ShiftCoreAir<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
pub type ShiftExecutor32 = crate::PreflightExecutorWrapperFp<
    ShiftExecutor<
        BaseAluAdapterExecutor<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
        RV32_REGISTER_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
    RV32_REGISTER_NUM_LIMBS,
    RV32_CELL_BITS,
>;
pub type ShiftChip<F> = VmChipWrapper<
    F,
    ShiftFiller<
        BaseAluAdapterFiller<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
        RV32_REGISTER_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;
