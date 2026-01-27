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

// Re-use BaseAluAdapter since multiplication has the same I/O pattern (read 2, write 1)
pub type MultiplicationAir = VmAirWrapper<
    BaseAluAdapterAir<RV32_REGISTER_NUM_LIMBS>,
    MultiplicationCoreAir<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
pub type MultiplicationExecutor32 = crate::PreflightExecutorWrapperFp<
    MultiplicationExecutor<
        BaseAluAdapterExecutor<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
        RV32_REGISTER_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
    RV32_REGISTER_NUM_LIMBS,
    RV32_CELL_BITS,
>;
pub type MultiplicationChip<F> = VmChipWrapper<
    F,
    MultiplicationFiller<
        BaseAluAdapterFiller<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
        RV32_REGISTER_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;
