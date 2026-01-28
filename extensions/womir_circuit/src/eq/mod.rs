use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::adapters::{
    BaseAluAdapterAir, BaseAluAdapterFiller, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS,
};

mod core;
pub use core::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

// #[cfg(test)]
// mod tests;

// Type aliases for 32-bit Eq chip
pub type EqAir = VmAirWrapper<
    BaseAluAdapterAir<RV32_REGISTER_NUM_LIMBS>,
    EqCoreAir<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
pub type EqExecutor32 = crate::PreflightExecutorWrapperFp<
    EqCoreExecutor<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
    RV32_REGISTER_NUM_LIMBS,
    RV32_CELL_BITS,
>;
pub type EqChip<F> = VmChipWrapper<
    F,
    EqFiller<
        BaseAluAdapterFiller<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
        RV32_REGISTER_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;

// Type aliases for 64-bit Eq chip (8 limbs)
const RV64_REGISTER_NUM_LIMBS: usize = 8;

pub type Eq64Air = VmAirWrapper<
    BaseAluAdapterAir<RV64_REGISTER_NUM_LIMBS>,
    EqCoreAir<RV64_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
pub type EqExecutor64 = crate::PreflightExecutorWrapperFp<
    EqCoreExecutor<RV64_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
    RV64_REGISTER_NUM_LIMBS,
    RV32_CELL_BITS,
>;
pub type Eq64Chip<F> = VmChipWrapper<
    F,
    EqFiller<
        BaseAluAdapterFiller<RV64_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
        RV64_REGISTER_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;
