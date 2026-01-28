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

// Type aliases for 32-bit LessThan chip
pub type LessThanAir = VmAirWrapper<
    BaseAluAdapterAir<RV32_REGISTER_NUM_LIMBS>,
    LessThanCoreAir<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
pub type LessThanExecutor32 = crate::PreflightExecutorWrapperFp<
    LessThanCoreExecutor<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
    RV32_REGISTER_NUM_LIMBS,
    RV32_CELL_BITS,
>;
pub type LessThanChip<F> = VmChipWrapper<
    F,
    LessThanFiller<
        BaseAluAdapterFiller<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
        RV32_REGISTER_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;

// Type aliases for 64-bit LessThan chip (8 limbs)
const RV64_REGISTER_NUM_LIMBS: usize = 8;

pub type LessThan64Air = VmAirWrapper<
    BaseAluAdapterAir<RV64_REGISTER_NUM_LIMBS>,
    LessThanCoreAir<RV64_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
pub type LessThan64Executor = crate::PreflightExecutorWrapperFp<
    LessThanCoreExecutor<RV64_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
    RV64_REGISTER_NUM_LIMBS,
    RV32_CELL_BITS,
>;
pub type LessThan64Chip<F> = VmChipWrapper<
    F,
    LessThanFiller<
        BaseAluAdapterFiller<RV64_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
        RV64_REGISTER_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;
