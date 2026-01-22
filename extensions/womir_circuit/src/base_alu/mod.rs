use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::adapters::{
    RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS, BaseAluAdapterAir, BaseAluAdapterExecutor,
    BaseAluAdapterFiller,
};

mod core;
pub use core::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

pub type BaseAluAir =
    VmAirWrapper<BaseAluAdapterAir, BaseAluCoreAir<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>>;

// Use FP wrapper that wraps the base executor
pub type BaseAluExecutor = crate::PreflightExecutorWrapperFp<
    BaseAluCoreExecutor<
        BaseAluAdapterExecutor<RV32_CELL_BITS>,
        RV32_REGISTER_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
    RV32_REGISTER_NUM_LIMBS,
    RV32_CELL_BITS,
>;

pub type BaseAluChip<F> = VmChipWrapper<
    F,
    BaseAluFiller<
        BaseAluAdapterFiller<RV32_CELL_BITS>,
        RV32_REGISTER_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;
