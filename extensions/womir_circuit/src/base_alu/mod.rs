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
    VmAirWrapper<BaseAluAdapterAir<RV32_REGISTER_NUM_LIMBS>, BaseAluCoreAir<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>>;

// Use FP wrapper that wraps the base executor
pub type BaseAluExecutor = crate::PreflightExecutorWrapperFp<
    BaseAluCoreExecutor<
        BaseAluAdapterExecutor<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
        RV32_REGISTER_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
    RV32_REGISTER_NUM_LIMBS,
    RV32_CELL_BITS,
>;

pub type BaseAluChip<F> = VmChipWrapper<
    F,
    BaseAluFiller<
        BaseAluAdapterFiller<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
        RV32_REGISTER_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;

// 64-bit versions (8 limbs of 8 bits = 64 bits)
pub const REGISTER_NUM_LIMBS_64: usize = 8;

pub type BaseAlu64Air =
    VmAirWrapper<BaseAluAdapterAir<REGISTER_NUM_LIMBS_64>, BaseAluCoreAir<REGISTER_NUM_LIMBS_64, RV32_CELL_BITS>>;

pub type BaseAlu64Executor = crate::PreflightExecutorWrapperFp<
    BaseAluCoreExecutor<
        BaseAluAdapterExecutor<REGISTER_NUM_LIMBS_64, RV32_CELL_BITS>,
        REGISTER_NUM_LIMBS_64,
        RV32_CELL_BITS,
    >,
    REGISTER_NUM_LIMBS_64,
    RV32_CELL_BITS,
>;

pub type BaseAlu64Chip<F> = VmChipWrapper<
    F,
    BaseAluFiller<
        BaseAluAdapterFiller<REGISTER_NUM_LIMBS_64, RV32_CELL_BITS>,
        REGISTER_NUM_LIMBS_64,
        RV32_CELL_BITS,
    >,
>;
