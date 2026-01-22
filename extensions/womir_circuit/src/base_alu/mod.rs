use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::adapters::{
    RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS, Rv32BaseAluAdapterAir, Rv32BaseAluAdapterExecutor,
    Rv32BaseAluAdapterFiller,
};

mod core;
pub use core::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

pub type Rv32BaseAluAir =
    VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>>;

// Use FP wrapper that wraps the base executor
pub type Rv32BaseAluExecutor = crate::PreflightExecutorWrapperFp<
    BaseAluExecutor<
        Rv32BaseAluAdapterExecutor<RV32_CELL_BITS>,
        RV32_REGISTER_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
    RV32_REGISTER_NUM_LIMBS,
    RV32_CELL_BITS,
>;

pub type Rv32BaseAluChip<F> = VmChipWrapper<
    F,
    BaseAluFiller<
        Rv32BaseAluAdapterFiller<RV32_CELL_BITS>,
        RV32_REGISTER_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;
