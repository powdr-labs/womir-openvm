use openvm_circuit::arch::VmChipWrapper;

use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};

mod core;
pub use core::*;

// Type aliases for CONST32 chip
pub type Const32Air = crate::adapters::Const32AdapterAir<RV32_REGISTER_NUM_LIMBS>;
pub type Const32Executor32 = crate::PreflightExecutorWrapperFp<
    Const32Executor<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
    RV32_REGISTER_NUM_LIMBS,
    RV32_CELL_BITS,
>;
pub type Const32Chip<F> = VmChipWrapper<F, Const32Filler<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>>;
