use super::adapters::{Const32AdapterAir, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
mod execution;
pub use execution::*;
use openvm_circuit::arch::VmChipWrapper;

pub type Const32Air = Const32AdapterAir<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>;
pub type Const32Executor32 = Const32Executor<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>;
pub type Const32Chip<F> = VmChipWrapper<F, Const32Filler<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>>;
