use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
mod core;
mod execution;
pub use execution::*;

pub type Const32Executor32 = Const32Executor<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>;
