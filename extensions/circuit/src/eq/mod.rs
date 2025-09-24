use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS, WomBaseAluAdapterChip};
use crate::VmChipWrapperWom;

mod core;
pub use core::*;

pub type EqChipWom<F> = VmChipWrapperWom<
    F,
    WomBaseAluAdapterChip<F, 2, RV32_REGISTER_NUM_LIMBS, RV32_REGISTER_NUM_LIMBS>,
    EqCoreChip<RV32_REGISTER_NUM_LIMBS, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
