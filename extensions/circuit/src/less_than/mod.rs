use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS, WomBaseAluAdapterChip};
use crate::VmChipWrapperWom;

mod core;
pub use core::*;

pub type LessThanChipWom<F> = VmChipWrapperWom<
    F,
    WomBaseAluAdapterChip<F, RV32_REGISTER_NUM_LIMBS, RV32_REGISTER_NUM_LIMBS, 1>,
    LessThanCoreChip<RV32_REGISTER_NUM_LIMBS, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
