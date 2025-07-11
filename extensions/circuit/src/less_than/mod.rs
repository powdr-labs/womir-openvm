use super::adapters::{WomBaseAluAdapterChip, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use crate::VmChipWrapperWom;

mod core;
pub use core::*;

pub type LessThanChipWom<F> = VmChipWrapperWom<
    F,
    WomBaseAluAdapterChip<F>,
    LessThanCoreChip<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
