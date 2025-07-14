use crate::VmChipWrapperWom;

use super::adapters::{WomMultAdapterChip, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};

mod core;
pub use core::*;

pub type WomDivRemChip<F> = VmChipWrapperWom<
    F,
    WomMultAdapterChip<F>,
    DivRemCoreChip<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
