use super::adapters::{WomMultAdapterChip, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use crate::VmChipWrapperWom;

mod core;
pub use core::*;

pub type WomMultiplicationChip<F> = VmChipWrapperWom<
    F,
    WomMultAdapterChip<F>,
    MultiplicationCoreChip<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
