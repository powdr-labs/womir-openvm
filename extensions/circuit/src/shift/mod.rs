use crate::VmChipWrapperWom;

use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS, WomBaseAluAdapterChip};

mod core;
pub use core::*;

pub type WomShiftChip<F> = VmChipWrapperWom<
    F,
    WomBaseAluAdapterChip<F, 2, RV32_REGISTER_NUM_LIMBS, RV32_REGISTER_NUM_LIMBS>,
    ShiftCoreChip<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
