use crate::VmChipWrapperWom;

use super::adapters::{WomBaseAluAdapterChip, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};

mod core;
pub use core::*;

pub type WomShiftChip<F> = VmChipWrapperWom<
    F,
    WomBaseAluAdapterChip<F, 2, RV32_REGISTER_NUM_LIMBS, RV32_REGISTER_NUM_LIMBS>,
    ShiftCoreChip<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
