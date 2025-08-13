use crate::VmChipWrapperWom;

use super::adapters::{WomBaseAluAdapterChip, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};

mod core;
pub use core::*;

pub type Rv32ShiftChip<F> = VmChipWrapperWom<
    F,
    WomBaseAluAdapterChip<F, 2, 4, 4>,
    ShiftCoreChip<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
