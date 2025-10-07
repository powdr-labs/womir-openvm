use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS, WomBaseAluAdapterChip};
use crate::VmChipWrapperWom;

use crate::eq::EqCoreChip;

pub type Eq64ChipWom<F> = VmChipWrapperWom<
    F,
    WomBaseAluAdapterChip<F, { RV32_REGISTER_NUM_LIMBS * 2 }, RV32_REGISTER_NUM_LIMBS, 2>,
    EqCoreChip<{ RV32_REGISTER_NUM_LIMBS * 2 }, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
