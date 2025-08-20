use super::adapters::{WomBaseAluAdapterChip, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use crate::VmChipWrapperWom;

use crate::eq::EqCoreChip;

pub type Eq64ChipWom<F> = VmChipWrapperWom<
    F,
    WomBaseAluAdapterChip<F, 4, { RV32_REGISTER_NUM_LIMBS * 2 }, RV32_REGISTER_NUM_LIMBS>,
    EqCoreChip<{ RV32_REGISTER_NUM_LIMBS * 2 }, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
