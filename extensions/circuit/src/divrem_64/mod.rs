use crate::VmChipWrapperWom;

use super::adapters::{WomBaseAluAdapterChip, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};

use crate::divrem::DivRemCoreChip;

pub type WomDivRem64Chip<F> = VmChipWrapperWom<
    F,
    WomBaseAluAdapterChip<F, 4, { RV32_REGISTER_NUM_LIMBS * 2 }, { RV32_REGISTER_NUM_LIMBS * 2 }>,
    DivRemCoreChip<{ RV32_REGISTER_NUM_LIMBS * 2 }, RV32_CELL_BITS>,
>;
