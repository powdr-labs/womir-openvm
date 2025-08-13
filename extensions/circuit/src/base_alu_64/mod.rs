use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use crate::{adapters::WomBaseAluAdapterChip, VmChipWrapperWom};

use crate::base_alu::BaseAluCoreChipWom;

pub type WomBaseAlu64Chip<F> = VmChipWrapperWom<
    F,
    WomBaseAluAdapterChip<F, 4, 8, 8>,
    BaseAluCoreChipWom<{ RV32_REGISTER_NUM_LIMBS * 2 }, RV32_CELL_BITS>,
>;
