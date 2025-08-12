use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use crate::{adapters::WomBaseAlu64AdapterChip, VmChipWrapperWom};

use crate::base_alu::BaseAluCoreChipWom;

pub type WomBaseAlu64Chip<F> = VmChipWrapperWom<
    F,
    WomBaseAlu64AdapterChip<F>,
    BaseAluCoreChipWom<{ RV32_REGISTER_NUM_LIMBS * 2 }, RV32_CELL_BITS>,
>;
