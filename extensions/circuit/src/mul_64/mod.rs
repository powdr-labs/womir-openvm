use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS, WomBaseAluAdapterChip};
use crate::VmChipWrapperWom;

use openvm_rv32im_circuit::MultiplicationCoreChip;

pub type WomMultiplication64Chip<F> = VmChipWrapperWom<
    F,
    WomBaseAluAdapterChip<F, { RV32_REGISTER_NUM_LIMBS * 2 }, { RV32_REGISTER_NUM_LIMBS * 2 }>,
    MultiplicationCoreChip<{ RV32_REGISTER_NUM_LIMBS * 2 }, RV32_CELL_BITS>,
>;
