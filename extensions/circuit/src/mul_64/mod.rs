use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS, WomBaseAluAdapterChip};
use crate::VmChipWrapperWom;

use openvm_rv32im_circuit::MultiplicationCoreCols;

pub type WomMultiplication64Chip<F> = VmChipWrapperWom<
    F,
    WomBaseAluAdapterChip<F, { RV32_REGISTER_NUM_LIMBS * 2 }, { RV32_REGISTER_NUM_LIMBS * 2 }, 2>,
    MultiplicationCoreCols<{ RV32_REGISTER_NUM_LIMBS * 2 }, RV32_CELL_BITS>,
>;
