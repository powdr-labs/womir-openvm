use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS, WomBaseAluAdapterChip};
use crate::VmChipWrapperWom;

use openvm_rv32im_circuit::MultiplicationCoreCols;

pub type WomMultiplicationChip<F> = VmChipWrapperWom<
    F,
    WomBaseAluAdapterChip<F, RV32_REGISTER_NUM_LIMBS, RV32_REGISTER_NUM_LIMBS, 1>,
    MultiplicationCoreCols<F, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
