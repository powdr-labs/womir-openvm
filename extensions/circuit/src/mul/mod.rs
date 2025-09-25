use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS, WomBaseAluAdapterChip};
use crate::VmChipWrapperWom;

use openvm_rv32im_circuit::MultiplicationCoreChip;

pub type WomMultiplicationChip<F> = VmChipWrapperWom<
    F,
    WomBaseAluAdapterChip<F, 2, RV32_REGISTER_NUM_LIMBS, RV32_REGISTER_NUM_LIMBS>,
    MultiplicationCoreChip<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
