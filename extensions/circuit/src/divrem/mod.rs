use crate::VmChipWrapperWom;

use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS, WomBaseAluAdapterChip};

use openvm_rv32im_circuit::DivRemCoreChip;

pub type WomDivRemChip<F> = VmChipWrapperWom<
    F,
    WomBaseAluAdapterChip<F, RV32_REGISTER_NUM_LIMBS, RV32_REGISTER_NUM_LIMBS, 1>,
    DivRemCoreChip<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
