use crate::VmChipWrapperWom;

use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS, WomBaseAluAdapterChip};

use openvm_rv32im_circuit::ShiftCoreCols;

pub type WomShiftChip<F> = VmChipWrapperWom<
    F,
    WomBaseAluAdapterChip<F, RV32_REGISTER_NUM_LIMBS, RV32_REGISTER_NUM_LIMBS, 1>,
    ShiftCoreCols<F, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
