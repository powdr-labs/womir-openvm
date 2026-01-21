use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use crate::{VmChipWrapperWom, adapters::WomBaseAluAdapterChip};

use openvm_rv32im_circuit::BaseAluCoreCols;

pub type WomBaseAluChip<F> = VmChipWrapperWom<
    F,
    WomBaseAluAdapterChip<F, RV32_REGISTER_NUM_LIMBS, RV32_REGISTER_NUM_LIMBS, 1>,
    BaseAluCoreCols<F, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
