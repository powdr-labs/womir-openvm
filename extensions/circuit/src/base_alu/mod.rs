use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use crate::{adapters::WomBaseAluAdapterChip, VmChipWrapperWom};

use openvm_rv32im_circuit::BaseAluCoreChip;

pub type WomBaseAluChip<F> = VmChipWrapperWom<
    F,
    WomBaseAluAdapterChip<F, 2, RV32_REGISTER_NUM_LIMBS, RV32_REGISTER_NUM_LIMBS>,
    BaseAluCoreChip<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
