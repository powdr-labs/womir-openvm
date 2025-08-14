use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use crate::{adapters::WomBaseAluAdapterChip, VmChipWrapperWom};

mod core;
pub use core::*;

pub type WomBaseAluChip<F> = VmChipWrapperWom<
    F,
    WomBaseAluAdapterChip<F, 2, RV32_REGISTER_NUM_LIMBS, RV32_REGISTER_NUM_LIMBS>,
    BaseAluCoreChipWom<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
