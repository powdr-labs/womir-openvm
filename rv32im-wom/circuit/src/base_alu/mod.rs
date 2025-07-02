
use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use crate::{adapters::Rv32WomBaseAluAdapterChip, VmChipWrapperWom};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32WomBaseAluChip<F> = VmChipWrapperWom<
    F,
    Rv32WomBaseAluAdapterChip<F>,
    BaseAluCoreChipWom<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
