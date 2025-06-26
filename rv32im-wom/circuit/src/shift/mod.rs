use openvm_circuit::arch::VmChipWrapper;

use super::adapters::{Rv32WomBaseAluAdapterChip, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32ShiftChip<F> = VmChipWrapper<
    F,
    Rv32WomBaseAluAdapterChip<F>,
    ShiftCoreChip<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
