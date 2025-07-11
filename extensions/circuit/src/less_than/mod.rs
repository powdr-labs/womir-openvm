use openvm_circuit::arch::VmChipWrapper;

use super::adapters::{WomBaseAluAdapterChip, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};

mod core;
pub use core::*;

pub type Rv32LessThanChip<F> = VmChipWrapper<
    F,
    WomBaseAluAdapterChip<F>,
    LessThanCoreChip<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
