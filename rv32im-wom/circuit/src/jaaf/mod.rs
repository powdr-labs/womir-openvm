use openvm_circuit::arch::VmChipWrapper;

use crate::adapters::Rv32JaafAdapterChip;

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32JaafChip<F> = VmChipWrapper<F, Rv32JaafAdapterChip<F>, Rv32JaafCoreChip>;
