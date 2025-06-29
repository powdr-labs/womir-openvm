use crate::{adapters::Rv32JaafAdapterChipWom, VmChipWrapperWom};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32JaafChipWom<F> = VmChipWrapperWom<F, Rv32JaafAdapterChipWom<F>, Rv32JaafCoreChipWom>;
