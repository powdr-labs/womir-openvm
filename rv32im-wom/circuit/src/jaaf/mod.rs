use crate::{adapters::Rv32JaafAdapterChipWom, VmChipWrapperWom};

mod core;
pub use core::*;

pub type Rv32JaafChipWom<F> = VmChipWrapperWom<F, Rv32JaafAdapterChipWom<F>, Rv32JaafCoreChipWom>;
