use crate::{adapters::JaafAdapterChipWom, VmChipWrapperWom};

mod core;
pub use core::*;

pub type JaafChipWom<F> = VmChipWrapperWom<F, JaafAdapterChipWom<F>, JaafCoreChipWom>;
