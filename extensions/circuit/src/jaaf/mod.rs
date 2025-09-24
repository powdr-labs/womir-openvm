use crate::{VmChipWrapperWom, adapters::JaafAdapterChipWom};

mod core;
pub use core::*;

pub type JaafChipWom<F> = VmChipWrapperWom<F, JaafAdapterChipWom<F>, JaafCoreChipWom>;
