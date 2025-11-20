mod core;

use crate::adapters::ConstsAdapterChipWom;
pub use core::ConstsCoreChipWom;

pub type ConstsChipWom<F> =
    crate::wom_traits::VmChipWrapperWom<F, ConstsAdapterChipWom<F>, ConstsCoreChipWom>;
