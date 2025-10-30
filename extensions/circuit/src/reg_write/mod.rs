mod core;

use crate::adapters::RegWriteAdapterChipWom;
pub use core::RegWriteCoreChipWom;

pub type RegWriteChipWom<F> =
    crate::wom_traits::VmChipWrapperWom<F, RegWriteAdapterChipWom<F>, RegWriteCoreChipWom>;
