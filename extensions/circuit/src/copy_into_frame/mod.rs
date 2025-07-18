mod core;

use crate::adapters::CopyIntoFrameAdapterChipWom;
pub use core::CopyIntoFrameCoreChipWom;

pub type CopyIntoFrameChipWom<F> = crate::wom_traits::VmChipWrapperWom<
    F,
    CopyIntoFrameAdapterChipWom<F>,
    CopyIntoFrameCoreChipWom,
>;
