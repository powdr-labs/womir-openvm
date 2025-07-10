pub mod core;

use crate::adapters::AllocateFrameAdapterChipWom;
use core::AllocateFrameCoreChipWom;

pub type AllocateFrameChipWom<F> =
    crate::wom_traits::VmChipWrapperWom<F, AllocateFrameAdapterChipWom, AllocateFrameCoreChipWom>;
