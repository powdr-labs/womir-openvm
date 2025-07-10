pub mod core;

use crate::adapters::Rv32AllocateFrameAdapterChipWom;
use core::Rv32AllocateFrameCoreChipWom;

pub type Rv32AllocateFrameChipWom<F> = crate::wom_traits::VmChipWrapperWom<
    F,
    Rv32AllocateFrameAdapterChipWom,
    Rv32AllocateFrameCoreChipWom,
>;
