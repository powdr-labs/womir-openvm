pub mod core;

use crate::adapters::Rv32CopyIntoFrameAdapterChipWom;
use core::Rv32CopyIntoFrameCoreChipWom;

pub type Rv32CopyIntoFrameChipWom<F> = crate::wom_traits::VmChipWrapperWom<
    F,
    Rv32CopyIntoFrameAdapterChipWom<F>,
    Rv32CopyIntoFrameCoreChipWom,
>;
