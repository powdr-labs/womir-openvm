use crate::{VmChipWrapperWom, adapters::JumpAdapterChipWom};

mod core;
pub use core::*;

pub type JumpChipWom<F> = VmChipWrapperWom<F, JumpAdapterChipWom<F>, JumpCoreChipWom>;
