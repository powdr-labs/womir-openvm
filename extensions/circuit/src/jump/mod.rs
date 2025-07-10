use crate::{adapters::JumpAdapterChipWom, VmChipWrapperWom};

mod core;
pub use core::*;

pub type JumpChipWom<F> = VmChipWrapperWom<F, JumpAdapterChipWom<F>, JumpCoreChipWom>;
