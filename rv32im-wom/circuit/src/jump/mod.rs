use crate::{adapters::Rv32JumpAdapterChipWom, VmChipWrapperWom};

mod core;
pub use core::*;

pub type Rv32JumpChipWom<F> = VmChipWrapperWom<F, Rv32JumpAdapterChipWom<F>, Rv32JumpCoreChipWom>;
