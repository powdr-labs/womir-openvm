use openvm_circuit::arch::VmChipWrapper;

use crate::adapters::Rv32CondRdWriteAdapterChip;

mod core;
pub use core::*;

pub type Rv32JalLuiChip<F> = VmChipWrapper<F, Rv32CondRdWriteAdapterChip<F>, Rv32JalLuiCoreChip>;
