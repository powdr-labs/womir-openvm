use openvm_circuit::arch::VmChipWrapper;

use crate::adapters::Rv32RdWriteAdapterChip;

mod core;
pub use core::*;

pub type Rv32AuipcChip<F> = VmChipWrapper<F, Rv32RdWriteAdapterChip<F>, Rv32AuipcCoreChip>;
