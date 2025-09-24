mod core;

pub use core::*;

use crate::VmChipWrapperWom;

use super::adapters::{RV32_REGISTER_NUM_LIMBS, Rv32LoadStoreAdapterChip};

pub type LoadStoreChip<F> =
    VmChipWrapperWom<F, Rv32LoadStoreAdapterChip<F>, LoadStoreCoreChip<RV32_REGISTER_NUM_LIMBS>>;
