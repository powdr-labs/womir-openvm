mod core;

pub use core::*;

use crate::VmChipWrapperWom;

use super::adapters::{Rv32LoadStoreAdapterChip, RV32_REGISTER_NUM_LIMBS};

pub type Rv32LoadStoreChip<F> =
    VmChipWrapperWom<F, Rv32LoadStoreAdapterChip<F>, LoadStoreCoreChip<RV32_REGISTER_NUM_LIMBS>>;
