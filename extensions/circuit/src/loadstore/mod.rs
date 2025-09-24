mod core;

use crate::VmChipWrapperWom;

use super::adapters::{Rv32LoadStoreAdapterChip, RV32_REGISTER_NUM_LIMBS};

use openvm_rv32im_circuit::LoadStoreCoreChip;

pub type LoadStoreChip<F> =
    VmChipWrapperWom<F, Rv32LoadStoreAdapterChip<F>, LoadStoreCoreChip<RV32_REGISTER_NUM_LIMBS>>;
