use crate::VmChipWrapperWom;

use super::adapters::{RV32_REGISTER_NUM_LIMBS, Rv32LoadStoreAdapterChip};

use openvm_rv32im_circuit::LoadStoreCoreCols;

pub type LoadStoreChip<F> =
    VmChipWrapperWom<F, Rv32LoadStoreAdapterChip<F>, LoadStoreCoreCols<RV32_REGISTER_NUM_LIMBS>>;
