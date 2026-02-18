use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};
use openvm_rv32im_circuit::{LoadStoreCoreAir, LoadStoreFiller};

use super::adapters::RV32_REGISTER_NUM_LIMBS;
use crate::{
    adapters::{Rv32LoadStoreAdapterAir, Rv32LoadStoreAdapterFiller},
    loadstore::execution::LoadStoreExecutor,
};

pub mod execution;

pub type Rv32LoadStoreAir =
    VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<RV32_REGISTER_NUM_LIMBS>>;
pub type Rv32LoadStoreExecutor =
    LoadStoreExecutor<RV32_REGISTER_NUM_LIMBS>;
pub type Rv32LoadStoreChip<F> = VmChipWrapper<F, LoadStoreFiller<Rv32LoadStoreAdapterFiller>>;
