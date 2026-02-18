use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};
use openvm_rv32im_circuit::{LoadSignExtendCoreAir, LoadSignExtendFiller};

use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use crate::{
    adapters::{Rv32LoadStoreAdapterAir, Rv32LoadStoreAdapterFiller},
    load_sign_extend::execution::LoadSignExtendExecutor,
};

pub mod execution;

pub type Rv32LoadSignExtendAir = VmAirWrapper<
    Rv32LoadStoreAdapterAir,
    LoadSignExtendCoreAir<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
pub type Rv32LoadSignExtendExecutor =
    LoadSignExtendExecutor<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>;
pub type Rv32LoadSignExtendChip<F> =
    VmChipWrapper<F, LoadSignExtendFiller<Rv32LoadStoreAdapterFiller>>;
