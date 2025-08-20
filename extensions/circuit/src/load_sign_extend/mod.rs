use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use crate::{adapters::Rv32LoadStoreAdapterChip, VmChipWrapperWom};

mod core;
pub use core::*;

pub type Rv32LoadSignExtendChip<F> = VmChipWrapperWom<
    F,
    Rv32LoadStoreAdapterChip<F>,
    LoadSignExtendCoreChip<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
