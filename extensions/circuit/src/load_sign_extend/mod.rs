use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use crate::{VmChipWrapperWom, adapters::Rv32LoadStoreAdapterChip};

mod core;
pub use core::*;

pub type LoadSignExtendChip<F> = VmChipWrapperWom<
    F,
    Rv32LoadStoreAdapterChip<F>,
    LoadSignExtendCoreChip<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
