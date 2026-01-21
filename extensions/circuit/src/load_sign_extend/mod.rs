use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use crate::{VmChipWrapperWom, adapters::Rv32LoadStoreAdapterChip};

use openvm_rv32im_circuit::LoadSignExtendCoreCols;

pub type LoadSignExtendChip<F> = VmChipWrapperWom<
    F,
    Rv32LoadStoreAdapterChip<F>,
    LoadSignExtendCoreCols<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
