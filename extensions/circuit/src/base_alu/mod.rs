use crate::adapters::WomBaseAluAdapterExecutor;

use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};

use openvm_rv32im_circuit::BaseAluExecutor;

pub type WomBaseAluExecutor = BaseAluExecutor<
    WomBaseAluAdapterExecutor<RV32_CELL_BITS>,
    RV32_REGISTER_NUM_LIMBS,
    RV32_CELL_BITS,
>;
