use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use crate::adapters::WomBaseAluAdapterExecutor;

use openvm_rv32im_circuit::BaseAluExecutor;

pub type WomBaseAlu64Executor<F> = BaseAluExecutor<
    WomBaseAluAdapterExecutor<F, { RV32_REGISTER_NUM_LIMBS * 2 }>,
    { RV32_REGISTER_NUM_LIMBS * 2 },
    RV32_CELL_BITS,
>;
