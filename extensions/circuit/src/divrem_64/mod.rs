use crate::adapters::WomBaseAluAdapterExecutor;

use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};

use openvm_rv32im_circuit::DivRemExecutor;

pub type WomDivRem64Executor<F> = DivRemExecutor<
    WomBaseAluAdapterExecutor<F, { RV32_REGISTER_NUM_LIMBS * 2 }>,
    { RV32_REGISTER_NUM_LIMBS * 2 },
    RV32_CELL_BITS,
>;
