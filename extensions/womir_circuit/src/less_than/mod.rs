use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};
use openvm_rv32im_circuit::{LessThanCoreAir, LessThanFiller};

use super::adapters::{
    RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS, Rv32BaseAluAdapterAir, Rv32BaseAluAdapterExecutor,
    Rv32BaseAluAdapterFiller,
};

mod execution;

use execution::LessThanExecutor;

pub type Rv32LessThanAir =
    VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>>;
pub type Rv32LessThanExecutor = LessThanExecutor<
    Rv32BaseAluAdapterExecutor<RV32_CELL_BITS>,
    RV32_REGISTER_NUM_LIMBS,
    RV32_CELL_BITS,
>;
pub type Rv32LessThanChip<F> = VmChipWrapper<
    F,
    LessThanFiller<
        Rv32BaseAluAdapterFiller<RV32_CELL_BITS>,
        RV32_REGISTER_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;
