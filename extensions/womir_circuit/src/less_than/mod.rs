use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};
use openvm_rv32im_circuit::{LessThanCoreAir, LessThanFiller};

use super::adapters::{
    BaseAluAdapterAir, BaseAluAdapterFiller, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS,
    Rv32BaseAluAdapterAir, Rv32BaseAluAdapterFiller,
};

mod execution;

pub use execution::LessThanExecutor;

// 32-bit type aliases
pub type Rv32LessThanAir =
    VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>>;
pub type Rv32LessThanExecutor = LessThanExecutor<RV32_REGISTER_NUM_LIMBS, 1>;
pub type Rv32LessThanChip<F> = VmChipWrapper<
    F,
    LessThanFiller<
        Rv32BaseAluAdapterFiller<RV32_CELL_BITS>,
        RV32_REGISTER_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;

// 64-bit type aliases (NUM_REG_OPS=2: two 4-byte register operations per operand)
pub type LessThan64Air = VmAirWrapper<BaseAluAdapterAir<8, 2>, LessThanCoreAir<8, RV32_CELL_BITS>>;
pub type LessThan64Executor = LessThanExecutor<8, 2>;
pub type LessThan64Chip<F> =
    VmChipWrapper<F, LessThanFiller<BaseAluAdapterFiller<2, RV32_CELL_BITS>, 8, RV32_CELL_BITS>>;
