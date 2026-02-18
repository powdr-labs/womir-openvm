use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};
use openvm_rv32im_circuit::{MultiplicationCoreAir, MultiplicationFiller};

use super::adapters::{
    BaseAluAdapterAir, BaseAluAdapterFiller, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS,
    Rv32BaseAluAdapterAir, Rv32BaseAluAdapterFiller,
};

mod execution;

pub use execution::MultiplicationExecutor;

// 32-bit type aliases
pub type Rv32MultiplicationAir = VmAirWrapper<
    Rv32BaseAluAdapterAir,
    MultiplicationCoreAir<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
pub type Rv32MultiplicationExecutor =
    MultiplicationExecutor<RV32_REGISTER_NUM_LIMBS, 1, RV32_CELL_BITS>;
pub type Rv32MultiplicationChip<F> = VmChipWrapper<
    F,
    MultiplicationFiller<
        Rv32BaseAluAdapterFiller<RV32_CELL_BITS>,
        RV32_REGISTER_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;

// 64-bit type aliases (NUM_REG_OPS=2: two 4-byte register operations per operand)
pub type Mul64Air = VmAirWrapper<BaseAluAdapterAir<8, 2>, MultiplicationCoreAir<8, RV32_CELL_BITS>>;
pub type Mul64Executor = MultiplicationExecutor<8, 2, RV32_CELL_BITS>;
pub type Mul64Chip<F> = VmChipWrapper<
    F,
    MultiplicationFiller<BaseAluAdapterFiller<2, RV32_CELL_BITS>, 8, RV32_CELL_BITS>,
>;
