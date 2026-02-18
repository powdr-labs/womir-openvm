use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};
use openvm_rv32im_circuit::{BaseAluCoreAir, BaseAluFiller};

use super::adapters::{
    BaseAluAdapterAir, BaseAluAdapterFiller, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS,
    Rv32BaseAluAdapterAir, Rv32BaseAluAdapterFiller,
};

mod execution;

pub use execution::BaseAluExecutor;

// 32-bit type aliases
pub type Rv32BaseAluAir =
    VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>>;
pub type Rv32BaseAluExecutor = BaseAluExecutor<RV32_REGISTER_NUM_LIMBS, 1>;
pub type Rv32BaseAluChip<F> = VmChipWrapper<
    F,
    BaseAluFiller<
        Rv32BaseAluAdapterFiller<RV32_CELL_BITS>,
        RV32_REGISTER_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;

// 64-bit type aliases (NUM_REG_OPS=2: two 4-byte register operations per operand)
pub type BaseAlu64Air = VmAirWrapper<BaseAluAdapterAir<8, 2>, BaseAluCoreAir<8, RV32_CELL_BITS>>;
pub type BaseAlu64Executor = BaseAluExecutor<8, 2>;
pub type BaseAlu64Chip<F> =
    VmChipWrapper<F, BaseAluFiller<BaseAluAdapterFiller<2, RV32_CELL_BITS>, 8, RV32_CELL_BITS>>;
