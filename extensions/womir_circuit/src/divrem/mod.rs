use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};
use openvm_rv32im_circuit::{DivRemCoreAir, DivRemFiller};

use super::adapters::{
    BaseAluAdapterAir, BaseAluAdapterFiller, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS,
    Rv32BaseAluAdapterAir, Rv32BaseAluAdapterFiller,
};

mod execution;

pub use execution::DivRemExecutor;

// 32-bit type aliases
pub type Rv32DivRemAir =
    VmAirWrapper<Rv32BaseAluAdapterAir, DivRemCoreAir<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>>;
pub type Rv32DivRemExecutor = DivRemExecutor<RV32_REGISTER_NUM_LIMBS, 1, RV32_CELL_BITS>;
pub type Rv32DivRemChip<F> = VmChipWrapper<
    F,
    DivRemFiller<Rv32BaseAluAdapterFiller<RV32_CELL_BITS>, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;

// 64-bit type aliases (NUM_REG_OPS=2: two 4-byte register operations per operand)
pub type DivRem64Air = VmAirWrapper<BaseAluAdapterAir<8, 2>, DivRemCoreAir<8, RV32_CELL_BITS>>;
pub type DivRem64Executor = DivRemExecutor<8, 2, RV32_CELL_BITS>;
pub type DivRem64Chip<F> =
    VmChipWrapper<F, DivRemFiller<BaseAluAdapterFiller<2, RV32_CELL_BITS>, 8, RV32_CELL_BITS>>;
