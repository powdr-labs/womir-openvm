use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};
use openvm_rv32im_circuit::{ShiftCoreAir, ShiftFiller};

use super::adapters::{
    BaseAluAdapterAir, BaseAluAdapterFiller, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS,
    Rv32BaseAluAdapterAir, Rv32BaseAluAdapterFiller,
};

mod execution;

pub use execution::ShiftExecutor;

// 32-bit type aliases
pub type Rv32ShiftAir =
    VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>>;
pub type Rv32ShiftExecutor = ShiftExecutor<RV32_REGISTER_NUM_LIMBS, 1>;
pub type Rv32ShiftChip<F> = VmChipWrapper<
    F,
    ShiftFiller<Rv32BaseAluAdapterFiller<RV32_CELL_BITS>, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;

// 64-bit type aliases (NUM_REG_OPS=2: two 4-byte register operations per operand)
pub type Shift64Air = VmAirWrapper<BaseAluAdapterAir<8, 2>, ShiftCoreAir<8, RV32_CELL_BITS>>;
pub type Shift64Executor = ShiftExecutor<8, 2>;
pub type Shift64Chip<F> =
    VmChipWrapper<F, ShiftFiller<BaseAluAdapterFiller<2, RV32_CELL_BITS>, 8, RV32_CELL_BITS>>;
