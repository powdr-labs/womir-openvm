use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};
use openvm_rv32im_circuit::{ShiftCoreAir, ShiftFiller};

use super::adapters::{
    BaseAluAdapterAir, BaseAluAdapterFiller, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS,
    Rv32BaseAluAdapterAir, Rv32BaseAluAdapterFiller, W32_REG_OPS, W64_NUM_LIMBS, W64_REG_OPS,
};

mod execution;

pub use execution::ShiftExecutor;

// 32-bit type aliases
pub type Rv32ShiftAir =
    VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>>;
pub type Rv32ShiftExecutor = ShiftExecutor<RV32_REGISTER_NUM_LIMBS, W32_REG_OPS, RV32_CELL_BITS>;
pub type Rv32ShiftChip<F> = VmChipWrapper<
    F,
    ShiftFiller<Rv32BaseAluAdapterFiller<RV32_CELL_BITS>, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;

// 64-bit type aliases
pub type Shift64Air = VmAirWrapper<
    BaseAluAdapterAir<W64_NUM_LIMBS, W64_REG_OPS, W64_REG_OPS>,
    ShiftCoreAir<W64_NUM_LIMBS, RV32_CELL_BITS>,
>;
pub type Shift64Executor = ShiftExecutor<W64_NUM_LIMBS, W64_REG_OPS, RV32_CELL_BITS>;
pub type Shift64Chip<F> = VmChipWrapper<
    F,
    ShiftFiller<
        BaseAluAdapterFiller<W64_REG_OPS, W64_REG_OPS, RV32_CELL_BITS>,
        W64_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;
