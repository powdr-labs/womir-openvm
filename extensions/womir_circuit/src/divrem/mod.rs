use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};
use openvm_rv32im_circuit::{DivRemCoreAir, DivRemFiller};

use super::adapters::{
    BaseAluAdapterAir, BaseAluAdapterFiller, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS,
    Rv32BaseAluAdapterAir, Rv32BaseAluAdapterFiller, W32_REG_OPS, W64_NUM_LIMBS, W64_REG_OPS,
};

mod execution;

pub use execution::DivRemExecutor;

// 32-bit type aliases
pub type Rv32DivRemAir =
    VmAirWrapper<Rv32BaseAluAdapterAir, DivRemCoreAir<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>>;
pub type Rv32DivRemExecutor = DivRemExecutor<RV32_REGISTER_NUM_LIMBS, W32_REG_OPS>;
pub type Rv32DivRemChip<F> = VmChipWrapper<
    F,
    DivRemFiller<Rv32BaseAluAdapterFiller, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;

// 64-bit type aliases
pub type DivRem64Air = VmAirWrapper<
    BaseAluAdapterAir<W64_NUM_LIMBS, W64_REG_OPS>,
    DivRemCoreAir<W64_NUM_LIMBS, RV32_CELL_BITS>,
>;
pub type DivRem64Executor = DivRemExecutor<W64_NUM_LIMBS, W64_REG_OPS>;
pub type DivRem64Chip<F> = VmChipWrapper<
    F,
    DivRemFiller<BaseAluAdapterFiller<W64_REG_OPS>, W64_NUM_LIMBS, RV32_CELL_BITS>,
>;
