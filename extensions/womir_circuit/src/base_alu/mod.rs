use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};
use openvm_rv32im_circuit::{BaseAluCoreAir, BaseAluFiller};

use super::adapters::{
    BaseAluAdapterAir, BaseAluAdapterFiller, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS,
    Rv32BaseAluAdapterAir, Rv32BaseAluAdapterFiller, W32_REG_OPS, W64_NUM_LIMBS, W64_REG_OPS,
};

mod execution;

pub use execution::BaseAluExecutor;

// 32-bit type aliases
pub type Rv32BaseAluAir =
    VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>>;
pub type Rv32BaseAluExecutor =
    BaseAluExecutor<RV32_REGISTER_NUM_LIMBS, W32_REG_OPS, RV32_CELL_BITS>;
pub type Rv32BaseAluChip<F> = VmChipWrapper<
    F,
    BaseAluFiller<
        Rv32BaseAluAdapterFiller<RV32_CELL_BITS>,
        RV32_REGISTER_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;

// 64-bit type aliases
pub type BaseAlu64Air = VmAirWrapper<
    BaseAluAdapterAir<W64_NUM_LIMBS, W64_REG_OPS>,
    BaseAluCoreAir<W64_NUM_LIMBS, RV32_CELL_BITS>,
>;
pub type BaseAlu64Executor = BaseAluExecutor<W64_NUM_LIMBS, W64_REG_OPS, RV32_CELL_BITS>;
pub type BaseAlu64Chip<F> = VmChipWrapper<
    F,
    BaseAluFiller<BaseAluAdapterFiller<W64_REG_OPS, RV32_CELL_BITS>, W64_NUM_LIMBS, RV32_CELL_BITS>,
>;
