use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};
use openvm_rv32im_circuit::{MultiplicationCoreAir, MultiplicationFiller};

use super::adapters::{
    BaseAluAdapterAir, BaseAluAdapterFiller, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS,
    Rv32BaseAluAdapterAir, Rv32BaseAluAdapterFiller, W32_REG_OPS, W64_NUM_LIMBS, W64_REG_OPS,
};

mod execution;

pub use execution::MultiplicationExecutor;

// 32-bit type aliases
pub type Rv32MultiplicationAir = VmAirWrapper<
    Rv32BaseAluAdapterAir,
    MultiplicationCoreAir<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
pub type Rv32MultiplicationExecutor = MultiplicationExecutor<RV32_REGISTER_NUM_LIMBS, W32_REG_OPS>;
pub type Rv32MultiplicationChip<F> = VmChipWrapper<
    F,
    MultiplicationFiller<
        Rv32BaseAluAdapterFiller<RV32_CELL_BITS>,
        RV32_REGISTER_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;

// 64-bit type aliases
pub type Mul64Air = VmAirWrapper<
    BaseAluAdapterAir<W64_NUM_LIMBS, W64_REG_OPS>,
    MultiplicationCoreAir<W64_NUM_LIMBS, RV32_CELL_BITS>,
>;
pub type Mul64Executor = MultiplicationExecutor<W64_NUM_LIMBS, W64_REG_OPS>;
pub type Mul64Chip<F> = VmChipWrapper<
    F,
    MultiplicationFiller<
        BaseAluAdapterFiller<W64_REG_OPS, RV32_CELL_BITS>,
        W64_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;
