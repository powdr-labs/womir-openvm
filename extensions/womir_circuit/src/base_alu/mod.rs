use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};
use openvm_rv32im_circuit::{BaseAluCoreAir, BaseAluFiller};

use super::adapters::{
    BaseAluAdapterAir, BaseAluAdapterExecutor, BaseAluAdapterFiller, RV32_CELL_BITS,
    RV32_REGISTER_NUM_LIMBS, Rv32BaseAluAdapterAir, Rv32BaseAluAdapterExecutor,
    Rv32BaseAluAdapterFiller,
};

mod execution;

pub use execution::BaseAluExecutor;

// 32-bit type aliases
pub type Rv32BaseAluAir =
    VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>>;
pub type Rv32BaseAluExecutor = BaseAluExecutor<
    Rv32BaseAluAdapterExecutor<RV32_CELL_BITS>,
    RV32_REGISTER_NUM_LIMBS,
    RV32_CELL_BITS,
>;
pub type Rv32BaseAluChip<F> = VmChipWrapper<
    F,
    BaseAluFiller<
        Rv32BaseAluAdapterFiller<RV32_CELL_BITS>,
        RV32_REGISTER_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;

// 64-bit type aliases
pub type BaseAlu64Air = VmAirWrapper<BaseAluAdapterAir<8>, BaseAluCoreAir<8, RV32_CELL_BITS>>;
pub type BaseAlu64Executor =
    BaseAluExecutor<BaseAluAdapterExecutor<8, RV32_CELL_BITS>, 8, RV32_CELL_BITS>;
pub type BaseAlu64Chip<F> =
    VmChipWrapper<F, BaseAluFiller<BaseAluAdapterFiller<8, RV32_CELL_BITS>, 8, RV32_CELL_BITS>>;
