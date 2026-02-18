use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::adapters::{
    BaseAluAdapterAir, BaseAluAdapterFiller, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS,
    Rv32BaseAluAdapterAir, Rv32BaseAluAdapterFiller,
};

mod core;
mod execution;

pub use core::{EqCoreAir, EqFiller};
pub use execution::EqExecutor;

pub type Rv32EqAir = VmAirWrapper<Rv32BaseAluAdapterAir, EqCoreAir<RV32_REGISTER_NUM_LIMBS>>;
pub type Rv32EqExecutor = EqExecutor<RV32_REGISTER_NUM_LIMBS, 1, RV32_CELL_BITS>;
pub type Rv32EqChip<F> =
    VmChipWrapper<F, EqFiller<Rv32BaseAluAdapterFiller<RV32_CELL_BITS>, RV32_REGISTER_NUM_LIMBS>>;

pub type Eq64Air = VmAirWrapper<BaseAluAdapterAir<8, 2>, EqCoreAir<8>>;
pub type Eq64Executor = EqExecutor<8, 2, RV32_CELL_BITS>;
pub type Eq64Chip<F> = VmChipWrapper<F, EqFiller<BaseAluAdapterFiller<2, RV32_CELL_BITS>, 8>>;
