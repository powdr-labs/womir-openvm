use openvm_circuit::arch::VmChipWrapper;

// Re-use upstream types (exported at crate level)
pub use openvm_rv32im_circuit::{
    Rv32HintStoreAir as HintStoreAir, Rv32HintStoreCols as HintStoreCols,
    Rv32HintStoreFiller as HintStoreFiller,
    Rv32HintStoreRecordHeader as HintStoreRecordHeader,
    Rv32HintStoreVar as HintStoreVar,
};

mod core;
pub use core::*;

mod execution;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

// #[cfg(test)]
// mod tests;  // TODO: Update tests for FP-aware executor

// Type alias for HintStore executor with FP support
pub type HintStoreExecutor32 = crate::PreflightExecutorWrapperFp<
    HintStoreCoreExecutor,
    { crate::adapters::RV32_REGISTER_NUM_LIMBS },
    { crate::adapters::RV32_CELL_BITS },
>;

pub type HintStoreChip<F> = VmChipWrapper<F, HintStoreFiller>;
