use openvm_circuit::arch::{
    ChipInventory, ChipInventoryError, DenseRecordArena, VmProverExtension,
};
use openvm_cuda_backend::{engine::GpuBabyBearPoseidon2Engine, prover_backend::GpuBackend};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;

use super::Womir;

pub struct WomirGpuProverExt;

// Minimal GPU prover extension for WOMIR.
// Currently empty - only supports programs using system instructions (halt, etc.)
// that don't require WOMIR-specific tracegen.
impl VmProverExtension<GpuBabyBearPoseidon2Engine, DenseRecordArena, Womir> for WomirGpuProverExt {
    fn extend_prover(
        &self,
        _: &Womir,
        _inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        // TODO: Add WOMIR GPU chips here when we need more than just halt()
        Ok(())
    }
}
