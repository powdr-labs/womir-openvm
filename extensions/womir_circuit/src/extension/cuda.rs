// Adapted from <openvm>/extensions/rv32im/circuit/src/extension/cuda.rs (stripped to BaseAlu only)
// Diff: https://gist.github.com/leonardoalt/09fd3d60bd571851bb656dc53cec0a4b#file-diff-extension-cuda-rs-diff
use openvm_circuit::{
    arch::{ChipInventory, ChipInventoryError, DenseRecordArena, VmProverExtension},
    system::cuda::extensions::{get_inventory_range_checker, get_or_create_bitwise_op_lookup},
};
use openvm_cuda_backend::{engine::GpuBabyBearPoseidon2Engine, prover_backend::GpuBackend};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;

use crate::{
    BaseAlu64Air, BaseAlu64ChipGpu, Rv32BaseAluAir, Rv32BaseAluChipGpu, Rv32ShiftAir,
    Rv32ShiftChipGpu, Shift64Air, Shift64ChipGpu,
};

use super::Womir;

pub struct WomirGpuProverExt;

impl VmProverExtension<GpuBabyBearPoseidon2Engine, DenseRecordArena, Womir> for WomirGpuProverExt {
    fn extend_prover(
        &self,
        _: &Womir,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let timestamp_max_bits = inventory.timestamp_max_bits();

        let range_checker = get_inventory_range_checker(inventory);
        let bitwise_lu = get_or_create_bitwise_op_lookup(inventory)?;

        inventory.next_air::<Rv32BaseAluAir>()?;
        let base_alu = Rv32BaseAluChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(base_alu);

        inventory.next_air::<BaseAlu64Air>()?;
        let base_alu_64 = BaseAlu64ChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(base_alu_64);

        inventory.next_air::<Rv32ShiftAir>()?;
        let shift = Rv32ShiftChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(shift);

        inventory.next_air::<Shift64Air>()?;
        let shift_64 = Shift64ChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(shift_64);

        // TODO: Add more WOMIR GPU chips here (mul, div, etc.)

        Ok(())
    }
}
