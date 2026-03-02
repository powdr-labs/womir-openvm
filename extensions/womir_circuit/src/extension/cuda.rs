use openvm_circuit::{
    arch::{ChipInventory, ChipInventoryError, DenseRecordArena, VmProverExtension},
    system::cuda::extensions::{get_inventory_range_checker, get_or_create_bitwise_op_lookup},
};
use openvm_cuda_backend::{engine::GpuBabyBearPoseidon2Engine, prover_backend::GpuBackend};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;

use crate::{Rv32BaseAluAir, Rv32BaseAluChipGpu};

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

        // TODO: Add more WOMIR GPU chips here (64-bit ALU, mul, div, etc.)

        Ok(())
    }
}
