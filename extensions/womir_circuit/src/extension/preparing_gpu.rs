//! WOMIR extension for incremental GPU testing (CUDA only).
//!
//! This module contains types with "PreparingGpu" suffix that are temporary
//! scaffolding for testing GPU chips one at a time. Once all WOMIR chips have
//! GPU implementations, these types will be removed and the main WomirConfig
//! will be used for GPU proving.
//!
//! Currently includes: BaseAlu32

use derive_more::derive::From;
use openvm_circuit::{
    arch::{
        AirInventory, AirInventoryError, ChipInventory, ChipInventoryError, DenseRecordArena,
        ExecutorInventoryBuilder, ExecutorInventoryError, VmCircuitExtension, VmExecutionExtension,
        VmProverExtension,
    },
    system::{
        SystemPort,
        cuda::extensions::{get_inventory_range_checker, get_or_create_bitwise_op_lookup},
    },
};
use openvm_circuit_derive::{AnyEnum, Executor, MeteredExecutor, PreflightExecutor};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, BitwiseOperationLookupBus,
};
use openvm_cuda_backend::{engine::GpuBabyBearPoseidon2Engine, prover_backend::GpuBackend};
use openvm_instructions::LocalOpcode;
use openvm_rv32im_circuit::BaseAluCoreAir;
use openvm_stark_backend::{config::StarkGenericConfig, p3_field::PrimeField32};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use openvm_womir_transpiler::BaseAluOpcode;
use serde::{Deserialize, Serialize};
use strum::IntoEnumIterator;

use openvm_circuit::arch::ExecutionBridge;

use crate::{
    Rv32BaseAluAir, Rv32BaseAluChipGpu, Rv32BaseAluExecutor, adapters::Rv32BaseAluAdapterAir,
};

// ============ Extension Struct ============

/// WOMIR extension for incremental GPU testing.
/// Contains only GPU-ready chips (currently: BaseAlu32).
/// Will be removed once all chips have GPU implementations.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct WomirPreparingGpu;

// ============ Executor Enum ============

/// Executor enum for WomirPreparingGpu extension.
#[derive(Clone, From, AnyEnum, Executor, MeteredExecutor, PreflightExecutor)]
pub enum WomirPreparingGpuExecutor {
    BaseAlu(Rv32BaseAluExecutor),
}

// ============ VmExtension Implementations ============

impl<F: PrimeField32> VmExecutionExtension<F> for WomirPreparingGpu {
    type Executor = WomirPreparingGpuExecutor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<F, WomirPreparingGpuExecutor>,
    ) -> Result<(), ExecutorInventoryError> {
        use crate::adapters::Rv32BaseAluAdapterExecutor;

        let base_alu = Rv32BaseAluExecutor::new(
            Rv32BaseAluAdapterExecutor::default(),
            BaseAluOpcode::CLASS_OFFSET,
        );
        inventory.add_executor(base_alu, BaseAluOpcode::iter().map(|x| x.global_opcode()))?;

        Ok(())
    }
}

impl<SC: StarkGenericConfig> VmCircuitExtension<SC> for WomirPreparingGpu {
    fn extend_circuit(&self, inventory: &mut AirInventory<SC>) -> Result<(), AirInventoryError> {
        let SystemPort {
            execution_bus,
            program_bus,
            memory_bridge,
        } = inventory.system().port();

        let exec_bridge = ExecutionBridge::new(execution_bus, program_bus);

        let bitwise_lu = {
            let existing_air = inventory.find_air::<BitwiseOperationLookupAir<8>>().next();
            if let Some(air) = existing_air {
                air.bus
            } else {
                let bus = BitwiseOperationLookupBus::new(inventory.new_bus_idx());
                let air = BitwiseOperationLookupAir::<8>::new(bus);
                inventory.add_air(air);
                air.bus
            }
        };

        let base_alu = Rv32BaseAluAir::new(
            Rv32BaseAluAdapterAir::new(exec_bridge, memory_bridge, bitwise_lu),
            BaseAluCoreAir::new(bitwise_lu, BaseAluOpcode::CLASS_OFFSET),
        );
        inventory.add_air(base_alu);

        Ok(())
    }
}

// ============ GPU Prover Extension ============

pub struct WomirPreparingGpuProverExt;

impl VmProverExtension<GpuBabyBearPoseidon2Engine, DenseRecordArena, WomirPreparingGpu>
    for WomirPreparingGpuProverExt
{
    fn extend_prover(
        &self,
        _: &WomirPreparingGpu,
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

        Ok(())
    }
}
