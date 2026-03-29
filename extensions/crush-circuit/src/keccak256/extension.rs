use std::sync::{Arc, Mutex};

use derive_more::derive::From;
use openvm_circuit::{
    arch::{
        AirInventory, AirInventoryError, ChipInventory, ChipInventoryError, ExecutionBridge,
        ExecutorInventoryBuilder, ExecutorInventoryError, RowMajorMatrixArena,
        VmCircuitExtension, VmExecutionExtension, VmProverExtension,
    },
    system::{memory::SharedMemoryHelper, SystemPort},
};
use openvm_circuit_derive::{AnyEnum, Executor, MeteredExecutor, PreflightExecutor};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
    SharedBitwiseOperationLookupChip,
};
use openvm_crush_transpiler::{KeccakfOpcode, XorinOpcode};
use openvm_instructions::LocalOpcode;
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    engine::StarkEngine,
    interaction::PermutationCheckBus,
    p3_field::PrimeField32,
    prover::cpu::{CpuBackend, CpuDevice},
};
use serde::{Deserialize, Serialize};
use strum::IntoEnumIterator;

use crate::keccak256::{
    keccakf_op::{KeccakfExecutor, KeccakfOpAir, KeccakfOpChip},
    keccakf_perm::{KeccakfPermAir, KeccakfPermChip},
    xorin::{air::XorinVmAir, XorinVmChip, XorinVmExecutor, XorinVmFiller},
};

// =================================== VM Extension Implementation =================================
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Keccak256;

#[derive(Clone, Copy, From, AnyEnum, Executor, MeteredExecutor, PreflightExecutor)]
pub enum Keccak256Executor {
    Keccakf(KeccakfExecutor),
    Xorin(XorinVmExecutor),
}

impl<F: PrimeField32> VmExecutionExtension<F> for Keccak256 {
    type Executor = Keccak256Executor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<F, Keccak256Executor>,
    ) -> Result<(), ExecutorInventoryError> {
        let pointer_max_bits = inventory.pointer_max_bits();

        let xorin_executor = XorinVmExecutor::new(XorinOpcode::CLASS_OFFSET, pointer_max_bits);
        inventory.add_executor(
            xorin_executor,
            XorinOpcode::iter().map(|x| x.global_opcode()),
        )?;

        let keccak_executor = KeccakfExecutor::new(KeccakfOpcode::CLASS_OFFSET, pointer_max_bits);
        inventory.add_executor(
            keccak_executor,
            KeccakfOpcode::iter().map(|x| x.global_opcode()),
        )?;

        Ok(())
    }
}

impl<SC: StarkGenericConfig> VmCircuitExtension<SC> for Keccak256 {
    fn extend_circuit(&self, inventory: &mut AirInventory<SC>) -> Result<(), AirInventoryError> {
        let SystemPort {
            execution_bus,
            program_bus,
            memory_bridge,
        } = inventory.system().port();

        let exec_bridge = ExecutionBridge::new(execution_bus, program_bus);
        let pointer_max_bits = inventory.pointer_max_bits();

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

        let xorin_air = XorinVmAir::new(
            exec_bridge,
            memory_bridge,
            bitwise_lu,
            pointer_max_bits,
            XorinOpcode::CLASS_OFFSET,
        );
        inventory.add_air(xorin_air);

        let keccakf_state_bus = PermutationCheckBus::new(inventory.new_bus_idx());
        let periphery_air = KeccakfPermAir::new(keccakf_state_bus);
        inventory.add_air(periphery_air);

        let op_air = KeccakfOpAir::new(
            exec_bridge,
            memory_bridge,
            bitwise_lu,
            keccakf_state_bus,
            pointer_max_bits,
            KeccakfOpcode::CLASS_OFFSET,
        );
        inventory.add_air(op_air);

        Ok(())
    }
}

pub struct Keccak256CpuProverExt;

impl<E, SC, RA> VmProverExtension<E, RA, Keccak256> for Keccak256CpuProverExt
where
    SC: StarkGenericConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    RA: RowMajorMatrixArena<Val<SC>>,
    Val<SC>: PrimeField32,
{
    fn extend_prover(
        &self,
        _: &Keccak256,
        inventory: &mut ChipInventory<SC, RA, CpuBackend<SC>>,
    ) -> Result<(), ChipInventoryError> {
        let range_checker = inventory.range_checker()?.clone();
        let timestamp_max_bits = inventory.timestamp_max_bits();
        let mem_helper = SharedMemoryHelper::new(range_checker, timestamp_max_bits);
        let pointer_max_bits = inventory.airs().pointer_max_bits();

        let bitwise_lu = {
            let existing_chip = inventory
                .find_chip::<SharedBitwiseOperationLookupChip<8>>()
                .next();

            if let Some(chip) = existing_chip {
                chip.clone()
            } else {
                let air: &BitwiseOperationLookupAir<8> = inventory.next_air()?;
                let chip = Arc::new(BitwiseOperationLookupChip::new(air.bus));
                inventory.add_periphery_chip(chip.clone());
                chip
            }
        };

        inventory.next_air::<XorinVmAir>()?;
        let xorin_chip = XorinVmChip::new(
            XorinVmFiller::new(bitwise_lu.clone(), pointer_max_bits),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(xorin_chip);

        inventory.next_air::<KeccakfPermAir>()?;
        let shared_records = Arc::new(Mutex::new(Vec::new()));
        let periphery_chip = KeccakfPermChip::new(shared_records.clone());
        // WARNING: the OpChip must be added _after_ the periphery chip so that its tracegen is done
        // _first_. After OpChip tracegen, the shared_record is set to the execution records,
        // effectively passing the records to the periphery chip.
        inventory.add_periphery_chip(periphery_chip);

        inventory.next_air::<KeccakfOpAir>()?;
        let op_chip = KeccakfOpChip::new(
            bitwise_lu,
            pointer_max_bits,
            mem_helper,
            shared_records,
        );
        inventory.add_executor_chip(op_chip);

        Ok(())
    }
}
