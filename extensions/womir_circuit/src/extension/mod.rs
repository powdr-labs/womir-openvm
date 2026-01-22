use std::sync::Arc;

use derive_more::derive::From;
use openvm_circuit::{
    arch::{
        AirInventory, AirInventoryError, ChipInventory, ChipInventoryError, ExecutionBridge,
        ExecutorInventoryBuilder, ExecutorInventoryError, RowMajorMatrixArena, VmCircuitExtension,
        VmExecutionExtension, VmProverExtension,
    },
    system::{SystemPort, memory::SharedMemoryHelper},
};
use openvm_circuit_derive::{AnyEnum, PreflightExecutor};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{
        BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
        SharedBitwiseOperationLookupChip,
    },
    range_tuple::{
        RangeTupleCheckerAir, RangeTupleCheckerBus, RangeTupleCheckerChip,
        SharedRangeTupleCheckerChip,
    },
};
use openvm_instructions::{LocalOpcode, PhantomDiscriminant, program::DEFAULT_PC_STEP};
use openvm_rv32im_transpiler::{
    BranchEqualOpcode, BranchLessThanOpcode, DivRemOpcode, LessThanOpcode, MulHOpcode, MulOpcode,
    Rv32AuipcOpcode, Rv32HintStoreOpcode, Rv32JalLuiOpcode, Rv32JalrOpcode, Rv32Phantom,
    ShiftOpcode,
};
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    engine::StarkEngine,
    p3_field::PrimeField32,
    prover::cpu::{CpuBackend, CpuDevice},
};
use openvm_womir_transpiler::{BaseAluOpcode, LoadStoreOpcode};
use serde::{Deserialize, Serialize};
use strum::IntoEnumIterator;

use crate::{adapters::*, *};

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        mod cuda;
        pub use cuda::{
            WomirGpuProverExt as WomirProverExt,
        };
    } else {
        pub use self::{
            WomirCpuProverExt as WomirProverExt,
        };
    }
}

// ============ Extension Struct Definitions ============

/// RISC-V 32-bit Base (RV32I) Extension
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Womir {
    #[serde(default = "default_range_tuple_checker_sizes")]
    pub range_tuple_checker_sizes: [u32; 2],
}

impl Default for Womir {
    fn default() -> Self {
        Self {
            range_tuple_checker_sizes: default_range_tuple_checker_sizes(),
        }
    }
}

fn default_range_tuple_checker_sizes() -> [u32; 2] {
    [1 << 8, 8 * (1 << 8)]
}

// ============ Executor and Periphery Enums for Extension ============

/// RISC-V 32-bit Base (RV32I) Instruction Executors
// ITS THIS DERIVES FAULT; not supporting aot traits?
#[derive(Clone, From, AnyEnum, Executor, MeteredExecutor, PreflightExecutor)]
#[cfg_attr(
    feature = "aot",
    derive(
        openvm_circuit_derive::AotExecutor,
        openvm_circuit_derive::AotMeteredExecutor
    )
)]
pub enum WomirExecutor {
    // Rv32 (for standard 32-bit integers):
    BaseAlu(Rv32BaseAluExecutor),
    // LessThan(Rv32LessThanExecutor),
    // Shift(Rv32ShiftExecutor),
    LoadStore(Rv32LoadStoreExecutor),
    LoadSignExtend(Rv32LoadSignExtendExecutor),
    // BranchEqual(Rv32BranchEqualExecutor),
    // BranchLessThan(Rv32BranchLessThanExecutor),
    // JalLui(Rv32JalLuiExecutor),
    // Jalr(Rv32JalrExecutor),
    // Auipc(Rv32AuipcExecutor),
}

// ============ VmExtension Implementations ============

impl<F: PrimeField32> VmExecutionExtension<F> for Womir {
    type Executor = WomirExecutor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<F, WomirExecutor>,
    ) -> Result<(), ExecutorInventoryError> {
        let pointer_max_bits = inventory.pointer_max_bits();

        let fp = std::sync::Arc::new(std::sync::Mutex::new(0));
        let base_alu = Rv32BaseAluExecutor::new(
            BaseAluExecutor::new(
                Rv32BaseAluAdapterExecutor::new(),
                BaseAluOpcode::CLASS_OFFSET
            ),
            fp
        );
        inventory.add_executor(base_alu, BaseAluOpcode::iter().map(|x| x.global_opcode()))?;
        //
        // let lt = LessThanExecutor::new(Rv32BaseAluAdapterExecutor, LessThanOpcode::CLASS_OFFSET);
        // inventory.add_executor(lt, LessThanOpcode::iter().map(|x| x.global_opcode()))?;
        //
        // let shift = ShiftExecutor::new(Rv32BaseAluAdapterExecutor, ShiftOpcode::CLASS_OFFSET);
        // inventory.add_executor(shift, ShiftOpcode::iter().map(|x| x.global_opcode()))?;
        //
        let load_store = LoadStoreExecutor::new(
            Rv32LoadStoreAdapterExecutor::new(pointer_max_bits),
            LoadStoreOpcode::CLASS_OFFSET,
        );
        inventory.add_executor(
            load_store,
            LoadStoreOpcode::iter()
                .take(LoadStoreOpcode::STOREB as usize + 1)
                .map(|x| x.global_opcode()),
        )?;

        let load_sign_extend =
            LoadSignExtendExecutor::new(Rv32LoadStoreAdapterExecutor::new(pointer_max_bits));
        inventory.add_executor(
            load_sign_extend,
            [LoadStoreOpcode::LOADB, LoadStoreOpcode::LOADH].map(|x| x.global_opcode()),
        )?;

        // let beq = BranchEqualExecutor::new(
        //     Rv32BranchAdapterExecutor,
        //     BranchEqualOpcode::CLASS_OFFSET,
        //     DEFAULT_PC_STEP,
        // );
        // inventory.add_executor(beq, BranchEqualOpcode::iter().map(|x| x.global_opcode()))?;
        //
        // let blt = BranchLessThanExecutor::new(
        //     Rv32BranchAdapterExecutor,
        //     BranchLessThanOpcode::CLASS_OFFSET,
        // );
        // inventory.add_executor(blt, BranchLessThanOpcode::iter().map(|x| x.global_opcode()))?;
        //
        // let jal_lui = Rv32JalLuiExecutor::new(Rv32CondRdWriteAdapterExecutor::new(
        //     Rv32RdWriteAdapterExecutor,
        // ));
        // inventory.add_executor(jal_lui, Rv32JalLuiOpcode::iter().map(|x| x.global_opcode()))?;
        //
        // let jalr = Rv32JalrExecutor::new(Rv32JalrAdapterExecutor);
        // inventory.add_executor(jalr, Rv32JalrOpcode::iter().map(|x| x.global_opcode()))?;
        //
        // let auipc = Rv32AuipcExecutor::new(Rv32RdWriteAdapterExecutor);
        // inventory.add_executor(auipc, Rv32AuipcOpcode::iter().map(|x| x.global_opcode()))?;
        //
        // // There is no downside to adding phantom sub-executors, so we do it in the base extension.
        // inventory.add_phantom_sub_executor(
        //     phantom::Rv32HintInputSubEx,
        //     PhantomDiscriminant(Rv32Phantom::HintInput as u16),
        // )?;
        // inventory.add_phantom_sub_executor(
        //     phantom::Rv32HintRandomSubEx,
        //     PhantomDiscriminant(Rv32Phantom::HintRandom as u16),
        // )?;
        // inventory.add_phantom_sub_executor(
        //     phantom::Rv32PrintStrSubEx,
        //     PhantomDiscriminant(Rv32Phantom::PrintStr as u16),
        // )?;
        // inventory.add_phantom_sub_executor(
        //     phantom::Rv32HintLoadByKeySubEx,
        //     PhantomDiscriminant(Rv32Phantom::HintLoadByKey as u16),
        // )?;

        Ok(())
    }
}

impl<SC: StarkGenericConfig> VmCircuitExtension<SC> for Womir {
    fn extend_circuit(&self, inventory: &mut AirInventory<SC>) -> Result<(), AirInventoryError> {
        let SystemPort {
            execution_bus,
            program_bus,
            memory_bridge,
        } = inventory.system().port();

        let exec_bridge = ExecutionBridge::new(execution_bus, program_bus);
        let range_checker = inventory.range_checker().bus;
        let _pointer_max_bits = inventory.pointer_max_bits();

        let bitwise_lu = {
            // A trick to get around Rust's borrow rules
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
        //
        // let lt = Rv32LessThanAir::new(
        //     Rv32BaseAluAdapterAir::new(exec_bridge, memory_bridge, bitwise_lu),
        //     LessThanCoreAir::new(bitwise_lu, LessThanOpcode::CLASS_OFFSET),
        // );
        // inventory.add_air(lt);
        //
        // let shift = Rv32ShiftAir::new(
        //     Rv32BaseAluAdapterAir::new(exec_bridge, memory_bridge, bitwise_lu),
        //     ShiftCoreAir::new(bitwise_lu, range_checker, ShiftOpcode::CLASS_OFFSET),
        // );
        // inventory.add_air(shift);
        //
        // let load_store = Rv32LoadStoreAir::new(
        //     Rv32LoadStoreAdapterAir::new(
        //         memory_bridge,
        //         exec_bridge,
        //         range_checker,
        //         pointer_max_bits,
        //     ),
        //     LoadStoreCoreAir::new(Rv32LoadStoreOpcode::CLASS_OFFSET),
        // );
        // inventory.add_air(load_store);
        //
        // let load_sign_extend = Rv32LoadSignExtendAir::new(
        //     Rv32LoadStoreAdapterAir::new(
        //         memory_bridge,
        //         exec_bridge,
        //         range_checker,
        //         pointer_max_bits,
        //     ),
        //     LoadSignExtendCoreAir::new(range_checker),
        // );
        // inventory.add_air(load_sign_extend);
        //
        // let beq = Rv32BranchEqualAir::new(
        //     Rv32BranchAdapterAir::new(exec_bridge, memory_bridge),
        //     BranchEqualCoreAir::new(BranchEqualOpcode::CLASS_OFFSET, DEFAULT_PC_STEP),
        // );
        // inventory.add_air(beq);
        //
        // let blt = Rv32BranchLessThanAir::new(
        //     Rv32BranchAdapterAir::new(exec_bridge, memory_bridge),
        //     BranchLessThanCoreAir::new(bitwise_lu, BranchLessThanOpcode::CLASS_OFFSET),
        // );
        // inventory.add_air(blt);
        //
        // let jal_lui = Rv32JalLuiAir::new(
        //     Rv32CondRdWriteAdapterAir::new(Rv32RdWriteAdapterAir::new(memory_bridge, exec_bridge)),
        //     Rv32JalLuiCoreAir::new(bitwise_lu),
        // );
        // inventory.add_air(jal_lui);
        //
        // let jalr = Rv32JalrAir::new(
        //     Rv32JalrAdapterAir::new(memory_bridge, exec_bridge),
        //     Rv32JalrCoreAir::new(bitwise_lu, range_checker),
        // );
        // inventory.add_air(jalr);
        //
        // let auipc = Rv32AuipcAir::new(
        //     Rv32RdWriteAdapterAir::new(memory_bridge, exec_bridge),
        //     Rv32AuipcCoreAir::new(bitwise_lu),
        // );
        // inventory.add_air(auipc);

        Ok(())
    }
}

pub struct WomirCpuProverExt;
// This implementation is specific to CpuBackend because the lookup chips (VariableRangeChecker,
// BitwiseOperationLookupChip) are specific to CpuBackend.
impl<E, SC, RA> VmProverExtension<E, RA, Womir> for WomirCpuProverExt
where
    SC: StarkGenericConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    RA: RowMajorMatrixArena<Val<SC>>,
    Val<SC>: PrimeField32,
{
    fn extend_prover(
        &self,
        _: &Womir,
        inventory: &mut ChipInventory<SC, RA, CpuBackend<SC>>,
    ) -> Result<(), ChipInventoryError> {
        let range_checker = inventory.range_checker()?.clone();
        let timestamp_max_bits = inventory.timestamp_max_bits();
        let _pointer_max_bits = inventory.airs().pointer_max_bits();
        let mem_helper = SharedMemoryHelper::new(range_checker.clone(), timestamp_max_bits);

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

        // These calls to next_air are not strictly necessary to construct the chips, but provide a
        // safeguard to ensure that chip construction matches the circuit definition
        inventory.next_air::<Rv32BaseAluAir>()?;
        let base_alu = Rv32BaseAluChip::new(
            BaseAluFiller::new(
                Rv32BaseAluAdapterFiller::new(bitwise_lu.clone()),
                bitwise_lu.clone(),
                BaseAluOpcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(base_alu);
        //
        // inventory.next_air::<Rv32LessThanAir>()?;
        // let lt = Rv32LessThanChip::new(
        //     LessThanFiller::new(
        //         Rv32BaseAluAdapterFiller::new(bitwise_lu.clone()),
        //         bitwise_lu.clone(),
        //         LessThanOpcode::CLASS_OFFSET,
        //     ),
        //     mem_helper.clone(),
        // );
        // inventory.add_executor_chip(lt);
        //
        // inventory.next_air::<Rv32ShiftAir>()?;
        // let shift = Rv32ShiftChip::new(
        //     ShiftFiller::new(
        //         Rv32BaseAluAdapterFiller::new(bitwise_lu.clone()),
        //         bitwise_lu.clone(),
        //         range_checker.clone(),
        //         ShiftOpcode::CLASS_OFFSET,
        //     ),
        //     mem_helper.clone(),
        // );
        // inventory.add_executor_chip(shift);
        //
        // inventory.next_air::<Rv32LoadStoreAir>()?;
        // let load_store_chip = Rv32LoadStoreChip::new(
        //     LoadStoreFiller::new(
        //         Rv32LoadStoreAdapterFiller::new(pointer_max_bits, range_checker.clone()),
        //         Rv32LoadStoreOpcode::CLASS_OFFSET,
        //     ),
        //     mem_helper.clone(),
        // );
        // inventory.add_executor_chip(load_store_chip);
        //
        // inventory.next_air::<Rv32LoadSignExtendAir>()?;
        // let load_sign_extend = Rv32LoadSignExtendChip::new(
        //     LoadSignExtendFiller::new(
        //         Rv32LoadStoreAdapterFiller::new(pointer_max_bits, range_checker.clone()),
        //         range_checker.clone(),
        //     ),
        //     mem_helper.clone(),
        // );
        // inventory.add_executor_chip(load_sign_extend);
        //
        // inventory.next_air::<Rv32BranchEqualAir>()?;
        // let beq = Rv32BranchEqualChip::new(
        //     BranchEqualFiller::new(
        //         Rv32BranchAdapterFiller,
        //         BranchEqualOpcode::CLASS_OFFSET,
        //         DEFAULT_PC_STEP,
        //     ),
        //     mem_helper.clone(),
        // );
        // inventory.add_executor_chip(beq);
        //
        // inventory.next_air::<Rv32BranchLessThanAir>()?;
        // let blt = Rv32BranchLessThanChip::new(
        //     BranchLessThanFiller::new(
        //         Rv32BranchAdapterFiller,
        //         bitwise_lu.clone(),
        //         BranchLessThanOpcode::CLASS_OFFSET,
        //     ),
        //     mem_helper.clone(),
        // );
        // inventory.add_executor_chip(blt);
        //
        // inventory.next_air::<Rv32JalLuiAir>()?;
        // let jal_lui = Rv32JalLuiChip::new(
        //     Rv32JalLuiFiller::new(
        //         Rv32CondRdWriteAdapterFiller::new(Rv32RdWriteAdapterFiller),
        //         bitwise_lu.clone(),
        //     ),
        //     mem_helper.clone(),
        // );
        // inventory.add_executor_chip(jal_lui);
        //
        // inventory.next_air::<Rv32JalrAir>()?;
        // let jalr = Rv32JalrChip::new(
        //     Rv32JalrFiller::new(
        //         Rv32JalrAdapterFiller,
        //         bitwise_lu.clone(),
        //         range_checker.clone(),
        //     ),
        //     mem_helper.clone(),
        // );
        // inventory.add_executor_chip(jalr);
        //
        // inventory.next_air::<Rv32AuipcAir>()?;
        // let auipc = Rv32AuipcChip::new(
        //     Rv32AuipcFiller::new(Rv32RdWriteAdapterFiller, bitwise_lu.clone()),
        //     mem_helper.clone(),
        // );
        // inventory.add_executor_chip(auipc);

        Ok(())
    }
}
