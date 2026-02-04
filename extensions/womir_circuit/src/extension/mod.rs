use std::sync::Arc;

use derive_more::derive::From;
use openvm_circuit::{
    arch::{
        AirInventory, AirInventoryError, ChipInventory, ChipInventoryError,
        ExecutorInventoryBuilder, ExecutorInventoryError, RowMajorMatrixArena, VmCircuitExtension,
        VmExecutionExtension, VmProverExtension,
    },
    system::{SystemPort, memory::SharedMemoryHelper},
};
use openvm_circuit_derive::{AnyEnum, PreflightExecutor};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
    SharedBitwiseOperationLookupChip,
};
use openvm_instructions::LocalOpcode;
use openvm_instructions::PhantomDiscriminant;
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    engine::StarkEngine,
    p3_field::PrimeField32,
    prover::cpu::{CpuBackend, CpuDevice},
};
use openvm_womir_transpiler::{
    BaseAlu64Opcode, BaseAluOpcode, ConstOpcodes, DivRemOpcode, Eq64Opcode, EqOpcode,
    HintStoreOpcode, LessThan64Opcode, LessThanOpcode, LoadStoreOpcode, MulOpcode, Phantom,
    Shift64Opcode, ShiftOpcode,
};
use serde::{Deserialize, Serialize};
use strum::IntoEnumIterator;

use crate::{
    adapters::*,
    base_alu::REGISTER_NUM_LIMBS_64,
    execution::{ExecutionBridge, FpBus},
    *,
};

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
    // 32-bit operations:
    BaseAlu(BaseAluExecutor),
    // 64-bit operations:
    BaseAlu64(BaseAlu64Executor),
    LessThan(LessThanExecutor32),
    LessThan64(LessThan64Executor),
    Eq(EqExecutor32),
    Eq64(EqExecutor64),
    Shift(ShiftExecutor32),
    Shift64(ShiftExecutor64),
    LoadStore(LoadStoreExecutor32),
    LoadSignExtend(Rv32LoadSignExtendExecutor),
    Multiplication(MultiplicationExecutor32),
    DivRem(DivRemExecutor32),
    Const32(Const32Executor32),
    HintStore(HintStoreExecutor32),
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
        let base_alu: BaseAluExecutor = crate::PreflightExecutorWrapperFp::new(
            BaseAluCoreExecutor::new(
                BaseAluAdapterExecutor::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>::new(),
                BaseAluOpcode::CLASS_OFFSET,
            ),
            fp.clone(),
        );
        inventory.add_executor(base_alu, BaseAluOpcode::iter().map(|x| x.global_opcode()))?;

        // 64-bit ALU operations
        let base_alu_64: BaseAlu64Executor = crate::PreflightExecutorWrapperFp::new(
            BaseAluCoreExecutor::new(
                BaseAluAdapterExecutor::<REGISTER_NUM_LIMBS_64, RV32_CELL_BITS>::new(),
                BaseAlu64Opcode::CLASS_OFFSET,
            ),
            fp.clone(),
        );
        inventory.add_executor(
            base_alu_64,
            BaseAlu64Opcode::iter().map(|x| x.global_opcode()),
        )?;

        let shift: ShiftExecutor32 = crate::PreflightExecutorWrapperFp::new(
            ShiftExecutor::new(BaseAluAdapterExecutor::new(), ShiftOpcode::CLASS_OFFSET),
            fp.clone(),
        );
        inventory.add_executor(shift, ShiftOpcode::iter().map(|x| x.global_opcode()))?;

        let shift_64: ShiftExecutor64 = crate::PreflightExecutorWrapperFp::new(
            ShiftExecutor::new(
                BaseAluAdapterExecutor::<REGISTER_NUM_LIMBS_64, RV32_CELL_BITS>::new(),
                Shift64Opcode::CLASS_OFFSET,
            ),
            fp.clone(),
        );
        inventory.add_executor(shift_64, Shift64Opcode::iter().map(|x| x.global_opcode()))?;

        let load_store: LoadStoreExecutor32 = crate::PreflightExecutorWrapperFp::new(
            LoadStoreExecutor::new(
                LoadStoreAdapterExecutor::new(pointer_max_bits),
                LoadStoreOpcode::CLASS_OFFSET,
            ),
            fp.clone(),
        );
        inventory.add_executor(
            load_store,
            LoadStoreOpcode::iter()
                .take(LoadStoreOpcode::STOREB as usize + 1)
                .map(|x| x.global_opcode()),
        )?;

        let load_sign_extend =
            LoadSignExtendExecutor::new(LoadStoreAdapterExecutor::new(pointer_max_bits));
        inventory.add_executor(
            load_sign_extend,
            [LoadStoreOpcode::LOADB, LoadStoreOpcode::LOADH].map(|x| x.global_opcode()),
        )?;

        let multiplication: MultiplicationExecutor32 = crate::PreflightExecutorWrapperFp::new(
            MultiplicationExecutor::new(BaseAluAdapterExecutor::new(), MulOpcode::CLASS_OFFSET),
            fp.clone(),
        );
        inventory.add_executor(multiplication, MulOpcode::iter().map(|x| x.global_opcode()))?;

        let divrem: DivRemExecutor32 = crate::PreflightExecutorWrapperFp::new(
            DivRemExecutor::new(BaseAluAdapterExecutor::new(), DivRemOpcode::CLASS_OFFSET),
            fp.clone(),
        );
        inventory.add_executor(divrem, DivRemOpcode::iter().map(|x| x.global_opcode()))?;

        let const32: Const32Executor32 = crate::PreflightExecutorWrapperFp::new(
            Const32Executor::new(ConstOpcodes::CLASS_OFFSET),
            fp.clone(),
        );
        inventory.add_executor(const32, ConstOpcodes::iter().map(|x| x.global_opcode()))?;

        let less_than: LessThanExecutor32 = crate::PreflightExecutorWrapperFp::new(
            LessThanCoreExecutor::new(LessThanOpcode::CLASS_OFFSET),
            fp.clone(),
        );
        inventory.add_executor(less_than, LessThanOpcode::iter().map(|x| x.global_opcode()))?;

        let less_than_64: LessThan64Executor = crate::PreflightExecutorWrapperFp::new(
            LessThanCoreExecutor::new(LessThan64Opcode::CLASS_OFFSET),
            fp.clone(),
        );
        inventory.add_executor(
            less_than_64,
            LessThan64Opcode::iter().map(|x| x.global_opcode()),
        )?;

        let eq: EqExecutor32 = crate::PreflightExecutorWrapperFp::new(
            EqCoreExecutor::new(EqOpcode::CLASS_OFFSET),
            fp.clone(),
        );
        inventory.add_executor(eq, EqOpcode::iter().map(|x| x.global_opcode()))?;

        let eq_64: EqExecutor64 = crate::PreflightExecutorWrapperFp::new(
            EqCoreExecutor::new(Eq64Opcode::CLASS_OFFSET),
            fp.clone(),
        );
        inventory.add_executor(eq_64, Eq64Opcode::iter().map(|x| x.global_opcode()))?;

        let hintstore: HintStoreExecutor32 = crate::PreflightExecutorWrapperFp::new(
            HintStoreCoreExecutor::new(pointer_max_bits, HintStoreOpcode::CLASS_OFFSET),
            fp.clone(),
        );
        inventory.add_executor(
            hintstore,
            HintStoreOpcode::iter().map(|x| x.global_opcode()),
        )?;

        // Register phantom sub-executors
        inventory.add_phantom_sub_executor(
            crate::phantom::HintInputSubEx,
            PhantomDiscriminant(Phantom::HintInput as u16),
        )?;
        inventory.add_phantom_sub_executor(
            crate::phantom::HintRandomSubEx,
            PhantomDiscriminant(Phantom::HintRandom as u16),
        )?;
        inventory.add_phantom_sub_executor(
            crate::phantom::PrintStrSubEx,
            PhantomDiscriminant(Phantom::PrintStr as u16),
        )?;
        inventory.add_phantom_sub_executor(
            crate::phantom::HintLoadByKeySubEx,
            PhantomDiscriminant(Phantom::HintLoadByKey as u16),
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

        // TODO: Use our execution bridge
        let fp_bus = FpBus::new(inventory.new_bus_idx());
        let exec_bridge = ExecutionBridge::new(execution_bus, fp_bus, program_bus);
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

        let base_alu = BaseAluAir::new(
            BaseAluAdapterAir::new(exec_bridge, memory_bridge, bitwise_lu),
            BaseAluCoreAir::new(bitwise_lu, BaseAluOpcode::CLASS_OFFSET),
        );
        inventory.add_air(base_alu);

        // 64-bit ALU AIR
        let base_alu_64 = BaseAlu64Air::new(
            BaseAluAdapterAir::new(exec_bridge, memory_bridge, bitwise_lu),
            BaseAluCoreAir::new(bitwise_lu, BaseAlu64Opcode::CLASS_OFFSET),
        );
        inventory.add_air(base_alu_64);

        // Multiplication AIR - re-uses BaseAluAdapter since it has the same I/O pattern
        let range_tuple_bus = openvm_circuit_primitives::range_tuple::RangeTupleCheckerBus::new(
            inventory.new_bus_idx(),
            self.range_tuple_checker_sizes,
        );
        let multiplication = MultiplicationAir::new(
            BaseAluAdapterAir::new(exec_bridge, memory_bridge, bitwise_lu),
            MultiplicationCoreAir::new(range_tuple_bus, MulOpcode::CLASS_OFFSET),
        );
        inventory.add_air(multiplication);

        // DivRem AIR - re-uses BaseAluAdapter since it has the same I/O pattern
        let divrem_range_tuple_bus =
            openvm_circuit_primitives::range_tuple::RangeTupleCheckerBus::new(
                inventory.new_bus_idx(),
                self.range_tuple_checker_sizes,
            );
        let divrem = DivRemAir::new(
            BaseAluAdapterAir::new(exec_bridge, memory_bridge, bitwise_lu),
            DivRemCoreAir::new(
                bitwise_lu,
                divrem_range_tuple_bus,
                DivRemOpcode::CLASS_OFFSET,
            ),
        );
        inventory.add_air(divrem);

        // Shift AIR - re-uses BaseAluAdapter since it has the same I/O pattern
        let shift = ShiftAir::new(
            BaseAluAdapterAir::new(exec_bridge, memory_bridge, bitwise_lu),
            ShiftCoreAir::new(bitwise_lu, range_checker, ShiftOpcode::CLASS_OFFSET),
        );
        inventory.add_air(shift);
        //
        // let lt = Rv32LessThanAir::new(
        //     BaseAluAdapterAir::new(exec_bridge, memory_bridge, bitwise_lu),
        //     LessThanCoreAir::new(bitwise_lu, LessThanOpcode::CLASS_OFFSET),
        // );
        // inventory.add_air(lt);
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
        inventory.next_air::<BaseAluAir>()?;
        let base_alu = BaseAluChip::new(
            BaseAluFiller::new(
                BaseAluAdapterFiller::new(bitwise_lu.clone()),
                bitwise_lu.clone(),
                BaseAluOpcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(base_alu);

        // 64-bit ALU chip
        inventory.next_air::<BaseAlu64Air>()?;
        let base_alu_64 = BaseAlu64Chip::new(
            BaseAluFiller::new(
                BaseAluAdapterFiller::new(bitwise_lu.clone()),
                bitwise_lu.clone(),
                BaseAlu64Opcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(base_alu_64);

        // Multiplication chip - re-uses BaseAluAdapterFiller since it has the same I/O pattern
        let multiplication_air: &MultiplicationAir = inventory.next_air()?;
        let range_tuple_chip = Arc::new(
            openvm_circuit_primitives::range_tuple::RangeTupleCheckerChip::new(
                multiplication_air.core.bus,
            ),
        );
        inventory.add_periphery_chip(range_tuple_chip.clone());
        let multiplication = MultiplicationChip::new(
            MultiplicationFiller::new(
                BaseAluAdapterFiller::new(bitwise_lu.clone()),
                range_tuple_chip,
                MulOpcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(multiplication);

        // DivRem chip - re-uses BaseAluAdapterFiller since it has the same I/O pattern
        let divrem_air: &DivRemAir = inventory.next_air()?;
        let divrem_range_tuple_chip = Arc::new(
            openvm_circuit_primitives::range_tuple::RangeTupleCheckerChip::new(
                divrem_air.core.range_tuple_bus,
            ),
        );
        inventory.add_periphery_chip(divrem_range_tuple_chip.clone());
        let divrem = DivRemChip::new(
            DivRemFiller::new(
                BaseAluAdapterFiller::new(bitwise_lu.clone()),
                bitwise_lu.clone(),
                divrem_range_tuple_chip,
                DivRemOpcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(divrem);

        // Shift chip - re-uses BaseAluAdapterFiller since it has the same I/O pattern
        inventory.next_air::<ShiftAir>()?;
        let shift = ShiftChip::new(
            ShiftFiller::new(
                BaseAluAdapterFiller::new(bitwise_lu.clone()),
                bitwise_lu.clone(),
                range_checker.clone(),
                ShiftOpcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(shift);
        //
        // inventory.next_air::<Rv32LessThanAir>()?;
        // let lt = Rv32LessThanChip::new(
        //     LessThanFiller::new(
        //         BaseAluAdapterFiller::new(bitwise_lu.clone()),
        //         bitwise_lu.clone(),
        //         LessThanOpcode::CLASS_OFFSET,
        //     ),
        //     mem_helper.clone(),
        // );
        // inventory.add_executor_chip(lt);
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
