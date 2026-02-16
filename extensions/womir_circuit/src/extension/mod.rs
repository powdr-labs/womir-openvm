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
use openvm_circuit_primitives::range_tuple::{
    RangeTupleCheckerAir, RangeTupleCheckerBus, RangeTupleCheckerChip, SharedRangeTupleCheckerChip,
};
use openvm_instructions::LocalOpcode;
use openvm_rv32im_circuit::{
    BaseAluCoreAir, BaseAluFiller, LessThanCoreAir, LessThanFiller, LoadSignExtendCoreAir,
    LoadSignExtendFiller, LoadStoreCoreAir, LoadStoreFiller, MultiplicationCoreAir,
    MultiplicationFiller,
};
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    engine::StarkEngine,
    p3_field::PrimeField32,
    prover::cpu::{CpuBackend, CpuDevice},
};
use openvm_womir_transpiler::{
    BaseAlu64Opcode, BaseAluOpcode, ConstOpcodes, JumpOpcode, LessThan64Opcode, LessThanOpcode,
    LoadStoreOpcode, Mul64Opcode, MulOpcode,
};
use serde::{Deserialize, Serialize};
use strum::IntoEnumIterator;

use openvm_circuit::arch::ExecutionBridge;

use crate::{
    adapters::*, const32::Const32Executor, load_sign_extend::execution::LoadSignExtendExecutor,
    loadstore::execution::LoadStoreExecutor, *,
};

pub use self::WomirCpuProverExt as WomirProverExt;

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
    BaseAlu(Rv32BaseAluExecutor),
    BaseAlu64(BaseAlu64Executor),
    Mul(Rv32MultiplicationExecutor),
    Mul64(Mul64Executor),
    LessThan(Rv32LessThanExecutor),
    LessThan64(LessThan64Executor),
    LoadStore(Rv32LoadStoreExecutor),
    LoadSignExtend(Rv32LoadSignExtendExecutor),
    Jump(JumpExecutor),
    Const32(Const32Executor),
}

// ============ VmExtension Implementations ============

impl<F: PrimeField32> VmExecutionExtension<F> for Womir {
    type Executor = WomirExecutor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<F, WomirExecutor>,
    ) -> Result<(), ExecutorInventoryError> {
        let pointer_max_bits = inventory.pointer_max_bits();

        let base_alu = Rv32BaseAluExecutor::new(
            Rv32BaseAluAdapterExecutor::default(),
            BaseAluOpcode::CLASS_OFFSET,
        );
        inventory.add_executor(base_alu, BaseAluOpcode::iter().map(|x| x.global_opcode()))?;

        let base_alu_64 = BaseAlu64Executor::new(
            BaseAluAdapterExecutor::<8, 2, RV32_CELL_BITS>::default(),
            BaseAlu64Opcode::CLASS_OFFSET,
        );
        inventory.add_executor(
            base_alu_64,
            BaseAlu64Opcode::iter().map(|x| x.global_opcode()),
        )?;

        let mul = Rv32MultiplicationExecutor::new(
            Rv32BaseAluAdapterExecutor::default(),
            MulOpcode::CLASS_OFFSET,
        );
        inventory.add_executor(mul, MulOpcode::iter().map(|x| x.global_opcode()))?;

        let mul_64 = Mul64Executor::new(
            BaseAluAdapterExecutor::<8, 2, RV32_CELL_BITS>::default(),
            Mul64Opcode::CLASS_OFFSET,
        );
        inventory.add_executor(mul_64, Mul64Opcode::iter().map(|x| x.global_opcode()))?;
        let less_than = Rv32LessThanExecutor::new(
            Rv32BaseAluAdapterExecutor::default(),
            LessThanOpcode::CLASS_OFFSET,
        );
        inventory.add_executor(less_than, LessThanOpcode::iter().map(|x| x.global_opcode()))?;

        let less_than_64 = LessThan64Executor::new(
            BaseAluAdapterExecutor::<8, 2, RV32_CELL_BITS>::default(),
            LessThan64Opcode::CLASS_OFFSET,
        );
        inventory.add_executor(
            less_than_64,
            LessThan64Opcode::iter().map(|x| x.global_opcode()),
        )?;

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

        let jump = JumpExecutor::new(JumpAdapterExecutor::default(), JumpOpcode::CLASS_OFFSET);
        inventory.add_executor(jump, JumpOpcode::iter().map(|x| x.global_opcode()))?;

        let const32 = Const32Executor::new(ConstOpcodes::CLASS_OFFSET);
        inventory.add_executor(const32, ConstOpcodes::iter().map(|x| x.global_opcode()))?;

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
        let pointer_max_bits = inventory.pointer_max_bits();

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

        let base_alu_64 = BaseAlu64Air::new(
            BaseAluAdapterAir::<8, 2>::new(exec_bridge, memory_bridge, bitwise_lu),
            BaseAluCoreAir::new(bitwise_lu, BaseAlu64Opcode::CLASS_OFFSET),
        );
        inventory.add_air(base_alu_64);

        let range_tuple_bus = {
            let existing_air = inventory.find_air::<RangeTupleCheckerAir<2>>().next();
            if let Some(air) = existing_air {
                air.bus
            } else {
                let bus = RangeTupleCheckerBus::new(
                    inventory.new_bus_idx(),
                    self.range_tuple_checker_sizes,
                );
                let air = RangeTupleCheckerAir { bus };
                inventory.add_air(air);
                bus
            }
        };

        let mul = Rv32MultiplicationAir::new(
            Rv32BaseAluAdapterAir::new(exec_bridge, memory_bridge, bitwise_lu),
            MultiplicationCoreAir::new(range_tuple_bus, MulOpcode::CLASS_OFFSET),
        );
        inventory.add_air(mul);

        let mul_64 = Mul64Air::new(
            BaseAluAdapterAir::<8, 2>::new(exec_bridge, memory_bridge, bitwise_lu),
            MultiplicationCoreAir::new(range_tuple_bus, Mul64Opcode::CLASS_OFFSET),
        );
        inventory.add_air(mul_64);
        let less_than = Rv32LessThanAir::new(
            Rv32BaseAluAdapterAir::new(exec_bridge, memory_bridge, bitwise_lu),
            LessThanCoreAir::new(bitwise_lu, LessThanOpcode::CLASS_OFFSET),
        );
        inventory.add_air(less_than);

        let less_than_64 = LessThan64Air::new(
            BaseAluAdapterAir::<8, 2>::new(exec_bridge, memory_bridge, bitwise_lu),
            LessThanCoreAir::new(bitwise_lu, LessThan64Opcode::CLASS_OFFSET),
        );
        inventory.add_air(less_than_64);

        let load_store = Rv32LoadStoreAir::new(
            Rv32LoadStoreAdapterAir::new(
                memory_bridge,
                exec_bridge,
                range_checker,
                pointer_max_bits,
            ),
            LoadStoreCoreAir::new(LoadStoreOpcode::CLASS_OFFSET),
        );
        inventory.add_air(load_store);

        let load_sign_extend = Rv32LoadSignExtendAir::new(
            Rv32LoadStoreAdapterAir::new(
                memory_bridge,
                exec_bridge,
                range_checker,
                pointer_max_bits,
            ),
            LoadSignExtendCoreAir::new(range_checker),
        );
        inventory.add_air(load_sign_extend);

        let jump = JumpAir::new(
            JumpAdapterAir::new(exec_bridge, memory_bridge),
            crate::jump::core::JumpCoreAir::new(JumpOpcode::CLASS_OFFSET),
        );
        inventory.add_air(jump);

        let const32 = Const32Air::new(
            bitwise_lu,
            ConstOpcodes::CLASS_OFFSET,
            exec_bridge,
            memory_bridge,
        );
        inventory.add_air(const32);

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
        let pointer_max_bits = inventory.airs().pointer_max_bits();
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

        inventory.next_air::<BaseAlu64Air>()?;
        let base_alu_64 = BaseAlu64Chip::new(
            BaseAluFiller::new(
                BaseAluAdapterFiller::<2, RV32_CELL_BITS>::new(bitwise_lu.clone()),
                bitwise_lu.clone(),
                BaseAlu64Opcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(base_alu_64);

        let range_tuple_chip = {
            let existing_chip = inventory
                .find_chip::<SharedRangeTupleCheckerChip<2>>()
                .next();
            if let Some(chip) = existing_chip {
                chip.clone()
            } else {
                let air: &RangeTupleCheckerAir<2> = inventory.next_air()?;
                let chip = Arc::new(RangeTupleCheckerChip::new(air.bus));
                inventory.add_periphery_chip(chip.clone());
                chip
            }
        };

        inventory.next_air::<Rv32MultiplicationAir>()?;
        let mul = Rv32MultiplicationChip::new(
            MultiplicationFiller::new(
                Rv32BaseAluAdapterFiller::new(bitwise_lu.clone()),
                range_tuple_chip.clone(),
                MulOpcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(mul);

        inventory.next_air::<Mul64Air>()?;
        let mul_64 = Mul64Chip::new(
            MultiplicationFiller::new(
                BaseAluAdapterFiller::<2, RV32_CELL_BITS>::new(bitwise_lu.clone()),
                range_tuple_chip.clone(),
                Mul64Opcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(mul_64);
        inventory.next_air::<Rv32LessThanAir>()?;
        let less_than = Rv32LessThanChip::new(
            LessThanFiller::new(
                Rv32BaseAluAdapterFiller::new(bitwise_lu.clone()),
                bitwise_lu.clone(),
                LessThanOpcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(less_than);

        inventory.next_air::<LessThan64Air>()?;
        let less_than_64 = LessThan64Chip::new(
            LessThanFiller::new(
                BaseAluAdapterFiller::<2, RV32_CELL_BITS>::new(bitwise_lu.clone()),
                bitwise_lu.clone(),
                LessThan64Opcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(less_than_64);

        inventory.next_air::<Rv32LoadStoreAir>()?;
        let load_store_chip = Rv32LoadStoreChip::new(
            LoadStoreFiller::new(
                Rv32LoadStoreAdapterFiller::new(pointer_max_bits, range_checker.clone()),
                LoadStoreOpcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(load_store_chip);

        inventory.next_air::<Rv32LoadSignExtendAir>()?;
        let load_sign_extend = Rv32LoadSignExtendChip::new(
            LoadSignExtendFiller::new(
                Rv32LoadStoreAdapterFiller::new(pointer_max_bits, range_checker.clone()),
                range_checker.clone(),
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(load_sign_extend);

        inventory.next_air::<JumpAir>()?;
        let jump = JumpChip::new(
            JumpFiller::new(
                JumpAdapterFiller::new(),
                crate::jump::core::JumpCoreFiller::new(JumpOpcode::CLASS_OFFSET),
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(jump);

        inventory.next_air::<Const32Air>()?;
        let const32 = Const32Chip::new(Const32Filler::new(bitwise_lu.clone()), mem_helper);
        inventory.add_executor_chip(const32);
        Ok(())
    }
}
