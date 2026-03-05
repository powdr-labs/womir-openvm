//! WOMIR extension for incremental GPU testing (CUDA only).
//!
//! This module contains types with "PreparingGpu" suffix that are temporary
//! scaffolding for testing GPU chips one at a time. Once all WOMIR chips have
//! GPU implementations, these types will be removed and the main WomirConfig
//! will be used for GPU proving.
//!
//! Currently includes: BaseAlu32, BaseAlu64, Shift32, Shift64, Mul32, Mul64, DivRem32, DivRem64,
//! LessThan32, LessThan64, Eq32, Eq64, LoadStore, LoadSignExtend, Call, Const32, Jump

use std::sync::Arc;

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
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupAir, BitwiseOperationLookupBus},
    range_tuple::{RangeTupleCheckerAir, RangeTupleCheckerBus, RangeTupleCheckerChipGPU},
};
use openvm_cuda_backend::{engine::GpuBabyBearPoseidon2Engine, prover_backend::GpuBackend};
use openvm_instructions::LocalOpcode;
use openvm_rv32im_circuit::{
    BaseAluCoreAir, DivRemCoreAir, LessThanCoreAir, LoadSignExtendCoreAir, LoadStoreCoreAir,
    MultiplicationCoreAir, ShiftCoreAir,
};
use openvm_stark_backend::{config::StarkGenericConfig, p3_field::PrimeField32};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use openvm_womir_transpiler::{
    BaseAlu64Opcode, BaseAluOpcode, CallOpcode, ConstOpcodes, DivRem64Opcode, DivRemOpcode,
    Eq64Opcode, EqOpcode, JumpOpcode, LessThan64Opcode, LessThanOpcode, LoadStoreOpcode,
    Mul64Opcode, MulOpcode, Shift64Opcode, ShiftOpcode,
};
use serde::{Deserialize, Serialize};
use strum::IntoEnumIterator;

use openvm_circuit::arch::ExecutionBridge;

use crate::{
    BaseAlu64Air, BaseAlu64ChipGpu, BaseAlu64Executor, CallAir, CallChipGpu, CallCoreAir,
    Const32Air, Const32ChipGpu, Const32Executor, DivRem64Air, DivRem64ChipGpu, DivRem64Executor,
    Eq64Air, Eq64ChipGpu, Eq64Executor, EqCoreAir, JumpAir, JumpChipGpu, JumpExecutor,
    LessThan64Air, LessThan64ChipGpu, LessThan64Executor, Mul64Air, Mul64ChipGpu, Mul64Executor,
    Rv32BaseAluAir, Rv32BaseAluChipGpu, Rv32BaseAluExecutor, Rv32CallExecutor, Rv32DivRemAir,
    Rv32DivRemChipGpu, Rv32DivRemExecutor, Rv32EqAir, Rv32EqChipGpu, Rv32EqExecutor,
    Rv32LessThanAir, Rv32LessThanChipGpu, Rv32LessThanExecutor, Rv32LoadSignExtendAir,
    Rv32LoadSignExtendChipGpu, Rv32LoadSignExtendExecutor, Rv32LoadStoreAir, Rv32LoadStoreChipGpu,
    Rv32LoadStoreExecutor, Rv32MultiplicationAir, Rv32MultiplicationChipGpu,
    Rv32MultiplicationExecutor, Rv32ShiftAir, Rv32ShiftChipGpu, Rv32ShiftExecutor, Shift64Air,
    Shift64ChipGpu, Shift64Executor,
    adapters::{
        BaseAluAdapterAir, BaseAluAdapterAirDifferentInputsOutputs, JumpAdapterAir,
        JumpAdapterExecutor, Rv32BaseAluAdapterAir, Rv32LoadStoreAdapterAir, W32_REG_OPS,
        W64_NUM_LIMBS, W64_REG_OPS,
        call::{CallAdapterAir, CallAdapterExecutor},
    },
};

// ============ Extension Struct ============

/// WOMIR extension for incremental GPU testing.
/// Will be removed once all chips have GPU implementations.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct WomirPreparingGpu {
    #[serde(default = "super::default_range_tuple_checker_sizes")]
    pub range_tuple_checker_sizes: [u32; 2],
}

impl Default for WomirPreparingGpu {
    fn default() -> Self {
        Self {
            range_tuple_checker_sizes: super::default_range_tuple_checker_sizes(),
        }
    }
}

// ============ Executor Enum ============

/// Executor enum for WomirPreparingGpu extension.
#[derive(Clone, From, AnyEnum, Executor, MeteredExecutor, PreflightExecutor)]
pub enum WomirPreparingGpuExecutor {
    BaseAlu(Rv32BaseAluExecutor),
    BaseAlu64(BaseAlu64Executor),
    Mul(Rv32MultiplicationExecutor),
    Mul64(Mul64Executor),
    Shift(Rv32ShiftExecutor),
    Shift64(Shift64Executor),
    LessThan(Rv32LessThanExecutor),
    LessThan64(LessThan64Executor),
    DivRem(Rv32DivRemExecutor),
    DivRem64(DivRem64Executor),
    Eq(Rv32EqExecutor),
    Eq64(Eq64Executor),
    LoadStore(Rv32LoadStoreExecutor),
    LoadSignExtend(Rv32LoadSignExtendExecutor),
    Call(Rv32CallExecutor),
    Const32(Const32Executor),
    Jump(JumpExecutor),
}

// ============ VmExtension Implementations ============

impl<F: PrimeField32> VmExecutionExtension<F> for WomirPreparingGpu {
    type Executor = WomirPreparingGpuExecutor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<F, WomirPreparingGpuExecutor>,
    ) -> Result<(), ExecutorInventoryError> {
        use crate::adapters::{
            BaseAluAdapterExecutor, BaseAluAdapterExecutorDifferentInputsOutputs,
            Rv32BaseAluAdapterExecutor, Rv32LoadStoreAdapterExecutor,
        };

        let pointer_max_bits = inventory.pointer_max_bits();

        let base_alu = Rv32BaseAluExecutor::new(
            Rv32BaseAluAdapterExecutor::default(),
            BaseAluOpcode::CLASS_OFFSET,
        );
        inventory.add_executor(base_alu, BaseAluOpcode::iter().map(|x| x.global_opcode()))?;

        let base_alu_64 = BaseAlu64Executor::new(
            BaseAluAdapterExecutor::default(),
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

        let mul_64 =
            Mul64Executor::new(BaseAluAdapterExecutor::default(), Mul64Opcode::CLASS_OFFSET);
        inventory.add_executor(mul_64, Mul64Opcode::iter().map(|x| x.global_opcode()))?;

        let shift = Rv32ShiftExecutor::new(
            Rv32BaseAluAdapterExecutor::default(),
            ShiftOpcode::CLASS_OFFSET,
        );
        inventory.add_executor(shift, ShiftOpcode::iter().map(|x| x.global_opcode()))?;

        let shift_64 = Shift64Executor::new(
            BaseAluAdapterExecutor::default(),
            Shift64Opcode::CLASS_OFFSET,
        );
        inventory.add_executor(shift_64, Shift64Opcode::iter().map(|x| x.global_opcode()))?;

        let less_than = Rv32LessThanExecutor::new(
            Rv32BaseAluAdapterExecutor::default(),
            LessThanOpcode::CLASS_OFFSET,
        );
        inventory.add_executor(less_than, LessThanOpcode::iter().map(|x| x.global_opcode()))?;

        let less_than_64 = LessThan64Executor::new(
            BaseAluAdapterExecutorDifferentInputsOutputs::default(),
            LessThan64Opcode::CLASS_OFFSET,
        );
        inventory.add_executor(
            less_than_64,
            LessThan64Opcode::iter().map(|x| x.global_opcode()),
        )?;

        let divrem = Rv32DivRemExecutor::new(
            Rv32BaseAluAdapterExecutor::default(),
            DivRemOpcode::CLASS_OFFSET,
        );
        inventory.add_executor(divrem, DivRemOpcode::iter().map(|x| x.global_opcode()))?;

        let divrem_64 = DivRem64Executor::new(
            BaseAluAdapterExecutor::default(),
            DivRem64Opcode::CLASS_OFFSET,
        );
        inventory.add_executor(divrem_64, DivRem64Opcode::iter().map(|x| x.global_opcode()))?;

        let eq = Rv32EqExecutor::new(
            Rv32BaseAluAdapterExecutor::default(),
            EqOpcode::CLASS_OFFSET,
        );
        inventory.add_executor(eq, EqOpcode::iter().map(|x| x.global_opcode()))?;

        let eq_64 = Eq64Executor::new(
            BaseAluAdapterExecutorDifferentInputsOutputs::default(),
            Eq64Opcode::CLASS_OFFSET,
        );
        inventory.add_executor(eq_64, Eq64Opcode::iter().map(|x| x.global_opcode()))?;

        let load_store = Rv32LoadStoreExecutor::new(
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
            Rv32LoadSignExtendExecutor::new(Rv32LoadStoreAdapterExecutor::new(pointer_max_bits));
        inventory.add_executor(
            load_sign_extend,
            [LoadStoreOpcode::LOADB, LoadStoreOpcode::LOADH].map(|x| x.global_opcode()),
        )?;

        let call = Rv32CallExecutor::new(CallAdapterExecutor, CallOpcode::CLASS_OFFSET);
        inventory.add_executor(call, CallOpcode::iter().map(|x| x.global_opcode()))?;

        let const32 = Const32Executor::new(ConstOpcodes::CLASS_OFFSET);
        inventory.add_executor(const32, ConstOpcodes::iter().map(|x| x.global_opcode()))?;

        let jump = JumpExecutor::new(JumpAdapterExecutor::default(), JumpOpcode::CLASS_OFFSET);
        inventory.add_executor(jump, JumpOpcode::iter().map(|x| x.global_opcode()))?;

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
        let range_checker = inventory.range_checker().bus;
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

        let base_alu = Rv32BaseAluAir::new(
            Rv32BaseAluAdapterAir::new(exec_bridge, memory_bridge, bitwise_lu),
            BaseAluCoreAir::new(bitwise_lu, BaseAluOpcode::CLASS_OFFSET),
        );
        inventory.add_air(base_alu);

        let base_alu_64 = BaseAlu64Air::new(
            BaseAluAdapterAir::<W64_NUM_LIMBS, W64_REG_OPS>::new(
                exec_bridge,
                memory_bridge,
                bitwise_lu,
            ),
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
            BaseAluAdapterAir::<W64_NUM_LIMBS, W64_REG_OPS>::new(
                exec_bridge,
                memory_bridge,
                bitwise_lu,
            ),
            MultiplicationCoreAir::new(range_tuple_bus, Mul64Opcode::CLASS_OFFSET),
        );
        inventory.add_air(mul_64);

        let shift = Rv32ShiftAir::new(
            Rv32BaseAluAdapterAir::new(exec_bridge, memory_bridge, bitwise_lu),
            ShiftCoreAir::new(bitwise_lu, range_checker, ShiftOpcode::CLASS_OFFSET),
        );
        inventory.add_air(shift);

        let shift_64 = Shift64Air::new(
            BaseAluAdapterAir::<W64_NUM_LIMBS, W64_REG_OPS>::new(
                exec_bridge,
                memory_bridge,
                bitwise_lu,
            ),
            ShiftCoreAir::new(bitwise_lu, range_checker, Shift64Opcode::CLASS_OFFSET),
        );
        inventory.add_air(shift_64);

        let less_than = Rv32LessThanAir::new(
            Rv32BaseAluAdapterAir::new(exec_bridge, memory_bridge, bitwise_lu),
            LessThanCoreAir::new(bitwise_lu, LessThanOpcode::CLASS_OFFSET),
        );
        inventory.add_air(less_than);

        let less_than_64 = LessThan64Air::new(
            BaseAluAdapterAirDifferentInputsOutputs::<W64_NUM_LIMBS, W64_REG_OPS, W32_REG_OPS>::new(
                exec_bridge,
                memory_bridge,
                bitwise_lu,
            ),
            LessThanCoreAir::new(bitwise_lu, LessThan64Opcode::CLASS_OFFSET),
        );
        inventory.add_air(less_than_64);

        let divrem = Rv32DivRemAir::new(
            Rv32BaseAluAdapterAir::new(exec_bridge, memory_bridge, bitwise_lu),
            DivRemCoreAir::new(bitwise_lu, range_tuple_bus, DivRemOpcode::CLASS_OFFSET),
        );
        inventory.add_air(divrem);

        let divrem_64 = DivRem64Air::new(
            BaseAluAdapterAir::<W64_NUM_LIMBS, W64_REG_OPS>::new(
                exec_bridge,
                memory_bridge,
                bitwise_lu,
            ),
            DivRemCoreAir::new(bitwise_lu, range_tuple_bus, DivRem64Opcode::CLASS_OFFSET),
        );
        inventory.add_air(divrem_64);

        let eq = Rv32EqAir::new(
            Rv32BaseAluAdapterAir::new(exec_bridge, memory_bridge, bitwise_lu),
            EqCoreAir::new(EqOpcode::CLASS_OFFSET),
        );
        inventory.add_air(eq);

        let eq_64 = Eq64Air::new(
            BaseAluAdapterAirDifferentInputsOutputs::new(exec_bridge, memory_bridge, bitwise_lu),
            EqCoreAir::new(Eq64Opcode::CLASS_OFFSET),
        );
        inventory.add_air(eq_64);

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

        let call = CallAir::new(
            CallAdapterAir::new(exec_bridge, memory_bridge, range_checker, pointer_max_bits),
            CallCoreAir::new(CallOpcode::CLASS_OFFSET),
        );
        inventory.add_air(call);

        let const32 = Const32Air::new(
            bitwise_lu,
            ConstOpcodes::CLASS_OFFSET,
            exec_bridge,
            memory_bridge,
        );
        inventory.add_air(const32);

        let jump = JumpAir::new(
            JumpAdapterAir::new(exec_bridge, memory_bridge),
            crate::jump::core::JumpCoreAir::new(JumpOpcode::CLASS_OFFSET),
        );
        inventory.add_air(jump);

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
        extension: &WomirPreparingGpu,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let timestamp_max_bits = inventory.timestamp_max_bits();
        let pointer_max_bits = inventory.airs().pointer_max_bits();

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

        let range_tuple_checker = {
            let existing_chip = inventory
                .find_chip::<Arc<RangeTupleCheckerChipGPU<2>>>()
                .find(|c| {
                    c.sizes[0] >= extension.range_tuple_checker_sizes[0]
                        && c.sizes[1] >= extension.range_tuple_checker_sizes[1]
                });
            if let Some(chip) = existing_chip {
                chip.clone()
            } else {
                inventory.next_air::<RangeTupleCheckerAir<2>>()?;
                let chip = Arc::new(RangeTupleCheckerChipGPU::new(
                    extension.range_tuple_checker_sizes,
                ));
                inventory.add_periphery_chip(chip.clone());
                chip
            }
        };

        inventory.next_air::<Rv32MultiplicationAir>()?;
        let mul = Rv32MultiplicationChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            range_tuple_checker.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(mul);

        inventory.next_air::<Mul64Air>()?;
        let mul_64 = Mul64ChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            range_tuple_checker.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(mul_64);

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

        inventory.next_air::<Rv32LessThanAir>()?;
        let less_than = Rv32LessThanChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(less_than);

        inventory.next_air::<LessThan64Air>()?;
        let less_than_64 = LessThan64ChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(less_than_64);

        inventory.next_air::<Rv32DivRemAir>()?;
        let divrem = Rv32DivRemChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            range_tuple_checker.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(divrem);

        inventory.next_air::<DivRem64Air>()?;
        let divrem_64 = DivRem64ChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            range_tuple_checker.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(divrem_64);

        inventory.next_air::<Rv32EqAir>()?;
        let eq = Rv32EqChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(eq);

        inventory.next_air::<Eq64Air>()?;
        let eq_64 = Eq64ChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(eq_64);

        inventory.next_air::<Rv32LoadStoreAir>()?;
        let load_store =
            Rv32LoadStoreChipGpu::new(range_checker.clone(), pointer_max_bits, timestamp_max_bits);
        inventory.add_executor_chip(load_store);

        inventory.next_air::<Rv32LoadSignExtendAir>()?;
        let load_sign_extend = Rv32LoadSignExtendChipGpu::new(
            range_checker.clone(),
            pointer_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(load_sign_extend);

        inventory.next_air::<CallAir>()?;
        let call = CallChipGpu::new(range_checker.clone(), pointer_max_bits, timestamp_max_bits);
        inventory.add_executor_chip(call);

        inventory.next_air::<Const32Air>()?;
        let const32 = Const32ChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(const32);

        inventory.next_air::<JumpAir>()?;
        let jump = JumpChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(jump);

        Ok(())
    }
}
