use std::borrow::Borrow;

use openvm_circuit::{
    arch::{
        AdapterAirContext, AdapterRuntimeContext, BasicAdapterInterface, ExecutionBridge,
        ExecutionBus, ExecutionState, MinimalInstruction, Result, VmAdapterAir, VmAdapterInterface,
    },
    system::{
        memory::{MemoryController, OfflineMemory, offline_checker::MemoryBridge},
        program::ProgramBus,
    },
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, PrimeField32},
    rap::ColumnsAir,
};
use serde::{Deserialize, Serialize};
use struct_reflection::{StructReflection, StructReflectionHelper};

use crate::{
    FrameBus, FrameState, VmAdapterChipWom, WomBridge, WomController, WomRecord,
    adapters::{compose, decompose},
};

use super::RV32_REGISTER_NUM_LIMBS;

#[derive(Debug)]
pub struct AllocateFrameAdapterChipWom {
    pub air: AllocateFrameAdapterAirWom,
    next_fp: u32,
}

impl AllocateFrameAdapterChipWom {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        frame_bus: FrameBus,
        memory_bridge: MemoryBridge,
        wom_bridge: WomBridge,
    ) -> Self {
        Self {
            air: AllocateFrameAdapterAirWom {
                _execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                wom_bridge,
                _frame_bus: frame_bus,
                _memory_bridge: memory_bridge,
            },
            // Start from 8 because 0 and 4 are used by the startup code.
            next_fp: 8,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocateFrameReadRecord {
    pub allocated_ptr: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocateFrameWriteRecord<F> {
    pub from_state: ExecutionState<u32>,
    pub from_frame: FrameState<u32>,
    pub target_reg: u32,
    pub amount_imm: u32,
    pub rd_write: Option<WomRecord<F>>,
}

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct AllocateFrameAdapterColsWom<T> {
    pub from_state: ExecutionState<T>,
    pub from_frame: FrameState<T>,
    pub target_reg_ptr: T,  // Pointer to target register
    pub allocation_size: T, // amount_imm field from instruction (b)
    pub src_reg_ptr: T,     // if reading from register
    pub result: T,
    pub read_imm_or_reg: T,
    pub write_mult: T,
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct AllocateFrameAdapterAirWom {
    pub(super) _memory_bridge: MemoryBridge,
    pub(super) wom_bridge: WomBridge,
    pub(super) _execution_bridge: ExecutionBridge,
    pub(super) _frame_bus: FrameBus,
}

impl<F: Field> BaseAir<F> for AllocateFrameAdapterAirWom {
    fn width(&self) -> usize {
        AllocateFrameAdapterColsWom::<F>::width()
    }
}

impl<F: Field> ColumnsAir<F> for AllocateFrameAdapterAirWom {
    fn columns(&self) -> Option<Vec<String>> {
        AllocateFrameAdapterColsWom::<F>::struct_reflection()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for AllocateFrameAdapterAirWom {
    type Interface = BasicAdapterInterface<
        AB::Expr,
        MinimalInstruction<AB::Expr>,
        2,
        1,
        RV32_REGISTER_NUM_LIMBS,
        RV32_REGISTER_NUM_LIMBS,
    >;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local: &AllocateFrameAdapterColsWom<_> = local.borrow();

        // read amount bytes
        builder.assert_bool(local.read_imm_or_reg);
        builder
            .when(local.read_imm_or_reg)
            .assert_one(ctx.instruction.is_valid.clone());

        self.wom_bridge
            .read(local.src_reg_ptr, ctx.reads[1].clone())
            .eval(builder, local.read_imm_or_reg);

        // write fp
        self.wom_bridge
            .write(
                local.target_reg_ptr,
                ctx.writes[0].clone(),
                local.write_mult,
            )
            .eval(builder, ctx.instruction.is_valid.clone());
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &AllocateFrameAdapterColsWom<_> = local.borrow();
        cols.from_state.pc
    }
}

impl<F: PrimeField32> VmAdapterChipWom<F> for AllocateFrameAdapterChipWom {
    type ReadRecord = AllocateFrameReadRecord;
    type WriteRecord = AllocateFrameWriteRecord<F>;
    type Air = AllocateFrameAdapterAirWom;
    type Interface = BasicAdapterInterface<
        F,
        MinimalInstruction<F>,
        2,
        1,
        RV32_REGISTER_NUM_LIMBS,
        RV32_REGISTER_NUM_LIMBS,
    >;

    fn preprocess(
        &mut self,
        _memory: &mut MemoryController<F>,
        wom: &mut WomController<F>,
        fp: u32,
        instruction: &Instruction<F>,
    ) -> Result<(
        <Self::Interface as VmAdapterInterface<F>>::Reads,
        Self::ReadRecord,
    )> {
        let Instruction {
            b: amount_imm,
            c: amount_reg,
            d: use_reg,
            ..
        } = *instruction;

        let amount = if use_reg == F::ZERO {
            // If use_reg is zero, we use the immediate value
            amount_imm.as_canonical_u32()
        } else {
            // Otherwise, we read the value from the register
            let fp_f = F::from_canonical_u32(fp);
            let (_, reg_data) = wom.read::<RV32_REGISTER_NUM_LIMBS>(amount_reg + fp_f);
            compose(reg_data)
        };
        let amount_bytes = RV32_REGISTER_NUM_LIMBS as u32 * amount;

        let allocated_ptr = self.next_fp;

        self.next_fp += amount_bytes;

        // TODO: "next fp" is being handled as a register/mem read (i.e., a read in the `BasicAdapterInterface`), is this right?
        let allocated_ptr_f = decompose(self.next_fp);
        let amount_bytes = decompose(amount_bytes);

        Ok((
            [allocated_ptr_f, amount_bytes],
            AllocateFrameReadRecord { allocated_ptr },
        ))
    }

    fn postprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        wom: &mut WomController<F>,
        instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
        from_frame: FrameState<u32>,
        _output: AdapterRuntimeContext<F, Self::Interface>,
        read_record: &Self::ReadRecord,
    ) -> Result<(ExecutionState<u32>, u32, Self::WriteRecord)> {
        let Instruction {
            a: target_reg,
            b,
            f: enabled,
            ..
        } = *instruction;

        memory.increment_timestamp();

        let mut write_result = None;

        if enabled != F::ZERO {
            write_result = Some(wom.write(
                target_reg + F::from_canonical_u32(from_frame.fp),
                decompose(read_record.allocated_ptr),
            ));
        }

        Ok((
            ExecutionState {
                pc: from_state.pc + DEFAULT_PC_STEP,
                timestamp: memory.timestamp(),
            },
            from_frame.fp,
            Self::WriteRecord {
                from_state,
                from_frame,
                target_reg: target_reg.as_canonical_u32(),
                amount_imm: b.as_canonical_u32(),
                rd_write: write_result,
            },
        ))
    }

    fn generate_trace_row(
        &self,
        _row_slice: &mut [F],
        _read_record: Self::ReadRecord,
        _write_record: Self::WriteRecord,
        _memory: &OfflineMemory<F>,
    ) {
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
