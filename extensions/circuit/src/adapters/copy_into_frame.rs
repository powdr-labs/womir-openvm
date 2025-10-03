use std::{borrow::Borrow, marker::PhantomData};

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
use openvm_instructions::{LocalOpcode, instruction::Instruction, program::DEFAULT_PC_STEP};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeField32},
    rap::ColumnsAir,
};
use openvm_womir_transpiler::CopyIntoFrameOpcode;
use serde::{Deserialize, Serialize};
use struct_reflection::{StructReflection, StructReflectionHelper};

use crate::{FrameBus, FrameState, VmAdapterChipWom, WomBridge, WomController, WomRecord};

use super::{RV32_REGISTER_NUM_LIMBS, compose, decompose};

#[derive(Debug)]
pub struct CopyIntoFrameAdapterChipWom<F: Field> {
    pub air: CopyIntoFrameAdapterAirWom,
    _marker: PhantomData<F>,
}

impl<F: PrimeField32> CopyIntoFrameAdapterChipWom<F> {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        frame_bus: FrameBus,
        memory_bridge: MemoryBridge,
        wom_bridge: WomBridge,
    ) -> Self {
        Self {
            air: CopyIntoFrameAdapterAirWom {
                _execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                _frame_bus: frame_bus,
                _memory_bridge: memory_bridge,
                wom_bridge,
            },
            _marker: PhantomData,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopyIntoFrameReadRecord<F> {
    pub rs1: Option<(WomRecord<F>, u32)>, // Value to copy
    pub rs2: Option<(WomRecord<F>, u32)>, // Frame pointer
}

#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopyIntoFrameWriteRecord<F> {
    pub from_state: ExecutionState<u32>,
    pub from_frame: FrameState<u32>,
    pub rd: u32,
    pub rd_rec: Option<WomRecord<F>>,
}

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct CopyIntoFrameAdapterColsWom<T> {
    pub from_state: ExecutionState<T>,
    pub from_frame: FrameState<T>,
    pub value_reg_ptr: T,     // rs1 pointer (register containing value to copy)
    pub frame_ptr_reg_ptr: T, // rs2 pointer (register containing frame pointer)
    pub destination_ptr: T,   // Where we write: frame_pointer + offset
    /// 0 if copy_from_frame
    /// 1 if copy_into_frame
    pub is_copy_into: T,
    pub write_mult: T,
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct CopyIntoFrameAdapterAirWom {
    pub(super) _memory_bridge: MemoryBridge,
    pub(super) wom_bridge: WomBridge,
    pub(super) _execution_bridge: ExecutionBridge,
    pub(super) _frame_bus: FrameBus,
}

impl<F: Field> BaseAir<F> for CopyIntoFrameAdapterAirWom {
    fn width(&self) -> usize {
        CopyIntoFrameAdapterColsWom::<F>::width()
    }
}

impl<F: Field> ColumnsAir<F> for CopyIntoFrameAdapterAirWom {
    fn columns(&self) -> Option<Vec<String>> {
        CopyIntoFrameAdapterColsWom::<F>::struct_reflection()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for CopyIntoFrameAdapterAirWom {
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
        let local: &CopyIntoFrameAdapterColsWom<_> = local.borrow();

        // read other fp
        self.wom_bridge
            .read(local.frame_ptr_reg_ptr, ctx.reads[1].clone())
            .eval(builder, ctx.instruction.is_valid.clone());

        // read src reg
        self.wom_bridge
            .read(local.value_reg_ptr, ctx.reads[0].clone())
            .eval(builder, ctx.instruction.is_valid.clone());

        // write dest reg
        self.wom_bridge
            .write(
                local.destination_ptr,
                ctx.writes[0].clone(),
                local.write_mult,
            )
            .eval(builder, ctx.instruction.is_valid.clone());
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &CopyIntoFrameAdapterColsWom<_> = local.borrow();
        cols.from_state.pc
    }
}

impl<F: PrimeField32> VmAdapterChipWom<F> for CopyIntoFrameAdapterChipWom<F> {
    type ReadRecord = CopyIntoFrameReadRecord<F>;
    type WriteRecord = CopyIntoFrameWriteRecord<F>;
    type Air = CopyIntoFrameAdapterAirWom;
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
        memory: &mut MemoryController<F>,
        wom: &mut WomController<F>,
        fp: u32,
        instruction: &Instruction<F>,
    ) -> Result<(
        <Self::Interface as VmAdapterInterface<F>>::Reads,
        Self::ReadRecord,
    )> {
        let Instruction { b, c, opcode, .. } = *instruction;

        // COPY_INTO_FRAME: target_reg (a), src_reg (b), target_fp (c)
        // COPY_FROM_FRAME: target_reg (a), src_reg (b), src_fp (c)

        let fp_f = F::from_canonical_u32(fp);

        let local_opcode = CopyIntoFrameOpcode::from_usize(
            opcode.local_opcode_idx(CopyIntoFrameOpcode::CLASS_OFFSET),
        );

        let other_fp = wom.read::<RV32_REGISTER_NUM_LIMBS>(c + fp_f);

        let other_fp_u32 = compose(other_fp.1);
        let other_fp_f = F::from_canonical_u32(other_fp_u32);

        memory.increment_timestamp();

        let value_to_copy = match local_opcode {
            CopyIntoFrameOpcode::COPY_INTO_FRAME => wom.read::<RV32_REGISTER_NUM_LIMBS>(b + fp_f),
            CopyIntoFrameOpcode::COPY_FROM_FRAME => {
                wom.read::<RV32_REGISTER_NUM_LIMBS>(b + other_fp_f)
            }
        };

        Ok((
            [value_to_copy.1, other_fp.1],
            CopyIntoFrameReadRecord {
                rs1: Some((value_to_copy.0, compose(value_to_copy.1))),
                rs2: Some((other_fp.0, compose(other_fp.1))),
            },
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
            a,
            f: enabled,
            opcode,
            ..
        } = *instruction;

        let mut write_result = None;

        let local_opcode = CopyIntoFrameOpcode::from_usize(
            opcode.local_opcode_idx(CopyIntoFrameOpcode::CLASS_OFFSET),
        );

        let fp_f = F::from_canonical_u32(from_frame.fp);

        if enabled != F::ZERO {
            let value = read_record.rs1.as_ref().unwrap().1;
            let other_fp = read_record.rs2.as_ref().unwrap().1;
            let other_fp_f = F::from_canonical_u32(other_fp);

            write_result = Some(match local_opcode {
                CopyIntoFrameOpcode::COPY_INTO_FRAME => wom.write(a + other_fp_f, decompose(value)),
                CopyIntoFrameOpcode::COPY_FROM_FRAME => wom.write(a + fp_f, decompose(value)),
            });
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
                rd: a.as_canonical_u32(),
                rd_rec: write_result,
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
