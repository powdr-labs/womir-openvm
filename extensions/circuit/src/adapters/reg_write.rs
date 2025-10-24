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
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::ColumnsAir,
};
use openvm_womir_transpiler::RegWriteOpcode;
use serde::{Deserialize, Serialize};
use struct_reflection::{StructReflection, StructReflectionHelper};

use strum::IntoEnumIterator;

use crate::{
    FrameBridge, FrameBus, FrameState, VmAdapterChipWom, WomBridge, WomController, WomRecord,
};

use super::{RV32_REGISTER_NUM_LIMBS, decompose};

#[derive(Debug)]
pub struct RegWriteAdapterChipWom<F: Field> {
    pub air: RegWriteAdapterAirWom,
    _marker: PhantomData<F>,
}

impl<F: PrimeField32> RegWriteAdapterChipWom<F> {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        frame_bus: FrameBus,
        memory_bridge: MemoryBridge,
        wom_bridge: WomBridge,
    ) -> Self {
        Self {
            air: RegWriteAdapterAirWom {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                frame_bridge: FrameBridge::new(frame_bus),
                _memory_bridge: memory_bridge,
                wom_bridge,
            },
            _marker: PhantomData,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegWriteWriteRecord<F> {
    pub from_state: ExecutionState<u32>,
    pub from_frame: FrameState<u32>,
    pub rd: Option<WomRecord<F>>,
}

#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegWriteReadRecord<F> {
    pub val: [F; RV32_REGISTER_NUM_LIMBS],
}

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct RegWriteAdapterColsWom<T> {
    pub from_state: ExecutionState<T>,
    pub from_frame: FrameState<T>,
    pub target_reg: T,
    pub lo_or_from_reg: T,
    pub hi: T,
    pub val: [T; RV32_REGISTER_NUM_LIMBS],
    pub write_mult: T,

    pub is_opcode_const32: T,
    pub is_opcode_const_field: T,
    pub is_opcode_copy_reg: T,
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct RegWriteAdapterAirWom {
    pub(super) _memory_bridge: MemoryBridge,
    pub(super) wom_bridge: WomBridge,
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) frame_bridge: FrameBridge,
}

impl<F: Field> BaseAir<F> for RegWriteAdapterAirWom {
    fn width(&self) -> usize {
        RegWriteAdapterColsWom::<F>::width()
    }
}

impl<F: Field> ColumnsAir<F> for RegWriteAdapterAirWom {
    fn columns(&self) -> Option<Vec<String>> {
        RegWriteAdapterColsWom::<F>::struct_reflection()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for RegWriteAdapterAirWom {
    type Interface = BasicAdapterInterface<
        AB::Expr,
        MinimalInstruction<AB::Expr>,
        0,
        1,
        0,
        RV32_REGISTER_NUM_LIMBS,
    >;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        _ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local: &RegWriteAdapterColsWom<_> = local.borrow();

        // TODO: to follow OVM conventions, we should probably move opcode based constraints to the core chip

        let flags = [
            local.is_opcode_const32,
            local.is_opcode_const_field,
            local.is_opcode_copy_reg,
        ];

        let is_valid = flags.iter().fold(AB::Expr::ZERO, |acc, &flag| {
            builder.assert_bool(flag);
            acc + flag.into()
        });
        builder.assert_bool(is_valid.clone());

        let opcode_offset = AB::Expr::from_canonical_usize(RegWriteOpcode::CLASS_OFFSET);
        let expected_opcode =
            opcode_offset +
            flags.iter().zip(RegWriteOpcode::iter()).fold(
                AB::Expr::ZERO,
                |acc, (flag, local_opcode)| {
                    acc + (*flag).into() * AB::Expr::from_canonical_u8(local_opcode as u8)
                },
            );

        // TODO: constrain local.val based on the opcode

        ///////////////////////////////////////////////////

        self.wom_bridge
            .read(local.lo_or_from_reg, local.val)
            .eval(builder, local.is_opcode_copy_reg);

        self.wom_bridge
            .write(
                local.target_reg + local.from_frame.fp,
                local.val,
                local.write_mult,
            )
            .eval(builder, is_valid.clone());

        let timestamp_change = AB::Expr::ONE;

        self.execution_bridge
            .execute_and_increment_pc::<AB>(
                expected_opcode,
                [
                    local.target_reg.into(),
                    local.lo_or_from_reg.into(),
                    local.hi.into(),
                    AB::Expr::ZERO,
                    AB::Expr::ZERO,
                    // TODO: is this always one?
                    AB::Expr::ONE,
                ],
                local.from_state,
                timestamp_change.clone(),
            )
            .eval(builder, is_valid.clone());

        self.frame_bridge
            .keep_fp(local.from_frame, timestamp_change)
            .eval(builder, is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &RegWriteAdapterColsWom<_> = local.borrow();
        cols.from_state.pc
    }
}

impl<F: PrimeField32> VmAdapterChipWom<F> for RegWriteAdapterChipWom<F> {
    type ReadRecord = RegWriteReadRecord<F>;
    type WriteRecord = RegWriteWriteRecord<F>;
    type Air = RegWriteAdapterAirWom;
    type Interface = BasicAdapterInterface<
        F,
        MinimalInstruction<F>,
        0,
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
        let Instruction { opcode, b, c, .. } = *instruction;

        let local_opcode =
            RegWriteOpcode::from_usize(opcode.local_opcode_idx(RegWriteOpcode::CLASS_OFFSET));

        let fp_f = F::from_canonical_u32(fp);

        let imm_lo = b.as_canonical_u32();
        let imm_hi = c.as_canonical_u32();
        assert!(
            imm_lo < (1 << 16) && imm_hi < (1 << 16),
            "Immediate values out of range",
        );
        let imm = imm_hi << 16 | imm_lo;

        let val = match local_opcode {
            RegWriteOpcode::CONST32 => decompose(imm),
            RegWriteOpcode::CONST_FIELD => {
                assert!(imm < F::ORDER_U32);
                [F::from_canonical_u32(imm), F::ZERO, F::ZERO, F::ZERO]
            }
            RegWriteOpcode::COPY_REG => {
                let from_reg = b;
                wom.read(from_reg + fp_f).1
            }
        };

        Ok(([], RegWriteReadRecord { val }))
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
        let Instruction { a, f: enabled, .. } = *instruction;

        let mut write_result = None;

        // TODO: should this only happen if enabled?
        memory.increment_timestamp();

        let fp_f = F::from_canonical_u32(from_frame.fp);

        if enabled != F::ZERO {
            write_result = Some(wom.write(a + fp_f, read_record.val))
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
                rd: write_result,
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
