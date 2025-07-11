use std::{
    borrow::{Borrow, BorrowMut},
    marker::PhantomData,
};

use openvm_circuit::{
    arch::{
        AdapterAirContext, BasicAdapterInterface, ExecutionBridge, ExecutionBus, ExecutionState,
        MinimalInstruction, Result as ResultVm, VmAdapterAir, VmAdapterInterface,
    },
    system::{
        memory::{
            offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
            MemoryAddress, MemoryController, OfflineMemory, RecordId,
        },
        program::ProgramBus,
    },
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    utils::not,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_IMM_AS, RV32_REGISTER_AS},
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::ColumnsAir,
};
use serde::{Deserialize, Serialize};
use struct_reflection::{StructReflection, StructReflectionHelper};

use crate::{AdapterRuntimeContextWom, FrameBridge, FrameBus, FrameState, VmAdapterChipWom};

use super::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};

/// Reads instructions of the form OP a, b, c, d, e where \[a:4\]_d = \[b:4\]_d op \[c:4\]_e.
/// Operand d can only be 1, and e can be either 1 (for register reads) or 0 (when c
/// is an immediate).
pub struct WomBaseAluAdapterChip<F: Field> {
    pub air: WomBaseAluAdapterAir,
    bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField32> WomBaseAluAdapterChip<F> {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        frame_bus: FrameBus,
        memory_bridge: MemoryBridge,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    ) -> Self {
        Self {
            air: WomBaseAluAdapterAir {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                frame_bridge: FrameBridge::new(frame_bus),
                memory_bridge,
                bitwise_lookup_bus: bitwise_lookup_chip.bus(),
            },
            bitwise_lookup_chip,
            _marker: PhantomData,
        }
    }
}

#[repr(C)]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "F: Field")]
pub struct WomBaseAluReadRecord<F: Field> {
    /// Read register value from address space d=1
    pub rs1: RecordId,
    /// Either
    /// - read rs2 register value or
    /// - if `rs2_is_imm` is true, this is None
    pub rs2: Option<RecordId>,
    /// immediate value of rs2 or 0
    pub rs2_imm: F,
}

#[repr(C)]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "F: Field")]
pub struct WomBaseAluWriteRecord<F: Field> {
    pub from_state: ExecutionState<u32>,
    pub from_frame: FrameState<u32>,
    /// Write to destination register
    pub rd: (RecordId, [F; 4]),
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct WomBaseAluAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub from_frame: FrameState<T>,
    pub rd_ptr: T,
    pub rs1_ptr: T,
    /// Pointer if rs2 was a read, immediate value otherwise
    pub rs2: T,
    /// 1 if rs2 was a read, 0 if an immediate
    pub rs2_as: T,
    pub reads_aux: [MemoryReadAuxCols<T>; 2],
    pub writes_aux: MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LIMBS>,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct WomBaseAluAdapterAir {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) frame_bridge: FrameBridge,
    pub(super) memory_bridge: MemoryBridge,
    bitwise_lookup_bus: BitwiseOperationLookupBus,
}

impl<F: Field> BaseAir<F> for WomBaseAluAdapterAir {
    fn width(&self) -> usize {
        WomBaseAluAdapterCols::<F>::width()
    }
}

impl<F: Field> ColumnsAir<F> for WomBaseAluAdapterAir {
    fn columns(&self) -> Option<Vec<String>> {
        WomBaseAluAdapterCols::<F>::struct_reflection()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for WomBaseAluAdapterAir {
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
        let local: &WomBaseAluAdapterCols<_> = local.borrow();
        let timestamp = local.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_canonical_usize(timestamp_delta - 1)
        };

        // If rs2 is an immediate value, constrain that:
        // 1. It's a 16-bit two's complement integer (stored in rs2_limbs[0] and rs2_limbs[1])
        // 2. It's properly sign-extended to 32-bits (the upper limbs must match the sign bit)
        let rs2_limbs = ctx.reads[1].clone();
        let rs2_sign = rs2_limbs[2].clone();
        let rs2_imm = rs2_limbs[0].clone()
            + rs2_limbs[1].clone() * AB::Expr::from_canonical_usize(1 << RV32_CELL_BITS)
            + rs2_sign.clone() * AB::Expr::from_canonical_usize(1 << (2 * RV32_CELL_BITS));
        builder.assert_bool(local.rs2_as);
        let mut rs2_imm_when = builder.when(not(local.rs2_as));
        rs2_imm_when.assert_eq(local.rs2, rs2_imm);
        rs2_imm_when.assert_eq(rs2_sign.clone(), rs2_limbs[3].clone());
        rs2_imm_when.assert_zero(
            rs2_sign.clone()
                * (AB::Expr::from_canonical_usize((1 << RV32_CELL_BITS) - 1) - rs2_sign),
        );
        self.bitwise_lookup_bus
            .send_range(rs2_limbs[0].clone(), rs2_limbs[1].clone())
            .eval(builder, ctx.instruction.is_valid.clone() - local.rs2_as);

        self.memory_bridge
            .read(
                MemoryAddress::new(AB::F::from_canonical_u32(RV32_REGISTER_AS), local.rs1_ptr),
                ctx.reads[0].clone(),
                timestamp_pp(),
                &local.reads_aux[0],
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        // This constraint ensures that the following memory read only occurs when `is_valid == 1`.
        builder
            .when(local.rs2_as)
            .assert_one(ctx.instruction.is_valid.clone());
        self.memory_bridge
            .read(
                MemoryAddress::new(local.rs2_as, local.rs2),
                ctx.reads[1].clone(),
                timestamp_pp(),
                &local.reads_aux[1],
            )
            .eval(builder, local.rs2_as);

        self.memory_bridge
            .write(
                MemoryAddress::new(AB::F::from_canonical_u32(RV32_REGISTER_AS), local.rd_ptr),
                ctx.writes[0].clone(),
                timestamp_pp(),
                &local.writes_aux,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.execution_bridge
            .execute_and_increment_or_set_pc(
                ctx.instruction.opcode,
                [
                    local.rd_ptr.into(),
                    local.rs1_ptr.into(),
                    local.rs2.into(),
                    AB::Expr::from_canonical_u32(RV32_REGISTER_AS),
                    local.rs2_as.into(),
                ],
                local.from_state,
                AB::F::from_canonical_usize(timestamp_delta),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &WomBaseAluAdapterCols<_> = local.borrow();
        cols.from_state.pc
    }
}

impl<F: PrimeField32> VmAdapterChipWom<F> for WomBaseAluAdapterChip<F> {
    type ReadRecord = WomBaseAluReadRecord<F>;
    type WriteRecord = WomBaseAluWriteRecord<F>;
    type Air = WomBaseAluAdapterAir;
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
        fp: u32,
        instruction: &Instruction<F>,
    ) -> ResultVm<(
        <Self::Interface as VmAdapterInterface<F>>::Reads,
        Self::ReadRecord,
    )> {
        let Instruction { b, c, d, e, .. } = *instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        debug_assert!(
            e.as_canonical_u32() == RV32_IMM_AS || e.as_canonical_u32() == RV32_REGISTER_AS
        );

        let fp_f = F::from_canonical_u32(fp);
        let rs1 = memory.read::<RV32_REGISTER_NUM_LIMBS>(d, b + fp_f);
        let (rs2, rs2_data, rs2_imm) = if e.is_zero() {
            let c_u32 = c.as_canonical_u32();
            debug_assert_eq!(c_u32 >> 24, 0);
            memory.increment_timestamp();
            (
                None,
                [
                    c_u32 as u8,
                    (c_u32 >> 8) as u8,
                    (c_u32 >> 16) as u8,
                    (c_u32 >> 16) as u8,
                ]
                .map(F::from_canonical_u8),
                c,
            )
        } else {
            let rs2_read = memory.read::<RV32_REGISTER_NUM_LIMBS>(e, c + fp_f);
            (Some(rs2_read.0), rs2_read.1, F::ZERO)
        };

        Ok((
            [rs1.1, rs2_data],
            Self::ReadRecord {
                rs1: rs1.0,
                rs2,
                rs2_imm,
            },
        ))
    }

    fn postprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
        from_frame: FrameState<u32>,
        output: AdapterRuntimeContextWom<F, Self::Interface>,
        _read_record: &Self::ReadRecord,
    ) -> ResultVm<(ExecutionState<u32>, u32, Self::WriteRecord)> {
        let Instruction { a, d, .. } = instruction;
        let fp_f = F::from_canonical_u32(from_frame.fp);
        let rd = memory.write(*d, *a + fp_f, output.writes[0]);

        let timestamp_delta = memory.timestamp() - from_state.timestamp;
        debug_assert!(
            timestamp_delta == 3,
            "timestamp delta is {timestamp_delta}, expected 3"
        );

        Ok((
            ExecutionState {
                pc: from_state.pc + DEFAULT_PC_STEP,
                timestamp: memory.timestamp(),
            },
            from_frame.fp,
            Self::WriteRecord {
                from_state,
                from_frame,
                rd,
            },
        ))
    }

    fn generate_trace_row(
        &self,
        row_slice: &mut [F],
        read_record: Self::ReadRecord,
        write_record: Self::WriteRecord,
        memory: &OfflineMemory<F>,
    ) {
        let row_slice: &mut WomBaseAluAdapterCols<_> = row_slice.borrow_mut();
        let aux_cols_factory = memory.aux_cols_factory();

        let rd = memory.record_by_id(write_record.rd.0);
        row_slice.from_state = write_record.from_state.map(F::from_canonical_u32);
        row_slice.rd_ptr = rd.pointer;

        let rs1 = memory.record_by_id(read_record.rs1);
        let rs2 = read_record.rs2.map(|rs2| memory.record_by_id(rs2));
        row_slice.rs1_ptr = rs1.pointer;

        if let Some(rs2) = rs2 {
            row_slice.rs2 = rs2.pointer;
            row_slice.rs2_as = rs2.address_space;
            aux_cols_factory.generate_read_aux(rs1, &mut row_slice.reads_aux[0]);
            aux_cols_factory.generate_read_aux(rs2, &mut row_slice.reads_aux[1]);
        } else {
            row_slice.rs2 = read_record.rs2_imm;
            row_slice.rs2_as = F::ZERO;
            let rs2_imm = row_slice.rs2.as_canonical_u32();
            let mask = (1 << RV32_CELL_BITS) - 1;
            self.bitwise_lookup_chip
                .request_range(rs2_imm & mask, (rs2_imm >> 8) & mask);
            aux_cols_factory.generate_read_aux(rs1, &mut row_slice.reads_aux[0]);
            // row_slice.reads_aux[1] is disabled
        }
        aux_cols_factory.generate_write_aux(rd, &mut row_slice.writes_aux);
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
