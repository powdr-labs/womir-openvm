use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::{
        AdapterAirContext, AdapterTraceFiller, BasicAdapterInterface, ExecutionBridge,
        ExecutionState, MinimalInstruction, VmAdapterAir, get_record_from_slice,
    },
    system::memory::{
        MemoryAddress, MemoryAuxColsFactory,
        offline_checker::{
            MemoryBridge, MemoryReadAuxCols, MemoryReadAuxRecord, MemoryWriteAuxCols,
            MemoryWriteBytesAuxRecord,
        },
        online::TracingMemory,
    },
};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_REGISTER_AS,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::ColumnsAir,
};

use super::{RV32_REGISTER_NUM_LIMBS, tracing_write};
use crate::adapters::tracing_read;
use struct_reflection::{StructReflection, StructReflectionHelper};

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct MultAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rd_ptr: T,
    pub rs1_ptr: T,
    pub rs2_ptr: T,
    pub reads_aux: [MemoryReadAuxCols<T>; 2],
    pub writes_aux: MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LIMBS>,
}

/// Reads instructions of the form OP a, b, c, d where \[a:4\]_d = \[b:4\]_d op \[c:4\]_d.
/// Operand d can only be 1, and there is no immediate support.
#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct MultAdapterAir {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
}

impl<F: Field> BaseAir<F> for MultAdapterAir {
    fn width(&self) -> usize {
        MultAdapterCols::<F>::width()
    }
}

impl<F: Field> ColumnsAir<F> for MultAdapterAir {
    fn columns(&self) -> Option<Vec<String>> {
        MultAdapterCols::<F>::struct_reflection()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for MultAdapterAir {
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
        let local: &MultAdapterCols<_> = local.borrow();
        let timestamp = local.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_canonical_usize(timestamp_delta - 1)
        };

        self.memory_bridge
            .read(
                MemoryAddress::new(AB::F::from_canonical_u32(RV32_REGISTER_AS), local.rs1_ptr),
                ctx.reads[0].clone(),
                timestamp_pp(),
                &local.reads_aux[0],
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .read(
                MemoryAddress::new(AB::F::from_canonical_u32(RV32_REGISTER_AS), local.rs2_ptr),
                ctx.reads[1].clone(),
                timestamp_pp(),
                &local.reads_aux[1],
            )
            .eval(builder, ctx.instruction.is_valid.clone());

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
                    local.rs2_ptr.into(),
                    AB::Expr::from_canonical_u32(RV32_REGISTER_AS),
                    AB::Expr::ZERO,
                ],
                local.from_state,
                AB::F::from_canonical_usize(timestamp_delta),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &MultAdapterCols<_> = local.borrow();
        cols.from_state.pc
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct MultAdapterRecord {
    pub from_pc: u32,
    pub from_timestamp: u32,
    /// Frame pointer for register access
    pub fp: u32,

    pub rd_ptr: u32,
    pub rs1_ptr: u32,
    pub rs2_ptr: u32,

    pub reads_aux: [MemoryReadAuxRecord; 2],
    pub writes_aux: MemoryWriteBytesAuxRecord<RV32_REGISTER_NUM_LIMBS>,
}

#[derive(Clone, Copy, derive_new::new)]
pub struct MultAdapterExecutor;

#[derive(Clone, Copy, derive_new::new)]
pub struct MultAdapterFiller;

// FP-aware implementation - uses fp + register_address for all register accesses
impl<F> crate::FpAdapterTraceExecutor<F> for MultAdapterExecutor
where
    F: PrimeField32,
{
    type ReadData = [[u8; RV32_REGISTER_NUM_LIMBS]; 2];
    type WriteData = [[u8; RV32_REGISTER_NUM_LIMBS]; 1];
    type RecordMut<'a> = &'a mut MultAdapterRecord;

    #[inline(always)]
    fn start_with_fp(pc: u32, fp: u32, memory: &TracingMemory, record: &mut Self::RecordMut<'_>) {
        record.from_pc = pc;
        record.from_timestamp = memory.timestamp;
        // Store fp in the record for use in read/write
        record.fp = fp;
    }

    #[inline(always)]
    fn read(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        record: &mut Self::RecordMut<'_>,
    ) -> Self::ReadData {
        let &Instruction { b, c, d, .. } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);

        // Use fp + register address for frame-local access
        record.rs1_ptr = b.as_canonical_u32();
        let rs1 = tracing_read(
            memory,
            RV32_REGISTER_AS,
            record.fp + record.rs1_ptr, // FP-relative address
            &mut record.reads_aux[0].prev_timestamp,
        );
        record.rs2_ptr = c.as_canonical_u32();
        let rs2 = tracing_read(
            memory,
            RV32_REGISTER_AS,
            record.fp + record.rs2_ptr, // FP-relative address
            &mut record.reads_aux[1].prev_timestamp,
        );

        [rs1, rs2]
    }

    #[inline(always)]
    fn write(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        data: Self::WriteData,
        record: &mut Self::RecordMut<'_>,
    ) {
        let &Instruction { a, d, .. } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);

        record.rd_ptr = a.as_canonical_u32();
        tracing_write(
            memory,
            RV32_REGISTER_AS,
            record.fp + record.rd_ptr, // FP-relative address
            data[0],
            &mut record.writes_aux.prev_timestamp,
            &mut record.writes_aux.prev_data,
        )
    }
}

impl<F: PrimeField32> AdapterTraceFiller<F> for MultAdapterFiller {
    const WIDTH: usize = size_of::<MultAdapterCols<u8>>();

    #[inline(always)]
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        // SAFETY:
        // - caller ensures `adapter_row` contains a valid record representation that was previously
        //   written by the executor
        // - get_record_from_slice correctly interprets the bytes as MultAdapterRecord
        let record: &MultAdapterRecord = unsafe { get_record_from_slice(&mut adapter_row, ()) };
        let adapter_row: &mut MultAdapterCols<F> = adapter_row.borrow_mut();

        let timestamp = record.from_timestamp;

        adapter_row
            .writes_aux
            .set_prev_data(record.writes_aux.prev_data.map(F::from_canonical_u8));
        mem_helper.fill(
            record.writes_aux.prev_timestamp,
            timestamp + 2,
            adapter_row.writes_aux.as_mut(),
        );

        mem_helper.fill(
            record.reads_aux[1].prev_timestamp,
            timestamp + 1,
            adapter_row.reads_aux[1].as_mut(),
        );

        mem_helper.fill(
            record.reads_aux[0].prev_timestamp,
            timestamp,
            adapter_row.reads_aux[0].as_mut(),
        );

        adapter_row.rs2_ptr = F::from_canonical_u32(record.rs2_ptr);
        adapter_row.rs1_ptr = F::from_canonical_u32(record.rs1_ptr);
        adapter_row.rd_ptr = F::from_canonical_u32(record.rd_ptr);

        adapter_row.from_state.timestamp = F::from_canonical_u32(record.from_timestamp);
        adapter_row.from_state.pc = F::from_canonical_u32(record.from_pc);
    }
}
