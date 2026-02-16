use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::{
        AdapterAirContext, AdapterTraceExecutor, AdapterTraceFiller, VmAdapterAir,
        VmAdapterInterface, get_record_from_slice,
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
use openvm_instructions::{LocalOpcode, instruction::Instruction, riscv::RV32_REGISTER_AS};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::ColumnsAir,
};
use openvm_womir_transpiler::CallOpcode;
use struct_reflection::{StructReflection, StructReflectionHelper};

use openvm_circuit::arch::ExecutionBridge;

use crate::execution::ExecutionState;
use crate::memory_config::FP_AS;

use super::{RV32_REGISTER_NUM_LIMBS, tracing_read, tracing_read_fp};

/// Call adapter columns.
///
/// Memory operations in timestamp order:
///   0. Read FP from FP_AS
///   1. Read to_fp_reg (FP offset for CALL/CALL_INDIRECT, absolute FP for RET) - always
///   2. Read to_pc_reg (for RET, CALL_INDIRECT) - conditional on has_pc_read
///   3. Write save_fp (for CALL, CALL_INDIRECT) - conditional on has_save
///   4. Write save_pc (for CALL, CALL_INDIRECT) - conditional on has_save
///   5. Write new FP to FP_AS - always
#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct CallAdapterCols<T> {
    pub from_state: ExecutionState<T>,

    /// Operand e: pointer to register holding FP offset (CALL/CALL_INDIRECT) or absolute FP (RET)
    pub to_fp_reg_ptr: T,
    /// Operand b: pointer to register where old FP is saved
    pub save_fp_ptr: T,
    /// Operand a: pointer to register where old PC+1 is saved
    pub save_pc_ptr: T,
    /// Operand c: pointer to register holding target PC (for RET, CALL_INDIRECT)
    pub to_pc_reg_ptr: T,
    /// Operand d: immediate PC target (for CALL)
    pub to_pc_imm: T,

    /// Auxiliary columns for memory operations
    pub fp_read_aux: MemoryReadAuxCols<T>,
    pub to_fp_read_aux: MemoryReadAuxCols<T>,
    pub to_pc_read_aux: MemoryReadAuxCols<T>,
    pub save_fp_write_aux: MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LIMBS>,
    pub save_pc_write_aux: MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LIMBS>,
    /// FP_AS write: prev_data is 1 field element (native32 cell type)
    pub fp_write_aux: MemoryWriteAuxCols<T, 1>,
}

/// Custom interface for Call chip.
pub struct CallAdapterInterface<AB: InteractionBuilder>(std::marker::PhantomData<AB>);

pub struct CallInstruction<T> {
    pub is_valid: T,
    pub opcode: T,
    /// 1 if we read a register for PC (RET, CALL_INDIRECT), 0 otherwise
    pub has_pc_read: T,
    /// 1 if we save FP and PC (CALL, CALL_INDIRECT), 0 otherwise
    pub has_save: T,
}

impl<AB: InteractionBuilder> VmAdapterInterface<AB::Expr> for CallAdapterInterface<AB> {
    /// Reads:
    ///   [0] = to_fp_reg data (4 limbs) - new FP value
    ///   [1] = to_pc_reg data (4 limbs) - target PC (when has_pc_read=1)
    type Reads = [[AB::Expr; RV32_REGISTER_NUM_LIMBS]; 2];
    /// Writes:
    ///   0 = [save_fp_data, save_pc_data] (4 limbs each)
    ///   1 = [new_fp_as_field] (1 field element for FP_AS write)
    type Writes = ([[AB::Expr; RV32_REGISTER_NUM_LIMBS]; 2], [AB::Expr; 1]);
    type ProcessedInstruction = CallInstruction<AB::Expr>;
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct CallAdapterAir {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
}

impl<F: Field> BaseAir<F> for CallAdapterAir {
    fn width(&self) -> usize {
        CallAdapterCols::<F>::width()
    }
}

impl<F: Field> ColumnsAir<F> for CallAdapterAir {
    fn columns(&self) -> Option<Vec<String>> {
        CallAdapterCols::<F>::struct_reflection()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for CallAdapterAir {
    type Interface = CallAdapterInterface<AB>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local: &CallAdapterCols<_> = local.borrow();
        let timestamp = local.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_canonical_usize(timestamp_delta - 1)
        };

        let is_valid = ctx.instruction.is_valid.clone();
        let has_pc_read = ctx.instruction.has_pc_read.clone();
        let has_save = ctx.instruction.has_save.clone();

        // 0. Read current FP from FP_AS
        self.memory_bridge
            .read(
                MemoryAddress::new(AB::F::from_canonical_u32(FP_AS), AB::F::ZERO),
                [local.from_state.fp],
                timestamp_pp(),
                &local.fp_read_aux,
            )
            .eval(builder, is_valid.clone());

        // 1. Read to_fp_reg: the new FP value from register
        let to_fp_data: [AB::Expr; RV32_REGISTER_NUM_LIMBS] =
            std::array::from_fn(|i| ctx.reads[0][i].clone());
        self.memory_bridge
            .read(
                MemoryAddress::new(
                    AB::F::from_canonical_u32(RV32_REGISTER_AS),
                    local.to_fp_reg_ptr + local.from_state.fp,
                ),
                to_fp_data,
                timestamp_pp(),
                &local.to_fp_read_aux,
            )
            .eval(builder, is_valid.clone());

        // 2. Read to_pc_reg (conditional on has_pc_read)
        let to_pc_data: [AB::Expr; RV32_REGISTER_NUM_LIMBS] =
            std::array::from_fn(|i| ctx.reads[1][i].clone());
        self.memory_bridge
            .read(
                MemoryAddress::new(
                    has_pc_read.clone(),
                    local.to_pc_reg_ptr + local.from_state.fp,
                ),
                to_pc_data,
                timestamp_pp(),
                &local.to_pc_read_aux,
            )
            .eval(builder, has_pc_read.clone());

        // The core provides the actual new FP value in writes.1[0].
        // For CALL/CALL_INDIRECT this is old_fp + offset, for RET this is the absolute FP.
        let new_fp_composed = ctx.writes.1[0].clone();

        // Constrain that old_fp_data (from core writes.0[0]) decomposes to from_state.fp
        let old_fp_composed = ctx.writes.0[0]
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (i, limb)| {
                acc + limb.clone() * AB::Expr::from_canonical_u32(1u32 << (i * 8))
            });
        builder
            .when(is_valid.clone())
            .assert_eq(old_fp_composed, local.from_state.fp);

        // 3. Write save_fp (conditional on has_save)
        // All saves are relative to the NEW frame pointer
        let save_fp_data: [AB::Expr; RV32_REGISTER_NUM_LIMBS] =
            std::array::from_fn(|i| ctx.writes.0[0][i].clone());
        self.memory_bridge
            .write(
                MemoryAddress::new(
                    has_save.clone(),
                    local.save_fp_ptr + new_fp_composed.clone(),
                ),
                save_fp_data,
                timestamp_pp(),
                &local.save_fp_write_aux,
            )
            .eval(builder, has_save.clone());

        // 4. Write save_pc (conditional on has_save)
        let save_pc_data: [AB::Expr; RV32_REGISTER_NUM_LIMBS] =
            std::array::from_fn(|i| ctx.writes.0[1][i].clone());
        self.memory_bridge
            .write(
                MemoryAddress::new(
                    has_save.clone(),
                    local.save_pc_ptr + new_fp_composed.clone(),
                ),
                save_pc_data,
                timestamp_pp(),
                &local.save_pc_write_aux,
            )
            .eval(builder, has_save);

        // 5. Write new FP to FP_AS
        let new_fp_write: [AB::Expr; 1] = std::array::from_fn(|i| ctx.writes.1[i].clone());
        self.memory_bridge
            .write(
                MemoryAddress::new(AB::F::from_canonical_u32(FP_AS), AB::F::ZERO),
                new_fp_write,
                timestamp_pp(),
                &local.fp_write_aux,
            )
            .eval(builder, is_valid.clone());

        // Determine to_pc: either from immediate (d operand) or from register read
        let to_pc_from_reg = ctx.reads[1]
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (i, limb)| {
                acc + limb.clone() * AB::Expr::from_canonical_u32(1u32 << (i * 8))
            });
        let to_pc =
            local.to_pc_imm * (AB::Expr::ONE - has_pc_read.clone()) + to_pc_from_reg * has_pc_read;

        self.execution_bridge
            .execute_and_increment_or_set_pc(
                ctx.instruction.opcode,
                [
                    local.save_pc_ptr.into(),
                    local.save_fp_ptr.into(),
                    local.to_pc_reg_ptr.into(),
                    local.to_pc_imm.into(),
                    local.to_fp_reg_ptr.into(),
                ],
                local.from_state.into(),
                AB::F::from_canonical_usize(timestamp_delta),
                (0u32, Some(to_pc)),
            )
            .eval(builder, is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &CallAdapterCols<_> = local.borrow();
        cols.from_state.pc
    }
}

// ========== Executor ==========

#[derive(Clone, Default)]
pub struct CallAdapterExecutor;

/// Record for the FP_AS write. The prev_data is a single u32 (stored as field element
/// in FP_AS which uses native32 cell type).
#[repr(C)]
#[derive(AlignedBytesBorrow, Debug, Clone, Copy)]
pub struct FpWriteAuxRecord {
    pub prev_timestamp: u32,
    pub prev_fp: u32,
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct CallAdapterRecord {
    pub from_pc: u32,
    pub fp: u32,
    pub from_timestamp: u32,

    pub to_fp_reg_ptr: u32,
    pub save_fp_ptr: u32,
    pub save_pc_ptr: u32,
    pub to_pc_reg_ptr: u32,
    pub to_pc_imm: u32,

    pub has_pc_read: u8,
    pub has_save: u8,

    pub fp_read_aux: MemoryReadAuxRecord,
    pub to_fp_read_aux: MemoryReadAuxRecord,
    pub to_pc_read_aux: MemoryReadAuxRecord,
    pub save_fp_write_aux: MemoryWriteBytesAuxRecord<RV32_REGISTER_NUM_LIMBS>,
    pub save_pc_write_aux: MemoryWriteBytesAuxRecord<RV32_REGISTER_NUM_LIMBS>,
    pub fp_write_aux: FpWriteAuxRecord,
}

impl<F: PrimeField32> AdapterTraceExecutor<F> for CallAdapterExecutor {
    const WIDTH: usize = size_of::<CallAdapterCols<u8>>();
    /// ReadData: [new_fp_bytes, to_pc_reg_bytes]
    type ReadData = [[u8; RV32_REGISTER_NUM_LIMBS]; 2];
    /// WriteData: (save_fp_bytes, save_pc_bytes, new_fp_value)
    type WriteData = (
        [u8; RV32_REGISTER_NUM_LIMBS],
        [u8; RV32_REGISTER_NUM_LIMBS],
        u32,
    );
    type RecordMut<'a> = &'a mut CallAdapterRecord;

    #[inline(always)]
    fn start(pc: u32, memory: &TracingMemory, record: &mut &mut CallAdapterRecord) {
        record.from_pc = pc;
        record.from_timestamp = memory.timestamp;
    }

    #[inline(always)]
    fn read(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        record: &mut &mut CallAdapterRecord,
    ) -> Self::ReadData {
        let &Instruction { a, b, c, d, e, .. } = instruction;

        // 0. Read FP
        record.fp = tracing_read_fp::<F>(memory, &mut record.fp_read_aux.prev_timestamp);

        // Decode instruction operands
        record.to_fp_reg_ptr = e.as_canonical_u32();
        record.save_fp_ptr = b.as_canonical_u32();
        record.save_pc_ptr = a.as_canonical_u32();
        record.to_pc_reg_ptr = c.as_canonical_u32();
        record.to_pc_imm = d.as_canonical_u32();

        // Determine flags from opcode
        let local_idx = instruction
            .opcode
            .local_opcode_idx(CallOpcode::CLASS_OFFSET);
        let opcode = CallOpcode::from_usize(local_idx);

        record.has_pc_read = match opcode {
            CallOpcode::RET | CallOpcode::CALL_INDIRECT => 1,
            _ => 0,
        };
        record.has_save = match opcode {
            CallOpcode::CALL | CallOpcode::CALL_INDIRECT => 1,
            _ => 0,
        };

        // 1. Read to_fp_reg: new FP value
        let new_fp_bytes: [u8; RV32_REGISTER_NUM_LIMBS] = tracing_read(
            memory,
            RV32_REGISTER_AS,
            record.to_fp_reg_ptr + record.fp,
            &mut record.to_fp_read_aux.prev_timestamp,
        );

        // 2. Read to_pc_reg (conditional)
        let to_pc_bytes: [u8; RV32_REGISTER_NUM_LIMBS] = if record.has_pc_read == 1 {
            tracing_read(
                memory,
                RV32_REGISTER_AS,
                record.to_pc_reg_ptr + record.fp,
                &mut record.to_pc_read_aux.prev_timestamp,
            )
        } else {
            memory.increment_timestamp();
            [0u8; RV32_REGISTER_NUM_LIMBS]
        };

        [new_fp_bytes, to_pc_bytes]
    }

    #[inline(always)]
    fn write(
        &self,
        memory: &mut TracingMemory,
        _instruction: &Instruction<F>,
        data: Self::WriteData,
        record: &mut &mut CallAdapterRecord,
    ) {
        let (save_fp_bytes, save_pc_bytes, new_fp_val) = data;

        // 3. Write save_fp (conditional on has_save) - relative to NEW frame
        if record.has_save == 1 {
            let (t_prev, prev_data) = super::timed_write(
                memory,
                RV32_REGISTER_AS,
                record.save_fp_ptr + new_fp_val,
                save_fp_bytes,
            );
            record.save_fp_write_aux.prev_timestamp = t_prev;
            record.save_fp_write_aux.prev_data = prev_data;
        } else {
            memory.increment_timestamp();
        }

        // 4. Write save_pc (conditional on has_save) - relative to NEW frame
        if record.has_save == 1 {
            let (t_prev, prev_data) = super::timed_write(
                memory,
                RV32_REGISTER_AS,
                record.save_pc_ptr + new_fp_val,
                save_pc_bytes,
            );
            record.save_pc_write_aux.prev_timestamp = t_prev;
            record.save_pc_write_aux.prev_data = prev_data;
        } else {
            memory.increment_timestamp();
        }

        // 5. Write new FP to FP_AS
        let new_fp_field = F::from_canonical_u32(new_fp_val);
        // SAFETY: FP_AS uses native32 cell type (F), block size 1, align 1.
        let (t_prev, prev_data) = unsafe { memory.write::<F, 1, 1>(FP_AS, 0, [new_fp_field]) };
        record.fp_write_aux.prev_timestamp = t_prev;
        record.fp_write_aux.prev_fp = prev_data[0].as_canonical_u32();
    }
}

// ========== Filler ==========

#[derive(derive_new::new)]
pub struct CallAdapterFiller;

impl<F: PrimeField32> AdapterTraceFiller<F> for CallAdapterFiller {
    const WIDTH: usize = size_of::<CallAdapterCols<u8>>();

    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        let record: &CallAdapterRecord = unsafe { get_record_from_slice(&mut adapter_row, ()) };
        let adapter_row: &mut CallAdapterCols<F> = adapter_row.borrow_mut();

        // Total: 6 timestamp increments (indices 0..5)
        let mut timestamp = record.from_timestamp + 5;

        // 5. FP write (native32 cell type: prev_data is a field element)
        adapter_row.fp_write_aux.prev_data = [F::from_canonical_u32(record.fp_write_aux.prev_fp)];
        mem_helper.fill(
            record.fp_write_aux.prev_timestamp,
            timestamp,
            adapter_row.fp_write_aux.as_mut(),
        );
        timestamp -= 1;

        // 4. save_pc write
        if record.has_save != 0 {
            adapter_row
                .save_pc_write_aux
                .set_prev_data(record.save_pc_write_aux.prev_data.map(F::from_canonical_u8));
            mem_helper.fill(
                record.save_pc_write_aux.prev_timestamp,
                timestamp,
                adapter_row.save_pc_write_aux.as_mut(),
            );
        } else {
            mem_helper.fill_zero(adapter_row.save_pc_write_aux.as_mut());
        }
        timestamp -= 1;

        // 3. save_fp write
        if record.has_save != 0 {
            adapter_row
                .save_fp_write_aux
                .set_prev_data(record.save_fp_write_aux.prev_data.map(F::from_canonical_u8));
            mem_helper.fill(
                record.save_fp_write_aux.prev_timestamp,
                timestamp,
                adapter_row.save_fp_write_aux.as_mut(),
            );
        } else {
            mem_helper.fill_zero(adapter_row.save_fp_write_aux.as_mut());
        }
        timestamp -= 1;

        // 2. to_pc_reg read
        if record.has_pc_read != 0 {
            mem_helper.fill(
                record.to_pc_read_aux.prev_timestamp,
                timestamp,
                adapter_row.to_pc_read_aux.as_mut(),
            );
        } else {
            mem_helper.fill_zero(adapter_row.to_pc_read_aux.as_mut());
        }
        timestamp -= 1;

        // 1. to_fp_reg read
        mem_helper.fill(
            record.to_fp_read_aux.prev_timestamp,
            timestamp,
            adapter_row.to_fp_read_aux.as_mut(),
        );
        timestamp -= 1;

        // 0. FP read
        mem_helper.fill(
            record.fp_read_aux.prev_timestamp,
            timestamp,
            adapter_row.fp_read_aux.as_mut(),
        );

        // Scalar fields
        adapter_row.to_pc_imm = F::from_canonical_u32(record.to_pc_imm);
        adapter_row.to_pc_reg_ptr = F::from_canonical_u32(record.to_pc_reg_ptr);
        adapter_row.save_pc_ptr = F::from_canonical_u32(record.save_pc_ptr);
        adapter_row.save_fp_ptr = F::from_canonical_u32(record.save_fp_ptr);
        adapter_row.to_fp_reg_ptr = F::from_canonical_u32(record.to_fp_reg_ptr);
        adapter_row.from_state.timestamp = F::from_canonical_u32(record.from_timestamp);
        adapter_row.from_state.fp = F::from_canonical_u32(record.fp);
        adapter_row.from_state.pc = F::from_canonical_u32(record.from_pc);
    }
}
