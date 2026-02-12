use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::*,
    system::memory::{
        MemoryAddress, MemoryAuxColsFactory,
        offline_checker::{
            MemoryBridge, MemoryReadAuxCols, MemoryReadAuxRecord, MemoryWriteAuxCols,
        },
        online::TracingMemory,
    },
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    utils::not,
};
use openvm_circuit_primitives_derive::{AlignedBorrow, AlignedBytesBorrow};
use openvm_instructions::{
    LocalOpcode,
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_CELL_BITS, RV32_MEMORY_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
};
use openvm_rv32im_circuit::Rv32HintStoreVar;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    p3_matrix::{Matrix, dense::RowMajorMatrix},
    p3_maybe_rayon::prelude::*,
    rap::{BaseAirWithPublicValues, ColumnsAir, PartitionedBaseAir},
};
use openvm_womir_transpiler::{
    HintStoreOpcode,
    HintStoreOpcode::{HINT_BUFFER, HINT_STOREW},
};

use crate::adapters::{read_rv32_register, tracing_read, tracing_read_fp, tracing_write};
use crate::execution::ExecutionState;
use crate::memory_config::{FP_AS, FpMemory};
use struct_reflection::{StructReflection, StructReflectionHelper};

mod execution;

#[repr(C)]
#[derive(AlignedBorrow, Debug, StructReflection)]
pub struct Rv32HintStoreCols<T> {
    // common
    pub is_single: T,
    pub is_buffer: T,
    // should be 1 for single
    pub rem_words_limbs: [T; RV32_REGISTER_NUM_LIMBS],

    pub from_state: ExecutionState<T>,
    pub fp_read_aux: MemoryReadAuxCols<T>,
    pub mem_ptr_ptr: T,
    pub mem_ptr_limbs: [T; RV32_REGISTER_NUM_LIMBS],
    pub mem_ptr_aux_cols: MemoryReadAuxCols<T>,

    pub write_aux: MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LIMBS>,
    pub data: [T; RV32_REGISTER_NUM_LIMBS],

    // only buffer
    pub is_buffer_start: T,
    pub num_words_ptr: T,
    pub num_words_aux_cols: MemoryReadAuxCols<T>,
    pub mem_imm: T,
}

#[derive(Copy, Clone, Debug, derive_new::new)]
pub struct Rv32HintStoreAir {
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
    pub bitwise_operation_lookup_bus: BitwiseOperationLookupBus,
    pub offset: usize,
    pointer_max_bits: usize,
}

impl<F: Field> BaseAir<F> for Rv32HintStoreAir {
    fn width(&self) -> usize {
        Rv32HintStoreCols::<F>::width()
    }
}

impl<F: Field> ColumnsAir<F> for Rv32HintStoreAir {
    fn columns(&self) -> Option<Vec<String>> {
        Rv32HintStoreCols::<F>::struct_reflection()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for Rv32HintStoreAir {}
impl<F: Field> PartitionedBaseAir<F> for Rv32HintStoreAir {}

impl<AB: InteractionBuilder> Air<AB> for Rv32HintStoreAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local_cols: &Rv32HintStoreCols<AB::Var> = (*local).borrow();
        let next = main.row_slice(1);
        let next_cols: &Rv32HintStoreCols<AB::Var> = (*next).borrow();

        let timestamp: AB::Var = local_cols.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::Expr::from_canonical_usize(timestamp_delta - 1)
        };

        builder.assert_bool(local_cols.is_single);
        builder.assert_bool(local_cols.is_buffer);
        builder.assert_bool(local_cols.is_buffer_start);
        builder
            .when(local_cols.is_buffer_start)
            .assert_one(local_cols.is_buffer);
        builder.assert_bool(local_cols.is_single + local_cols.is_buffer);

        let is_valid = local_cols.is_single + local_cols.is_buffer;
        let is_start = local_cols.is_single + local_cols.is_buffer_start;
        // `is_end` is false iff the next row is a buffer row that is not buffer start
        // This is boolean because is_buffer_start == 1 => is_buffer == 1
        // Note: every non-valid row has `is_end == 1`
        let is_end = not::<AB::Expr>(next_cols.is_buffer) + next_cols.is_buffer_start;

        let mut rem_words = AB::Expr::ZERO;
        let mut next_rem_words = AB::Expr::ZERO;
        let mut mem_ptr = AB::Expr::ZERO;
        let mut next_mem_ptr = AB::Expr::ZERO;
        for i in (0..RV32_REGISTER_NUM_LIMBS).rev() {
            rem_words = rem_words * AB::F::from_canonical_u32(1 << RV32_CELL_BITS)
                + local_cols.rem_words_limbs[i];
            next_rem_words = next_rem_words * AB::F::from_canonical_u32(1 << RV32_CELL_BITS)
                + next_cols.rem_words_limbs[i];
            mem_ptr = mem_ptr * AB::F::from_canonical_u32(1 << RV32_CELL_BITS)
                + local_cols.mem_ptr_limbs[i];
            next_mem_ptr = next_mem_ptr * AB::F::from_canonical_u32(1 << RV32_CELL_BITS)
                + next_cols.mem_ptr_limbs[i];
        }

        // Constrain that if local is invalid, then the next state is invalid as well
        builder
            .when_transition()
            .when(not::<AB::Expr>(is_valid.clone()))
            .assert_zero(next_cols.is_single + next_cols.is_buffer);

        // Constrain that when we start a buffer, the is_buffer_start is set to 1
        builder
            .when(local_cols.is_single)
            .assert_one(is_end.clone());
        builder
            .when_first_row()
            .assert_one(not::<AB::Expr>(local_cols.is_buffer) + local_cols.is_buffer_start);

        // 1. Read FP from FP address space
        self.memory_bridge
            .read(
                MemoryAddress::new(AB::F::from_canonical_u32(FP_AS), AB::F::ZERO),
                [local_cols.from_state.fp],
                timestamp_pp(),
                &local_cols.fp_read_aux,
            )
            .eval(builder, is_start.clone());

        // For STOREW: constrain mem_ptr_limbs = mem_ptr_ptr + fp (direct, no register read)
        builder.when(local_cols.is_single).assert_eq(
            mem_ptr.clone(),
            local_cols.mem_ptr_ptr + local_cols.from_state.fp,
        );

        // 2. Read mem_ptr register (BUFFER only)
        self.memory_bridge
            .read(
                MemoryAddress::new(
                    AB::F::from_canonical_u32(RV32_REGISTER_AS),
                    local_cols.mem_ptr_ptr + local_cols.from_state.fp,
                ),
                local_cols.mem_ptr_limbs,
                timestamp_pp(),
                &local_cols.mem_ptr_aux_cols,
            )
            .eval(builder, local_cols.is_buffer_start);

        // 3. Read num_words register (BUFFER only)
        self.memory_bridge
            .read(
                MemoryAddress::new(
                    AB::F::from_canonical_u32(RV32_REGISTER_AS),
                    local_cols.num_words_ptr + local_cols.from_state.fp,
                ),
                local_cols.rem_words_limbs,
                timestamp_pp(),
                &local_cols.num_words_aux_cols,
            )
            .eval(builder, local_cols.is_buffer_start);

        // 4. Write hint data
        // Write AS: STOREW writes to RV32_REGISTER_AS, BUFFER writes to RV32_MEMORY_AS
        let write_as = local_cols.is_single * AB::F::from_canonical_u32(RV32_REGISTER_AS)
            + local_cols.is_buffer * AB::F::from_canonical_u32(RV32_MEMORY_AS);
        // Write address: for STOREW = compose(mem_ptr_limbs), for BUFFER = compose(mem_ptr_limbs) + mem_imm
        let write_addr = mem_ptr.clone() + local_cols.is_buffer * local_cols.mem_imm;
        self.memory_bridge
            .write(
                MemoryAddress::new(write_as, write_addr),
                local_cols.data,
                timestamp_pp(),
                &local_cols.write_aux,
            )
            .eval(builder, is_valid.clone());

        let expected_opcode = (local_cols.is_single
            * AB::F::from_canonical_usize(HINT_STOREW as usize + self.offset))
            + (local_cols.is_buffer
                * AB::F::from_canonical_usize(HINT_BUFFER as usize + self.offset));

        self.execution_bridge
            .execute_and_increment_pc(
                expected_opcode,
                [
                    local_cols.is_single * local_cols.mem_ptr_ptr,
                    local_cols.is_buffer * local_cols.num_words_ptr,
                    local_cols.is_buffer * local_cols.mem_ptr_ptr,
                    local_cols.is_single * AB::F::from_canonical_u32(RV32_REGISTER_AS)
                        + local_cols.is_buffer * local_cols.mem_imm,
                    AB::Expr::ZERO,
                ],
                local_cols.from_state.into(),
                rem_words.clone() * AB::F::from_canonical_usize(timestamp_delta),
            )
            .eval(builder, is_start.clone());

        // Preventing mem_ptr and rem_words overflow
        self.bitwise_operation_lookup_bus
            .send_range(
                local_cols.mem_ptr_limbs[RV32_REGISTER_NUM_LIMBS - 1]
                    * AB::F::from_canonical_usize(
                        1 << (RV32_REGISTER_NUM_LIMBS * RV32_CELL_BITS - self.pointer_max_bits),
                    ),
                local_cols.rem_words_limbs[RV32_REGISTER_NUM_LIMBS - 1]
                    * AB::F::from_canonical_usize(
                        1 << (RV32_REGISTER_NUM_LIMBS * RV32_CELL_BITS - self.pointer_max_bits),
                    ),
            )
            .eval(builder, is_start.clone());

        // Checking that hint is bytes
        for i in 0..RV32_REGISTER_NUM_LIMBS / 2 {
            self.bitwise_operation_lookup_bus
                .send_range(local_cols.data[2 * i], local_cols.data[(2 * i) + 1])
                .eval(builder, is_valid.clone());
        }

        // buffer transition
        builder
            .when(is_valid)
            .when(is_end.clone())
            .assert_one(rem_words.clone());

        let mut when_buffer_transition = builder.when(not::<AB::Expr>(is_end.clone()));
        when_buffer_transition.assert_one(rem_words.clone() - next_rem_words.clone());
        when_buffer_transition.assert_eq(
            next_mem_ptr.clone() - mem_ptr.clone(),
            AB::F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS),
        );
        when_buffer_transition.assert_eq(
            timestamp + AB::F::from_canonical_usize(timestamp_delta),
            next_cols.from_state.timestamp,
        );
        when_buffer_transition.assert_eq(local_cols.mem_imm, next_cols.mem_imm);
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Rv32HintStoreMetadata {
    num_words: usize,
}

impl MultiRowMetadata for Rv32HintStoreMetadata {
    #[inline(always)]
    fn get_num_rows(&self) -> usize {
        self.num_words
    }
}

pub type Rv32HintStoreLayout = MultiRowLayout<Rv32HintStoreMetadata>;

// This is the part of the record that we keep only once per instruction
#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct Rv32HintStoreRecordHeader {
    pub num_words: u32,

    pub from_pc: u32,
    pub timestamp: u32,

    pub fp: u32,
    pub fp_read_aux: MemoryReadAuxRecord,

    pub mem_ptr_ptr: u32,
    pub mem_ptr: u32,
    pub mem_ptr_aux_record: MemoryReadAuxRecord,

    // will set `num_words_ptr` to `u32::MAX` in case of single hint
    pub num_words_ptr: u32,
    pub num_words_read: MemoryReadAuxRecord,

    pub mem_imm: u32,
}

/// **SAFETY**: the order of the fields in `Rv32HintStoreRecord` and `Rv32HintStoreVar` is
/// important. The chip also assumes that the offset of the fields `write_aux` and `data` in
/// `Rv32HintStoreCols` is bigger than `size_of::<Rv32HintStoreRecord>()`
#[derive(Debug)]
pub struct Rv32HintStoreRecordMut<'a> {
    pub inner: &'a mut Rv32HintStoreRecordHeader,
    pub var: &'a mut [Rv32HintStoreVar],
}

/// Custom borrowing that splits the buffer into a fixed `Rv32HintStoreRecord` header
/// followed by a slice of `Rv32HintStoreVar`'s of length `num_words` provided at runtime.
/// Uses `align_to_mut()` to make sure the slice is properly aligned to `Rv32HintStoreVar`.
/// Has debug assertions to make sure the above works as expected.
impl<'a> CustomBorrow<'a, Rv32HintStoreRecordMut<'a>, Rv32HintStoreLayout> for [u8] {
    fn custom_borrow(&'a mut self, layout: Rv32HintStoreLayout) -> Rv32HintStoreRecordMut<'a> {
        // SAFETY:
        // - Caller guarantees through the layout that self has sufficient length for all splits
        // - size_of::<Rv32HintStoreRecordHeader>() is guaranteed <= self.len() by layout
        //   precondition
        let (header_buf, rest) =
            unsafe { self.split_at_mut_unchecked(size_of::<Rv32HintStoreRecordHeader>()) };

        // SAFETY:
        // - rest contains bytes that will be interpreted as Rv32HintStoreVar records
        // - align_to_mut ensures proper alignment for Rv32HintStoreVar type
        // - The layout guarantees sufficient space for layout.metadata.num_words records
        let (_, vars, _) = unsafe { rest.align_to_mut::<Rv32HintStoreVar>() };
        Rv32HintStoreRecordMut {
            inner: header_buf.borrow_mut(),
            var: &mut vars[..layout.metadata.num_words],
        }
    }

    unsafe fn extract_layout(&self) -> Rv32HintStoreLayout {
        let header: &Rv32HintStoreRecordHeader = self.borrow();
        MultiRowLayout::new(Rv32HintStoreMetadata {
            num_words: header.num_words as usize,
        })
    }
}

impl SizedRecord<Rv32HintStoreLayout> for Rv32HintStoreRecordMut<'_> {
    fn size(layout: &Rv32HintStoreLayout) -> usize {
        let mut total_len = size_of::<Rv32HintStoreRecordHeader>();
        // Align the pointer to the alignment of `Rv32HintStoreVar`
        total_len = total_len.next_multiple_of(align_of::<Rv32HintStoreVar>());
        total_len += size_of::<Rv32HintStoreVar>() * layout.metadata.num_words;
        total_len
    }

    fn alignment(_layout: &Rv32HintStoreLayout) -> usize {
        align_of::<Rv32HintStoreRecordHeader>()
    }
}

#[derive(Clone, Copy, derive_new::new)]
pub struct Rv32HintStoreExecutor {
    pub pointer_max_bits: usize,
    pub offset: usize,
}

#[derive(Clone, derive_new::new)]
pub struct Rv32HintStoreFiller {
    pointer_max_bits: usize,
    bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
}

impl<F, RA> PreflightExecutor<F, RA> for Rv32HintStoreExecutor
where
    F: PrimeField32,
    for<'buf> RA:
        RecordArena<'buf, MultiRowLayout<Rv32HintStoreMetadata>, Rv32HintStoreRecordMut<'buf>>,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        if opcode == HINT_STOREW.global_opcode().as_usize() {
            String::from("HINT_STOREW")
        } else if opcode == HINT_BUFFER.global_opcode().as_usize() {
            String::from("HINT_BUFFER")
        } else {
            unreachable!("unsupported opcode: {opcode}")
        }
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let &Instruction {
            opcode, a, b, c, d, ..
        } = instruction;

        let a = a.as_canonical_u32();
        let b = b.as_canonical_u32();
        let c = c.as_canonical_u32();
        let d = d.as_canonical_u32();

        let local_opcode = HintStoreOpcode::from_usize(opcode.local_opcode_idx(self.offset));

        // Read FP untraced first for allocation
        let fp = state.memory.data().fp::<F>();

        // Untraced read of num_words for allocation
        let num_words = if local_opcode == HINT_STOREW {
            1
        } else {
            read_rv32_register(state.memory.data(), b + fp)
        };

        let record = state.ctx.alloc(MultiRowLayout::new(Rv32HintStoreMetadata {
            num_words: num_words as usize,
        }));

        record.inner.from_pc = *state.pc;
        record.inner.timestamp = state.memory.timestamp;

        // Read FP traced (timestamp slot 0)
        record.inner.fp =
            tracing_read_fp::<F>(state.memory, &mut record.inner.fp_read_aux.prev_timestamp);

        debug_assert_ne!(num_words, 0);
        debug_assert!(num_words <= (1 << self.pointer_max_bits));
        record.inner.num_words = num_words;

        if local_opcode == HINT_STOREW {
            // STOREW: a=rd*4, dest is (RV32_REGISTER_AS, a + fp)
            record.inner.mem_ptr_ptr = a;
            record.inner.mem_ptr = a + fp;
            // Skip 2 phantom timestamps for mem_ptr and num_words read slots
            state.memory.increment_timestamp();
            state.memory.increment_timestamp();
            record.inner.num_words_ptr = u32::MAX;
            record.inner.mem_imm = 0;
        } else {
            // BUFFER: read mem_ptr register at c + fp (timestamp slot 1)
            record.inner.mem_ptr_ptr = c;
            record.inner.mem_ptr = u32::from_le_bytes(tracing_read(
                state.memory,
                RV32_REGISTER_AS,
                c + fp,
                &mut record.inner.mem_ptr_aux_record.prev_timestamp,
            ));
            // Read num_words register at b + fp (timestamp slot 2)
            record.inner.num_words_ptr = b;
            tracing_read::<RV32_REGISTER_NUM_LIMBS>(
                state.memory,
                RV32_REGISTER_AS,
                b + fp,
                &mut record.inner.num_words_read.prev_timestamp,
            );
            record.inner.mem_imm = d;
        }

        debug_assert!(record.inner.mem_ptr <= (1 << self.pointer_max_bits));

        if state.streams.hint_stream.len() < RV32_REGISTER_NUM_LIMBS * num_words as usize {
            return Err(ExecutionError::HintOutOfBounds { pc: *state.pc });
        }

        let write_as = if local_opcode == HINT_STOREW {
            RV32_REGISTER_AS
        } else {
            RV32_MEMORY_AS
        };
        let write_base_addr = if local_opcode == HINT_STOREW {
            record.inner.mem_ptr // = a + fp
        } else {
            record.inner.mem_ptr + d // = mem_ptr_value + mem_imm
        };

        for idx in 0..(num_words as usize) {
            if idx != 0 {
                // 3 phantom timestamp increments for non-first words
                // (FP read, mem_ptr read, num_words read slots)
                state.memory.increment_timestamp();
                state.memory.increment_timestamp();
                state.memory.increment_timestamp();
            }

            let data_f: [F; RV32_REGISTER_NUM_LIMBS] =
                std::array::from_fn(|_| state.streams.hint_stream.pop_front().unwrap());
            let data: [u8; RV32_REGISTER_NUM_LIMBS] =
                data_f.map(|byte| byte.as_canonical_u32() as u8);

            record.var[idx].data = data;

            // Write at timestamp slot 3
            tracing_write(
                state.memory,
                write_as,
                write_base_addr + (RV32_REGISTER_NUM_LIMBS * idx) as u32,
                data,
                &mut record.var[idx].data_write_aux.prev_timestamp,
                &mut record.var[idx].data_write_aux.prev_data,
            );
        }
        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}

impl<F: PrimeField32> TraceFiller<F> for Rv32HintStoreFiller {
    fn fill_trace(
        &self,
        mem_helper: &MemoryAuxColsFactory<F>,
        trace: &mut RowMajorMatrix<F>,
        rows_used: usize,
    ) {
        if rows_used == 0 {
            return;
        }

        let width = trace.width;
        debug_assert_eq!(width, size_of::<Rv32HintStoreCols<u8>>());
        let mut trace = &mut trace.values[..width * rows_used];
        let mut sizes = Vec::with_capacity(rows_used);
        let mut chunks = Vec::with_capacity(rows_used);

        while !trace.is_empty() {
            // SAFETY:
            // - caller ensures `trace` contains a valid record representation that was previously
            //   written by the executor
            // - header is the first element of the record
            let record: &Rv32HintStoreRecordHeader =
                unsafe { get_record_from_slice(&mut trace, ()) };
            let (chunk, rest) = trace.split_at_mut(width * record.num_words as usize);
            sizes.push(record.num_words);
            chunks.push(chunk);
            trace = rest;
        }

        let msl_rshift: u32 = ((RV32_REGISTER_NUM_LIMBS - 1) * RV32_CELL_BITS) as u32;
        let msl_lshift: u32 =
            (RV32_REGISTER_NUM_LIMBS * RV32_CELL_BITS - self.pointer_max_bits) as u32;

        chunks
            .par_iter_mut()
            .zip(sizes.par_iter())
            .for_each(|(chunk, &num_words)| {
                // SAFETY:
                // - caller ensures `trace` contains a valid record representation that was
                //   previously written by the executor
                // - chunk contains a valid Rv32HintStoreRecordMut with the exact layout specified
                // - get_record_from_slice will correctly split the buffer into header and variable
                //   components based on this layout
                let record: Rv32HintStoreRecordMut = unsafe {
                    get_record_from_slice(
                        chunk,
                        MultiRowLayout::new(Rv32HintStoreMetadata {
                            num_words: num_words as usize,
                        }),
                    )
                };
                self.bitwise_lookup_chip.request_range(
                    (record.inner.mem_ptr >> msl_rshift) << msl_lshift,
                    (num_words >> msl_rshift) << msl_lshift,
                );

                // 4 timestamp slots per word
                let mut timestamp = record.inner.timestamp + num_words * 4;
                let mut mem_ptr = record.inner.mem_ptr + num_words * RV32_REGISTER_NUM_LIMBS as u32;

                let is_single = record.inner.num_words_ptr == u32::MAX;

                // SAFETY: The record header overlaps with the beginning of the cols struct
                // in the trace buffer. When we fill `fp_read_aux` (starting at cols FE 9),
                // it overwrites record fields at FE 9+ (`num_words_read.prev_timestamp`
                // and `mem_imm`). Save these values before any aux fills.
                let num_words_read_prev_ts = record.inner.num_words_read.prev_timestamp;
                let mem_imm = record.inner.mem_imm;

                chunk
                    .rchunks_exact_mut(width)
                    .zip(record.var.iter().enumerate().rev())
                    .for_each(|(row, (idx, var))| {
                        for pair in var.data.chunks_exact(2) {
                            self.bitwise_lookup_chip
                                .request_range(pair[0] as u32, pair[1] as u32);
                        }

                        let cols: &mut Rv32HintStoreCols<F> = row.borrow_mut();
                        timestamp -= 4;

                        // Fill fp_read_aux (timestamp slot 0, gated by is_start)
                        if idx == 0 {
                            mem_helper.fill(
                                record.inner.fp_read_aux.prev_timestamp,
                                timestamp,
                                cols.fp_read_aux.as_mut(),
                            );
                        } else {
                            mem_helper.fill_zero(cols.fp_read_aux.as_mut());
                        }

                        // Fill mem_ptr_aux_cols (timestamp slot 1, gated by is_buffer_start)
                        if idx == 0 && !is_single {
                            mem_helper.fill(
                                record.inner.mem_ptr_aux_record.prev_timestamp,
                                timestamp + 1,
                                cols.mem_ptr_aux_cols.as_mut(),
                            );
                        } else {
                            mem_helper.fill_zero(cols.mem_ptr_aux_cols.as_mut());
                        }

                        // Fill num_words_aux_cols (timestamp slot 2, gated by is_buffer_start)
                        if idx == 0 && !is_single {
                            mem_helper.fill(
                                num_words_read_prev_ts,
                                timestamp + 2,
                                cols.num_words_aux_cols.as_mut(),
                            );
                            cols.num_words_ptr = F::from_canonical_u32(record.inner.num_words_ptr);
                        } else {
                            mem_helper.fill_zero(cols.num_words_aux_cols.as_mut());
                            cols.num_words_ptr = F::ZERO;
                        }

                        cols.is_buffer_start = F::from_bool(idx == 0 && !is_single);

                        // Note: writing in reverse
                        cols.data = var.data.map(|x| F::from_canonical_u8(x));

                        // Fill write_aux (timestamp slot 3)
                        cols.write_aux.set_prev_data(
                            var.data_write_aux
                                .prev_data
                                .map(|x| F::from_canonical_u8(x)),
                        );
                        mem_helper.fill(
                            var.data_write_aux.prev_timestamp,
                            timestamp + 3,
                            cols.write_aux.as_mut(),
                        );

                        mem_ptr -= RV32_REGISTER_NUM_LIMBS as u32;
                        cols.mem_ptr_limbs = mem_ptr.to_le_bytes().map(|x| F::from_canonical_u8(x));
                        cols.mem_ptr_ptr = F::from_canonical_u32(record.inner.mem_ptr_ptr);

                        cols.from_state.timestamp = F::from_canonical_u32(timestamp);
                        cols.from_state.pc = F::from_canonical_u32(record.inner.from_pc);
                        cols.from_state.fp = F::from_canonical_u32(record.inner.fp);

                        cols.rem_words_limbs = (num_words - idx as u32)
                            .to_le_bytes()
                            .map(|x| F::from_canonical_u8(x));
                        cols.is_buffer = F::from_bool(!is_single);
                        cols.is_single = F::from_bool(is_single);
                        cols.mem_imm = F::from_canonical_u32(mem_imm);
                    });
            })
    }
}

pub type Rv32HintStoreChip<F> = VmChipWrapper<F, Rv32HintStoreFiller>;
