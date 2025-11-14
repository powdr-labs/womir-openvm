use std::{
    borrow::Borrow,
    sync::{Arc, Mutex},
};

use openvm_circuit::{
    arch::{
        AdapterAirContext, AdapterTraceExecutor, BasicAdapterInterface, ExecutionBridge,
        ExecutionState, MinimalInstruction, VmAdapterAir,
    },
    system::memory::offline_checker::MemoryBridge,
};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::instruction::Instruction;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::ColumnsAir,
};
use serde::{Deserialize, Serialize};
use struct_reflection::{StructReflection, StructReflectionHelper};

use crate::{
    FrameBridge, FrameState, WomBridge, WomRecord, WomState,
    adapters::{compose, decompose},
};

use super::RV32_REGISTER_NUM_LIMBS;

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct AllocateFrameAdapterRecord {
    // TODO: figure out what fields are actually needed here
}

#[derive(Clone, derive_new::new)]
pub struct AllocateFrameAdapterExecutor<F> {
    wom_state: Arc<Mutex<WomState<F>>>,
}

impl<F: PrimeField32> AdapterTraceExecutor<F> for AllocateFrameAdapterExecutor<F> {
    // TODO: fix this - seems to be proportional to the number of air columns
    const WIDTH: usize = 4;

    /// Number of entries to allocate
    type ReadData = [[u8; RV32_REGISTER_NUM_LIMBS]; 1];

    /// The pointer to the allocated frame
    type WriteData = [[u8; RV32_REGISTER_NUM_LIMBS]; 1];

    type RecordMut<'a> = AllocateFrameAdapterRecord;

    fn start(
        pc: u32,
        memory: &openvm_circuit::system::memory::online::TracingMemory,
        record: &mut Self::RecordMut<'_>,
    ) {
        // TODO: write some record stuff here
    }

    fn read(
        &self,
        memory: &mut openvm_circuit::system::memory::online::TracingMemory,
        instruction: &Instruction<F>,
        record: &mut Self::RecordMut<'_>,
    ) -> Self::ReadData {
        let Instruction {
            b: amount_imm,
            c: amount_reg,
            d: use_reg,
            ..
        } = *instruction;

        let wom_state = self.wom_state.lock().unwrap();

        let amount = if use_reg == F::ZERO {
            // If use_reg is zero, we use the immediate value
            amount_imm.as_canonical_u32()
        } else {
            // Otherwise, we read the value from the register
            let fp_f = F::from_canonical_u32(wom_state.fp);
            let (_, reg_data) = wom_state
                .wom
                .read::<RV32_REGISTER_NUM_LIMBS>(amount_reg + fp_f);
            compose(reg_data)
        };
        let amount_bytes = RV32_REGISTER_NUM_LIMBS as u32 * amount;

        let allocated_ptr = wom_state
            .frame_allocator
            .lock()
            .unwrap()
            .allocate(amount_bytes)
            .expect("WOM frame allocation failed: not enough free contiguous space");

        wom_state.frame_stack.push(allocated_ptr);
        //println!("A STACK: {frame_stack:?}");

        let amount_bytes = decompose(amount_bytes);

        [amount_bytes]
    }

    fn write(
        &self,
        memory: &mut openvm_circuit::system::memory::online::TracingMemory,
        instruction: &Instruction<F>,
        data: Self::WriteData,
        record: &mut Self::RecordMut<'_>,
    ) {
        let Instruction {
            a: target_reg,
            b,
            f: enabled,
            ..
        } = *instruction;

        memory.increment_timestamp();

        let wom_state = self.wom_state.lock().unwrap();

        if enabled != F::ZERO {
            wom_state
                .wom
                .write(target_reg + F::from_canonical_u32(wom_state.fp), data);
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
    pub amount_reg: T,
    // amount from register
    pub amount: [T; RV32_REGISTER_NUM_LIMBS],
    // immediate amount
    pub amount_imm: T,
    // 0 if imm, 1 if reg
    pub amount_imm_or_reg: T,
    // new frame pointer: provided by the prover
    pub next_frame_ptr: [T; RV32_REGISTER_NUM_LIMBS],
    pub dest_reg: T,
    pub write_mult: T,
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct AllocateFrameAdapterAirWom {
    pub(super) _memory_bridge: MemoryBridge,
    pub(super) wom_bridge: WomBridge,
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) frame_bridge: FrameBridge,
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
        1,
        0,
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
        builder.assert_bool(local.amount_imm_or_reg);

        self.wom_bridge
            .read(local.amount_reg + local.from_frame.fp, local.amount)
            .eval(builder, local.amount_imm_or_reg);

        // write fp
        self.wom_bridge
            .write(
                local.dest_reg + local.from_frame.fp,
                local.next_frame_ptr,
                local.write_mult,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        let timestamp_change = AB::Expr::ONE;

        self.execution_bridge
            .execute_and_increment_pc::<AB>(
                ctx.instruction.opcode,
                [
                    local.dest_reg.into(),
                    local.amount_imm.into(),
                    local.amount_reg.into(),
                    local.amount_imm_or_reg.into(),
                    AB::Expr::ZERO,
                    // TODO: is this always one?
                    AB::Expr::ONE,
                ],
                local.from_state,
                timestamp_change.clone(),
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.frame_bridge
            .keep_fp(local.from_frame, timestamp_change)
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &AllocateFrameAdapterColsWom<_> = local.borrow();
        cols.from_state.pc
    }
}

pub mod frame_allocator {
    use std::collections::{BTreeMap, btree_map::Entry};

    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
    pub struct Range {
        pub start: u32,
        pub end: u32,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct FrameAllocator {
        /// The set of free frame ranges, indexed by its length.
        free_ranges: BTreeMap<u32, Vec<Range>>,
        /// The set of allocated frame ranges. Maps start address to end address.
        allocated_ranges: BTreeMap<u32, u32>,
    }

    impl FrameAllocator {
        pub fn new(largest_address: u32, existing_allocations: BTreeMap<u32, u32>) -> Self {
            // Initialize free_ranges based on existing_allocations
            let mut free_ranges: BTreeMap<u32, Vec<Range>> = BTreeMap::new();
            let mut current_start = 0;

            for (start, end) in &existing_allocations {
                if current_start < *start {
                    let range = Range {
                        start: current_start,
                        end: *start,
                    };
                    let length = range.end - range.start;
                    free_ranges.entry(length).or_default().push(range);
                }
                current_start = *end;
            }

            if current_start < largest_address {
                let range = Range {
                    start: current_start,
                    end: largest_address,
                };
                let length = range.end - range.start;
                free_ranges.entry(length).or_default().push(range);
            }

            Self {
                free_ranges,
                allocated_ranges: existing_allocations,
            }
        }

        pub fn allocate(&mut self, size: u32) -> Option<u32> {
            // Find a free range that can accommodate the requested size
            let fittest_len = *self.free_ranges.range_mut(size..).next()?.0;

            // Apparently there is no way find this entry without searching again...
            // TODO: change this when https://github.com/rust-lang/rust/issues/107540
            // is stabilized.
            let Entry::Occupied(mut entry) = self.free_ranges.entry(fittest_len) else {
                unreachable!();
            };
            let mut range = entry.get_mut().pop().unwrap();
            if entry.get().is_empty() {
                entry.remove();
            }

            let allocated_start = range.start;
            range.start += size;

            if range.start < range.end {
                let length = range.end - range.start;
                self.free_ranges.entry(length).or_default().push(range);
            }

            self.allocated_ranges
                .insert(allocated_start, allocated_start + size);

            Some(allocated_start)
        }

        pub fn get_allocated_ranges(&self) -> &BTreeMap<u32, u32> {
            &self.allocated_ranges
        }
    }
}
