use std::{
    borrow::Borrow,
    sync::{Arc, Mutex},
};

use openvm_circuit::{
    arch::{
        AdapterAirContext, BasicAdapterInterface, ExecutionBridge,
        ExecutionBus, ExecutionState, MinimalInstruction, VmAdapterAir, VmAdapterInterface,
    },
    system::{
        memory::{MemoryController, offline_checker::MemoryBridge},
        program::ProgramBus,
    },
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::ColumnsAir,
};
use serde::{Deserialize, Serialize};
use struct_reflection::{StructReflection, StructReflectionHelper};

use crate::{
    FrameBridge, FrameBus, FrameState, VmAdapterChipWom, WomBridge, WomController, WomRecord,
    adapters::{compose, decompose, frame_allocator::FrameAllocator},
};

use super::RV32_REGISTER_NUM_LIMBS;

#[derive(Debug)]
pub struct AllocateFrameAdapterChipWom {
    pub air: AllocateFrameAdapterAirWom,
    frame_stack: Arc<Mutex<Vec<u32>>>,
    frame_allocator: Arc<Mutex<FrameAllocator>>,
}

impl AllocateFrameAdapterChipWom {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        frame_bus: FrameBus,
        memory_bridge: MemoryBridge,
        wom_bridge: WomBridge,
        frame_allocator: Arc<Mutex<FrameAllocator>>,
        frame_stack: Arc<Mutex<Vec<u32>>>,
    ) -> Self {
        Self {
            air: AllocateFrameAdapterAirWom {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                wom_bridge,
                frame_bridge: FrameBridge::new(frame_bus),
                _memory_bridge: memory_bridge,
            },
            frame_allocator,
            frame_stack,
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

impl<F: PrimeField32> VmAdapterChipWom<F> for AllocateFrameAdapterChipWom {
    type ReadRecord = AllocateFrameReadRecord;
    type WriteRecord = AllocateFrameWriteRecord<F>;
    type Air = AllocateFrameAdapterAirWom;
    type Interface = BasicAdapterInterface<
        F,
        MinimalInstruction<F>,
        1,
        0,
        RV32_REGISTER_NUM_LIMBS,
        RV32_REGISTER_NUM_LIMBS,
    >;

    fn preprocess(
        &mut self,
        _memory: &mut MemoryController<F>,
        wom: &mut WomController<F>,
        fp: u32,
        instruction: &Instruction<F>,
    ) -> eyre::Result<(
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

        let allocated_ptr = self
            .frame_allocator
            .lock()
            .unwrap()
            .allocate(amount_bytes)
            .expect("WOM frame allocation failed: not enough free contiguous space");

        {
            let mut frame_stack = self.frame_stack.lock().unwrap();
            frame_stack.push(allocated_ptr);
            //println!("A STACK: {frame_stack:?}");
        }

        let amount_bytes = decompose(amount_bytes);

        Ok(([amount_bytes], AllocateFrameReadRecord { allocated_ptr }))
    }

    fn postprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        wom: &mut WomController<F>,
        instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
        from_frame: FrameState<u32>,
        read_record: &Self::ReadRecord,
    ) -> eyre::Result<(ExecutionState<u32>, u32, Self::WriteRecord)> {
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
    ) {
    }

    fn air(&self) -> &Self::Air {
        &self.air
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
