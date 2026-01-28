use openvm_circuit::arch::*;
use openvm_circuit::system::memory::online::TracingMemory;
use openvm_instructions::{
    LocalOpcode,
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_IMM_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
};
use openvm_rv32im_transpiler::ShiftOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::adapters::imm_to_bytes;

// Re-export upstream types that we don't modify
pub use openvm_rv32im_circuit::{ShiftCoreAir, ShiftCoreCols, ShiftCoreRecord, ShiftFiller};

// Our own ShiftExecutor that uses FP-aware adapters
#[derive(Clone, Copy)]
pub struct ShiftExecutor<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    adapter: A,
    pub offset: usize,
}

impl<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> ShiftExecutor<A, NUM_LIMBS, LIMB_BITS> {
    pub fn new(adapter: A, offset: usize) -> Self {
        assert_eq!(NUM_LIMBS % 2, 0, "Number of limbs must be divisible by 2");
        Self { adapter, offset }
    }
}

// FpPreflightExecutor implementation when adapter is FP-aware
impl<F, A, RA, const NUM_LIMBS: usize, const LIMB_BITS: usize> crate::FpPreflightExecutor<F, RA>
    for ShiftExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    A: 'static
        + crate::FpAdapterTraceExecutor<
            F,
            ReadData: Into<[[u8; NUM_LIMBS]; 2]>,
            WriteData: From<[[u8; NUM_LIMBS]; 1]>,
        >,
    for<'buf> RA: RecordArena<
            'buf,
            EmptyAdapterCoreLayout<F, A>,
            (
                A::RecordMut<'buf>,
                &'buf mut ShiftCoreRecord<NUM_LIMBS, LIMB_BITS>,
            ),
        >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("{:?}", ShiftOpcode::from_usize(opcode - self.offset))
    }

    fn execute_with_fp(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
        fp: u32,
    ) -> Result<Option<u32>, ExecutionError> {
        let Instruction {
            opcode, b, c, e, ..
        } = instruction;

        let local_opcode = ShiftOpcode::from_usize(opcode.local_opcode_idx(self.offset));

        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

        // Call FP-aware start
        A::start_with_fp(*state.pc, fp, state.memory, &mut adapter_record);

        // Check if second operand is immediate or register
        let e_u32 = e.as_canonical_u32();
        let is_imm = e_u32 == RV32_IMM_AS;

        let rs1: [u8; NUM_LIMBS];
        let rs2: [u8; NUM_LIMBS];

        if is_imm {
            // Read only rs1 from adapter (first operand)
            let rs1_addr = b.as_canonical_u32() + fp;
            let (_counter, rs1_read) = unsafe {
                state
                    .memory
                    .read::<u8, NUM_LIMBS, RV32_REGISTER_NUM_LIMBS>(RV32_REGISTER_AS, rs1_addr)
            };
            rs1 = rs1_read;

            // Construct rs2 from immediate
            let c_u32 = c.as_canonical_u32();
            let imm_bytes = imm_to_bytes(c_u32);
            let mut rs2_imm = [0u8; NUM_LIMBS];
            rs2_imm[..4.min(NUM_LIMBS)].copy_from_slice(&imm_bytes[..4.min(NUM_LIMBS)]);
            rs2 = rs2_imm;
        } else {
            // Read both operands from adapter
            let [rs1_read, rs2_read] = self
                .adapter
                .read(state.memory, instruction, &mut adapter_record)
                .into();
            rs1 = rs1_read;
            rs2 = rs2_read;
        }

        let (output, _, _) = run_shift::<NUM_LIMBS, LIMB_BITS>(local_opcode, &rs1, &rs2);

        core_record.b = rs1;
        core_record.c = rs2;
        core_record.local_opcode = local_opcode as u8;

        self.adapter.write(
            state.memory,
            instruction,
            [output].into(),
            &mut adapter_record,
        );
        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        // FP doesn't change for shift operations
        Ok(None)
    }
}

// Helper functions for shift computation
// These need to be local since they're not exported from upstream

pub(super) fn run_shift<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    opcode: ShiftOpcode,
    x: &[u8; NUM_LIMBS],
    y: &[u8; NUM_LIMBS],
) -> ([u8; NUM_LIMBS], usize, usize) {
    let (limb_shift, bit_shift) = get_shift::<NUM_LIMBS, LIMB_BITS>(y);
    let output = match opcode {
        ShiftOpcode::SLL => run_shift_left::<NUM_LIMBS, LIMB_BITS>(x, limb_shift, bit_shift),
        ShiftOpcode::SRL => {
            run_shift_right::<NUM_LIMBS, LIMB_BITS>(x, limb_shift, bit_shift, false)
        }
        ShiftOpcode::SRA => run_shift_right::<NUM_LIMBS, LIMB_BITS>(x, limb_shift, bit_shift, true),
    };
    (output, limb_shift, bit_shift)
}

fn run_shift_left<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u8; NUM_LIMBS],
    limb_shift: usize,
    bit_shift: usize,
) -> [u8; NUM_LIMBS] {
    let mut res = [0u8; NUM_LIMBS];
    let limb_mask = (1 << LIMB_BITS) - 1;
    for i in 0..NUM_LIMBS {
        if i >= limb_shift {
            let prev_val = u32::from(x[i - limb_shift]) << bit_shift;
            res[i] = (prev_val & limb_mask) as u8;
            if bit_shift != 0 && i > limb_shift {
                res[i] |= (u32::from(x[i - limb_shift - 1]) >> (LIMB_BITS - bit_shift)) as u8;
            }
        }
    }
    res
}

fn run_shift_right<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u8; NUM_LIMBS],
    limb_shift: usize,
    bit_shift: usize,
    is_signed: bool,
) -> [u8; NUM_LIMBS] {
    let mut res = [0u8; NUM_LIMBS];
    let limb_mask = (1 << LIMB_BITS) - 1;
    let sign = if is_signed && (x[NUM_LIMBS - 1] >> (LIMB_BITS - 1) == 1) {
        limb_mask
    } else {
        0
    };
    for i in 0..NUM_LIMBS {
        if i + limb_shift < NUM_LIMBS {
            let next_val = u32::from(x[i + limb_shift]) >> bit_shift;
            res[i] = (next_val & limb_mask) as u8;
            if bit_shift != 0 && i + limb_shift + 1 < NUM_LIMBS {
                res[i] |= ((u32::from(x[i + limb_shift + 1]) << (LIMB_BITS - bit_shift))
                    & limb_mask) as u8;
            } else if bit_shift != 0 && i + limb_shift + 1 == NUM_LIMBS {
                res[i] |= ((sign << (LIMB_BITS - bit_shift)) & limb_mask) as u8;
            }
        } else {
            res[i] = sign as u8;
        }
    }
    res
}

fn get_shift<const NUM_LIMBS: usize, const LIMB_BITS: usize>(y: &[u8]) -> (usize, usize) {
    let shift_amount = (u32::from_le_bytes(y[0..4].try_into().unwrap())
        % (NUM_LIMBS as u32 * LIMB_BITS as u32)) as usize;
    let limb_shift = shift_amount / LIMB_BITS;
    let bit_shift = shift_amount % LIMB_BITS;
    (limb_shift, bit_shift)
}
