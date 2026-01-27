use openvm_circuit::arch::*;
use openvm_circuit::system::memory::online::TracingMemory;
use openvm_instructions::{LocalOpcode, instruction::Instruction, program::DEFAULT_PC_STEP};
use openvm_rv32im_transpiler::DivRemOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

// Re-export upstream types that we don't modify
pub use openvm_rv32im_circuit::{DivRemCoreAir, DivRemCoreCols, DivRemCoreRecord, DivRemFiller};

// Our own DivRemExecutor that uses FP-aware adapters
#[derive(Clone, Copy, derive_new::new)]
pub struct DivRemExecutor<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    adapter: A,
    pub offset: usize,
}

// FpPreflightExecutor implementation when adapter is FP-aware
impl<F, A, RA, const NUM_LIMBS: usize, const LIMB_BITS: usize> crate::FpPreflightExecutor<F, RA>
    for DivRemExecutor<A, NUM_LIMBS, LIMB_BITS>
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
            (A::RecordMut<'buf>, &'buf mut DivRemCoreRecord<NUM_LIMBS>),
        >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("{:?}", DivRemOpcode::from_usize(opcode - self.offset))
    }

    fn execute_with_fp(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
        fp: u32,
    ) -> Result<Option<u32>, ExecutionError> {
        let Instruction { opcode, .. } = instruction;

        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

        // Call FP-aware start
        A::start_with_fp(*state.pc, fp, state.memory, &mut adapter_record);

        core_record.local_opcode = opcode.local_opcode_idx(self.offset) as u8;

        let is_signed = core_record.local_opcode == DivRemOpcode::DIV as u8
            || core_record.local_opcode == DivRemOpcode::REM as u8;
        let is_div = core_record.local_opcode == DivRemOpcode::DIV as u8
            || core_record.local_opcode == DivRemOpcode::DIVU as u8;

        [core_record.b, core_record.c] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();

        let b = core_record.b.map(u32::from);
        let c = core_record.c.map(u32::from);
        let (q, r, _, _, _, _) = run_divrem::<NUM_LIMBS, LIMB_BITS>(is_signed, &b, &c);

        let rd = if is_div {
            q.map(|x| x as u8)
        } else {
            r.map(|x| x as u8)
        };

        self.adapter
            .write(state.memory, instruction, [rd].into(), &mut adapter_record);

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        // FP doesn't change for divrem operations
        Ok(None)
    }
}

// Helper functions for division/remainder computation
// These need to be local since they're not exported from upstream

use num_bigint::BigUint;
use num_integer::Integer;

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DivRemCoreSpecialCase {
    None,
    ZeroDivisor,
    SignedOverflow,
}

#[inline(always)]
fn limbs_to_biguint<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u32; NUM_LIMBS],
) -> BigUint {
    let base = BigUint::new(vec![1 << LIMB_BITS]);
    let mut res = BigUint::new(vec![0]);
    for val in x.iter().rev() {
        res *= base.clone();
        res += BigUint::new(vec![*val]);
    }
    res
}

#[inline(always)]
fn biguint_to_limbs<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &BigUint,
) -> [u32; NUM_LIMBS] {
    let limb_mask = (1 << LIMB_BITS) - 1;
    let mut res = [0u32; NUM_LIMBS];
    for (i, limb) in res.iter_mut().enumerate() {
        *limb = ((x >> (i * LIMB_BITS))
            .to_u32_digits()
            .first()
            .copied()
            .unwrap_or(0))
            & limb_mask;
    }
    res
}

#[inline(always)]
fn negate<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u32; NUM_LIMBS],
) -> [u32; NUM_LIMBS] {
    let all_zeros = x.iter().all(|val| *val == 0);
    if all_zeros {
        return [0; NUM_LIMBS];
    }
    let base = BigUint::new(vec![1 << LIMB_BITS]);
    let mut modulus = BigUint::new(vec![1]);
    for _ in 0..NUM_LIMBS {
        modulus *= base.clone();
    }
    biguint_to_limbs::<NUM_LIMBS, LIMB_BITS>(
        &(&modulus - limbs_to_biguint::<NUM_LIMBS, LIMB_BITS>(x)),
    )
}

#[inline(always)]
pub(super) fn run_divrem<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    signed: bool,
    x: &[u32; NUM_LIMBS],
    y: &[u32; NUM_LIMBS],
) -> (
    [u32; NUM_LIMBS],
    [u32; NUM_LIMBS],
    bool,
    bool,
    bool,
    DivRemCoreSpecialCase,
) {
    let x_sign = signed && (x[NUM_LIMBS - 1] >> (LIMB_BITS - 1) == 1);
    let y_sign = signed && (y[NUM_LIMBS - 1] >> (LIMB_BITS - 1) == 1);
    let max_limb = (1 << LIMB_BITS) - 1;

    let zero_divisor = y.iter().all(|val| *val == 0);
    let overflow = x[NUM_LIMBS - 1] == 1 << (LIMB_BITS - 1)
        && x[..(NUM_LIMBS - 1)].iter().all(|val| *val == 0)
        && y.iter().all(|val| *val == max_limb)
        && x_sign
        && y_sign;

    if zero_divisor {
        return (
            [max_limb; NUM_LIMBS],
            *x,
            x_sign,
            y_sign,
            signed,
            DivRemCoreSpecialCase::ZeroDivisor,
        );
    } else if overflow {
        return (
            *x,
            [0; NUM_LIMBS],
            x_sign,
            y_sign,
            false,
            DivRemCoreSpecialCase::SignedOverflow,
        );
    }

    let x_abs = if x_sign {
        negate::<NUM_LIMBS, LIMB_BITS>(x)
    } else {
        *x
    };
    let y_abs = if y_sign {
        negate::<NUM_LIMBS, LIMB_BITS>(y)
    } else {
        *y
    };

    let x_big = limbs_to_biguint::<NUM_LIMBS, LIMB_BITS>(&x_abs);
    let y_big = limbs_to_biguint::<NUM_LIMBS, LIMB_BITS>(&y_abs);
    let (q_abs, r_abs) = x_big.div_rem(&y_big);

    let q_sign = x_sign ^ y_sign;
    let q_nonzero = q_abs.to_u32_digits().iter().any(|&val| val != 0);
    let q = if q_sign && q_nonzero {
        negate::<NUM_LIMBS, LIMB_BITS>(&biguint_to_limbs::<NUM_LIMBS, LIMB_BITS>(&q_abs))
    } else {
        biguint_to_limbs::<NUM_LIMBS, LIMB_BITS>(&q_abs)
    };

    let r_nonzero = r_abs.to_u32_digits().iter().any(|&val| val != 0);
    let r = if x_sign && r_nonzero {
        negate::<NUM_LIMBS, LIMB_BITS>(&biguint_to_limbs::<NUM_LIMBS, LIMB_BITS>(&r_abs))
    } else {
        biguint_to_limbs::<NUM_LIMBS, LIMB_BITS>(&r_abs)
    };

    let q_sign_result = signed && (q[NUM_LIMBS - 1] >> (LIMB_BITS - 1) == 1);

    (
        q,
        r,
        x_sign,
        y_sign,
        q_sign_result,
        DivRemCoreSpecialCase::None,
    )
}

#[allow(dead_code)]
#[inline(always)]
pub(super) fn run_sltu_diff_idx<const NUM_LIMBS: usize>(
    x: &[u32; NUM_LIMBS],
    y: &[u32; NUM_LIMBS],
    cmp: bool,
) -> usize {
    for i in (0..NUM_LIMBS).rev() {
        if x[i] != y[i] {
            assert!((x[i] < y[i]) == cmp);
            return i;
        }
    }
    assert!(!cmp);
    NUM_LIMBS
}

// returns carries of d * q + r
#[allow(dead_code)]
#[inline(always)]
pub(super) fn run_mul_carries<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    signed: bool,
    d: &[u32; NUM_LIMBS],
    q: &[u32; NUM_LIMBS],
    r: &[u32; NUM_LIMBS],
    q_sign: bool,
) -> Vec<u32> {
    let mut carry = vec![0u32; 2 * NUM_LIMBS];
    for i in 0..NUM_LIMBS {
        let mut val = r[i] + if i > 0 { carry[i - 1] } else { 0 };
        for j in 0..=i {
            val += d[j] * q[i - j];
        }
        carry[i] = val >> LIMB_BITS;
    }

    let q_ext = if q_sign && signed {
        (1 << LIMB_BITS) - 1
    } else {
        0
    };
    let d_ext =
        (d[NUM_LIMBS - 1] >> (LIMB_BITS - 1)) * if signed { (1 << LIMB_BITS) - 1 } else { 0 };
    let r_ext =
        (r[NUM_LIMBS - 1] >> (LIMB_BITS - 1)) * if signed { (1 << LIMB_BITS) - 1 } else { 0 };
    let mut d_prefix = 0;
    let mut q_prefix = 0;

    for i in 0..NUM_LIMBS {
        d_prefix += d[i];
        q_prefix += q[i];
        let mut val = carry[NUM_LIMBS + i - 1] + d_prefix * q_ext + q_prefix * d_ext + r_ext;
        for j in (i + 1)..NUM_LIMBS {
            val += d[j] * q[NUM_LIMBS + i - j];
        }
        carry[NUM_LIMBS + i] = val >> LIMB_BITS;
    }
    carry
}
