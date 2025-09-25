use std::{
    array,
    borrow::{Borrow, BorrowMut},
};

use openvm_circuit::arch::{
    AdapterAirContext, AdapterRuntimeContext, MinimalInstruction, VmAdapterInterface, VmCoreAir,
    VmCoreChip,
};
use openvm_circuit_primitives::utils::not;
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{LocalOpcode, instruction::Instruction};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::{BaseAirWithPublicValues, ColumnsAir},
};
use openvm_womir_transpiler::EqOpcode;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_big_array::BigArray;
use struct_reflection::{StructReflection, StructReflectionHelper};
use strum::IntoEnumIterator;

use openvm_circuit::arch::Result as ResultVm;

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct EqCoreCols<
    T,
    const NUM_LIMBS_READ: usize,
    const NUM_LIMBS_WRITE: usize,
    const LIMB_BITS: usize,
> {
    pub a: [T; NUM_LIMBS_WRITE],
    pub b: [T; NUM_LIMBS_READ],
    pub c: [T; NUM_LIMBS_READ],

    pub cmp_result: T,

    pub opcode_eq_flag: T,
    pub opcode_ne_flag: T,

    pub diff_inv_marker: [T; NUM_LIMBS_READ],
}

#[derive(Copy, Clone, Debug)]
pub struct EqCoreAir<
    const NUM_LIMBS_READ: usize,
    const NUM_LIMBS_WRITE: usize,
    const LIMB_BITS: usize,
> {
    offset: usize,
}

impl<F: Field, const NUM_LIMBS_READ: usize, const NUM_LIMBS_WRITE: usize, const LIMB_BITS: usize>
    BaseAir<F> for EqCoreAir<NUM_LIMBS_READ, NUM_LIMBS_WRITE, LIMB_BITS>
{
    fn width(&self) -> usize {
        EqCoreCols::<F, NUM_LIMBS_READ, NUM_LIMBS_WRITE, LIMB_BITS>::width()
    }
}

impl<F: Field, const NUM_LIMBS_READ: usize, const NUM_LIMBS_WRITE: usize, const LIMB_BITS: usize>
    ColumnsAir<F> for EqCoreAir<NUM_LIMBS_READ, NUM_LIMBS_WRITE, LIMB_BITS>
{
    fn columns(&self) -> Option<Vec<String>> {
        EqCoreCols::<F, NUM_LIMBS_READ, NUM_LIMBS_WRITE, LIMB_BITS>::struct_reflection()
    }
}

impl<F: Field, const NUM_LIMBS_READ: usize, const NUM_LIMBS_WRITE: usize, const LIMB_BITS: usize>
    BaseAirWithPublicValues<F> for EqCoreAir<NUM_LIMBS_READ, NUM_LIMBS_WRITE, LIMB_BITS>
{
}

impl<AB, I, const NUM_LIMBS_READ: usize, const NUM_LIMBS_WRITE: usize, const LIMB_BITS: usize>
    VmCoreAir<AB, I> for EqCoreAir<NUM_LIMBS_READ, NUM_LIMBS_WRITE, LIMB_BITS>
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; NUM_LIMBS_READ]; 2]>,
    I::Writes: From<[[AB::Expr; NUM_LIMBS_WRITE]; 1]>,
    I::ProcessedInstruction: From<MinimalInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &EqCoreCols<_, NUM_LIMBS_READ, NUM_LIMBS_WRITE, LIMB_BITS> = local.borrow();
        let flags = [cols.opcode_eq_flag, cols.opcode_ne_flag];

        let is_valid = flags.iter().fold(AB::Expr::ZERO, |acc, &flag| {
            builder.assert_bool(flag);
            acc + flag.into()
        });
        builder.assert_bool(is_valid.clone());
        builder.assert_bool(cols.cmp_result);

        let a = &cols.a;
        let b = &cols.b;
        let c = &cols.c;
        let inv_marker = &cols.diff_inv_marker;

        // 1 if cmp_result indicates b and c are equal, 0 otherwise
        let cmp_eq =
            cols.cmp_result * cols.opcode_eq_flag + not(cols.cmp_result) * cols.opcode_ne_flag;
        let mut sum = cmp_eq.clone();

        // For EQ, inv_marker is used to check equality of b and c:
        // - If b == c, all inv_marker values must be 0 (sum = 0)
        // - If b != c, inv_marker contains 0s for all positions except ONE position i where b[i] !=
        //   c[i]
        // - At this position, inv_marker[i] contains the multiplicative inverse of (b[i] - c[i])
        // - This ensures inv_marker[i] * (b[i] - c[i]) = 1, making the sum = 1
        // Note: There might be multiple valid inv_marker if b != c.
        // But as long as the trace can provide at least one, that’s sufficient to prove b != c.
        //
        // Note:
        // - If cmp_eq == 0, then it is impossible to have sum != 0 if b == c.
        // - If cmp_eq == 1, then it is impossible for b[i] - c[i] == 0 to pass for all i if b != c.
        for i in 0..NUM_LIMBS_READ {
            sum += (b[i] - c[i]) * inv_marker[i];
            builder.assert_zero(cmp_eq.clone() * (b[i] - c[i]));
        }
        builder.when(is_valid.clone()).assert_one(sum);

        // a == cmp_result
        builder.assert_eq(a[0], cols.cmp_result);
        for limb in a.iter().skip(1) {
            builder.assert_zero(*limb);
        }

        let expected_opcode = flags
            .iter()
            .zip(EqOpcode::iter())
            .fold(AB::Expr::ZERO, |acc, (flag, opcode)| {
                acc + (*flag).into() * AB::Expr::from_canonical_u8(opcode as u8)
            })
            + AB::Expr::from_canonical_usize(self.offset);

        AdapterAirContext {
            to_pc: None,
            reads: [cols.b.map(Into::into), cols.c.map(Into::into)].into(),
            writes: [cols.a.map(Into::into)].into(),
            instruction: MinimalInstruction {
                is_valid,
                opcode: expected_opcode,
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        self.offset
    }
}

#[repr(C)]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "T: Serialize + DeserializeOwned")]
pub struct EqCoreRecord<
    T,
    const NUM_LIMBS_READ: usize,
    const NUM_LIMBS_WRITE: usize,
    const LIMB_BITS: usize,
> {
    #[serde(with = "BigArray")]
    pub a: [T; NUM_LIMBS_WRITE],
    #[serde(with = "BigArray")]
    pub b: [T; NUM_LIMBS_READ],
    #[serde(with = "BigArray")]
    pub c: [T; NUM_LIMBS_READ],
    pub cmp_result: T,
    pub diff_inv_val: T,
    pub diff_idx: usize,
    pub opcode: EqOpcode,
}

pub struct EqCoreChip<
    const NUM_LIMBS_READ: usize,
    const NUM_LIMBS_WRITE: usize,
    const LIMB_BITS: usize,
> {
    pub air: EqCoreAir<NUM_LIMBS_READ, NUM_LIMBS_WRITE, LIMB_BITS>,
}

impl<const NUM_LIMBS_READ: usize, const NUM_LIMBS_WRITE: usize, const LIMB_BITS: usize>
    EqCoreChip<NUM_LIMBS_READ, NUM_LIMBS_WRITE, LIMB_BITS>
{
    pub fn new(offset: usize) -> Self {
        Self {
            air: EqCoreAir { offset },
        }
    }
}

impl<
    F: PrimeField32,
    I: VmAdapterInterface<F>,
    const NUM_LIMBS_READ: usize,
    const NUM_LIMBS_WRITE: usize,
    const LIMB_BITS: usize,
> VmCoreChipWom<F, I> for EqCoreChip<NUM_LIMBS_READ, NUM_LIMBS_WRITE, LIMB_BITS>
where
    I::Reads: Into<[[F; NUM_LIMBS_READ]; 2]>,
    I::Writes: From<[[F; NUM_LIMBS_WRITE]; 1]>,
{
    type Record = EqCoreRecord<F, NUM_LIMBS_READ, NUM_LIMBS_WRITE, LIMB_BITS>;
    type Air = EqCoreAir<NUM_LIMBS_READ, NUM_LIMBS_WRITE, LIMB_BITS>;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        _from_pc: u32,
        reads: I::Reads,
    ) -> ResultVm<(AdapterRuntimeContext<F, I>, Self::Record)> {
        let Instruction { opcode, .. } = instruction;
        let eq_opcode = EqOpcode::from_usize(opcode.local_opcode_idx(self.air.offset));

        let data: [[F; NUM_LIMBS_READ]; 2] = reads.into();
        let b = data[0].map(|x| x.as_canonical_u32());
        let c = data[1].map(|y| y.as_canonical_u32());
        let (cmp_result, diff_idx, diff_inv_val) = run_eq::<F, NUM_LIMBS_READ>(eq_opcode, &b, &c);
        let mut a: [F; NUM_LIMBS_WRITE] = [F::ZERO; NUM_LIMBS_WRITE];
        a[0] = F::from_bool(cmp_result);

        let output = AdapterRuntimeContext {
            to_pc: None,
            writes: [a].into(),
        };

        let record = EqCoreRecord {
            opcode: eq_opcode,
            a,
            b: data[0],
            c: data[1],
            cmp_result: F::from_bool(cmp_result),
            diff_idx,
            diff_inv_val,
        };

        Ok((output, record))
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("{:?}", EqOpcode::from_usize(opcode - self.air.offset))
    }

    fn generate_trace_row(&self, row_slice: &mut [F], record: Self::Record) {
        let row_slice: &mut EqCoreCols<_, NUM_LIMBS_READ, NUM_LIMBS_WRITE, LIMB_BITS> =
            row_slice.borrow_mut();
        row_slice.a = record.a;
        row_slice.b = record.b;
        row_slice.c = record.c;
        row_slice.cmp_result = record.cmp_result;
        row_slice.opcode_eq_flag = F::from_bool(record.opcode == EqOpcode::EQ);
        row_slice.opcode_ne_flag = F::from_bool(record.opcode == EqOpcode::NEQ);
        row_slice.diff_inv_marker = array::from_fn(|i| {
            if i == record.diff_idx {
                record.diff_inv_val
            } else {
                F::ZERO
            }
        });
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}

// Returns (cmp_result, diff_idx, x[diff_idx] - y[diff_idx])
pub(super) fn run_eq<F: PrimeField32, const NUM_LIMBS: usize>(
    local_opcode: EqOpcode,
    x: &[u32; NUM_LIMBS],
    y: &[u32; NUM_LIMBS],
) -> (bool, usize, F) {
    for i in 0..NUM_LIMBS {
        if x[i] != y[i] {
            return (
                local_opcode == EqOpcode::NEQ,
                i,
                (F::from_canonical_u32(x[i]) - F::from_canonical_u32(y[i])).inverse(),
            );
        }
    }
    (local_opcode == EqOpcode::EQ, 0, F::ZERO)
}
