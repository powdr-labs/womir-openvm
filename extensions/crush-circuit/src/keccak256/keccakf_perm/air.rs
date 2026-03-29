use std::{borrow::Borrow, iter};

use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_stark_backend::{
    air_builders::sub::SubAirBuilder,
    interaction::{InteractionBuilder, PermutationCheckBus},
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::FieldAlgebra,
    p3_matrix::Matrix,
    rap::{BaseAirWithPublicValues, ColumnsAir, PartitionedBaseAir},
};
use p3_keccak_air::{KeccakAir, KeccakCols, NUM_KECCAK_COLS, U64_LIMBS};
use struct_reflection::StructReflectionHelper;

use crate::keccak256::KECCAK_WIDTH_U64S;

#[repr(C)]
#[derive(Debug, AlignedBorrow)]
pub struct KeccakfPermCols<T> {
    pub inner: KeccakCols<T>,
    pub timestamp: T,
}

impl<T> StructReflectionHelper for KeccakfPermCols<T> {
    fn struct_reflection() -> Option<Vec<String>> {
        None
    }
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct KeccakfPermAir {
    pub keccakf_state_bus: PermutationCheckBus,
}

impl<T: Copy> KeccakfPermCols<T> {
    pub fn postimage(&self, y: usize, x: usize, limb: usize) -> T {
        self.inner.a_prime_prime_prime(y, x, limb)
    }

    pub fn is_last_round(&self) -> T {
        *self.inner.step_flags.last().unwrap()
    }
}

pub const NUM_KECCAKF_PERM_COLS: usize = size_of::<KeccakfPermCols<u8>>();

impl<F> BaseAirWithPublicValues<F> for KeccakfPermAir {}
impl<F> PartitionedBaseAir<F> for KeccakfPermAir {}
impl<F> ColumnsAir<F> for KeccakfPermAir {}
impl<F> BaseAir<F> for KeccakfPermAir {
    fn width(&self) -> usize {
        NUM_KECCAKF_PERM_COLS
    }
}

impl<AB: InteractionBuilder> Air<AB> for KeccakfPermAir {
    fn eval(&self, builder: &mut AB) {
        self.eval_keccak_f(builder);

        let main = builder.main();
        let local = main.row_slice(0);
        let local: &KeccakfPermCols<_> = (*local).borrow();

        builder
            .when(local.inner.export)
            .assert_one(local.is_last_round());

        // preimage
        // With transposed input to p3, preimage[y][x] = A[x, y].
        // Sequential bytes: state[i] = A[i%5, i/5] = preimage[i/5][i%5].
        self.keccakf_state_bus.receive(
            builder,
            iter::empty()
                .chain([AB::Expr::ZERO, local.timestamp.into()])
                .chain((0..KECCAK_WIDTH_U64S).flat_map(|i| {
                    let y = i / 5;
                    let x = i % 5;
                    (0..U64_LIMBS).map(move |limb| local.inner.preimage[y][x][limb].into())
                })),
            local.inner.export,
        );
        // postimage
        self.keccakf_state_bus.receive(
            builder,
            iter::empty()
                .chain([AB::Expr::ONE, local.timestamp.into()])
                .chain((0..KECCAK_WIDTH_U64S).flat_map(|i| {
                    let y = i / 5;
                    let x = i % 5;
                    (0..U64_LIMBS).map(move |limb| local.postimage(y, x, limb).into())
                })),
            local.inner.export,
        );
    }
}

impl KeccakfPermAir {
    #[inline]
    pub fn eval_keccak_f<AB: AirBuilder>(&self, builder: &mut AB) {
        let keccakf_air = KeccakAir {};
        let mut sub_builder =
            SubAirBuilder::<AB, KeccakAir, AB::Var>::new(builder, 0..NUM_KECCAK_COLS);
        keccakf_air.eval(&mut sub_builder);
    }
}
