use std::{borrow::Borrow, iter};

use itertools::izip;
use openvm_circuit::{
    arch::ExecutionBridge,
    system::memory::{
        offline_checker::{MemoryBridge, MemoryWriteAuxCols},
        MemoryAddress,
    },
};
use openvm_circuit_primitives::bitwise_op_lookup::BitwiseOperationLookupBus;
use openvm_crush_transpiler::KeccakfOpcode;
use openvm_instructions::{
    riscv::{RV32_CELL_BITS, RV32_MEMORY_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_stark_backend::{
    interaction::{InteractionBuilder, PermutationCheckBus},
    p3_air::{Air, BaseAir},
    p3_field::FieldAlgebra,
    p3_matrix::Matrix,
    rap::{BaseAirWithPublicValues, ColumnsAir, PartitionedBaseAir},
};
use struct_reflection::StructReflectionHelper;

use crate::{
    adapters::{fp_addr, reg_addr},
    execution::ExecutionState,
    keccak256::{
        keccakf_op::columns::{KeccakfOpCols, NUM_KECCAKF_OP_COLS},
        KECCAK_WORD_SIZE,
    },
};

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct KeccakfOpAir {
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
    pub bitwise_lookup_bus: BitwiseOperationLookupBus,
    /// Direct bus with keccakf pre- or post-state. Bus message is
    /// ```text
    /// is_post || timestamp || state_u16_limbs
    /// ```
    pub keccakf_state_bus: PermutationCheckBus,
    pub ptr_max_bits: usize,
    pub(super) offset: usize,
}

impl<F> BaseAirWithPublicValues<F> for KeccakfOpAir {}
impl<F> PartitionedBaseAir<F> for KeccakfOpAir {}
impl<F> ColumnsAir<F> for KeccakfOpAir {
    fn columns(&self) -> Option<Vec<String>> {
        <KeccakfOpCols<F> as struct_reflection::StructReflectionHelper>::struct_reflection()
    }
}
impl<F> BaseAir<F> for KeccakfOpAir {
    fn width(&self) -> usize {
        NUM_KECCAKF_OP_COLS
    }
}

impl<AB: InteractionBuilder> Air<AB> for KeccakfOpAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &KeccakfOpCols<_> = (*local).borrow();

        let is_valid = local.is_valid;
        builder.assert_bool(is_valid);

        let start_timestamp = local.timestamp;
        let mut timestamp_delta = 0usize;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            start_timestamp + AB::F::from_canonical_usize(timestamp_delta - 1)
        };

        // ======== Read FP from FP_AS =========
        let fp = local.fp;
        self.memory_bridge
            .read(
                fp_addr::<AB::F>(),
                [fp],
                timestamp_pp(),
                &local.fp_aux,
            )
            .eval(builder, is_valid);

        // ======== Read `rd` (FP-relative) =========
        let rd_ptr = local.rd_ptr;
        let buffer_ptr_limbs = local.buffer_ptr_limbs;
        self.memory_bridge
            .read(
                reg_addr(rd_ptr + fp),
                buffer_ptr_limbs,
                timestamp_pp(),
                &local.rd_aux,
            )
            .eval(builder, is_valid);
        // Range check that buffer_ptr_limbs fits in [0, 2^ptr_max_bits) as u32
        {
            assert!(self.ptr_max_bits >= RV32_CELL_BITS * (RV32_REGISTER_NUM_LIMBS - 1));
            let limb_shift = AB::F::from_canonical_usize(
                1 << (RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - self.ptr_max_bits),
            );
            let need_range_check = [
                buffer_ptr_limbs[RV32_REGISTER_NUM_LIMBS - 1],
                buffer_ptr_limbs[RV32_REGISTER_NUM_LIMBS - 1],
            ];
            for pair in need_range_check.chunks_exact(2) {
                self.bitwise_lookup_bus
                    .send_range(pair[0] * limb_shift, pair[1] * limb_shift)
                    .eval(builder, is_valid);
            }
        }
        // Now it is safe to cast buffer_ptr to F
        let buffer_ptr: AB::Expr = crate::adapters::abstract_compose(local.buffer_ptr_limbs);

        // ======== Constrain that post-state consists of bytes =========
        for pair in local.postimage.chunks_exact(2) {
            self.bitwise_lookup_bus
                .send_range(pair[0], pair[1])
                .eval(builder, is_valid);
        }

        // ======== Constrain new writes of `buffer` to memory =========
        for (word_idx, (prev_word, post_word, base_aux)) in izip!(
            local.preimage.chunks_exact(KECCAK_WORD_SIZE),
            local.postimage.chunks_exact(KECCAK_WORD_SIZE),
            local.buffer_word_aux
        )
        .enumerate()
        {
            let ptr = buffer_ptr.clone() + AB::F::from_canonical_usize(word_idx * KECCAK_WORD_SIZE);
            let prev_data: &[_; KECCAK_WORD_SIZE] = prev_word.try_into().unwrap();
            let data: &[_; KECCAK_WORD_SIZE] = post_word.try_into().unwrap();
            let write_aux = MemoryWriteAuxCols {
                base: base_aux,
                prev_data: *prev_data,
            };
            self.memory_bridge
                .write(
                    MemoryAddress::new(AB::F::from_canonical_u32(RV32_MEMORY_AS), ptr),
                    *data,
                    timestamp_pp(),
                    &write_aux,
                )
                .eval(builder, is_valid);
        }

        // ======== Execution bus =========
        self.execution_bridge
            .execute_and_increment_pc(
                AB::Expr::from_canonical_usize(KeccakfOpcode::KECCAKF as usize + self.offset),
                [
                    rd_ptr.into(),
                    AB::Expr::ZERO,
                    AB::Expr::ZERO,
                    AB::Expr::from_canonical_u32(RV32_REGISTER_AS),
                    AB::Expr::from_canonical_u32(RV32_MEMORY_AS),
                ],
                ExecutionState::new(local.pc, local.fp, local.timestamp).into(),
                AB::F::from_canonical_usize(timestamp_delta),
            )
            .eval(builder, is_valid);

        // ======== KeccakF State Interaction =======
        self.keccakf_state_bus.send(
            builder,
            iter::empty()
                .chain([AB::Expr::ZERO, local.timestamp.into()])
                .chain(
                    local
                        .preimage
                        .chunks(2)
                        .map(|pair| pair[0] + pair[1] * AB::F::from_canonical_u32(256)),
                ),
            is_valid,
        );
        self.keccakf_state_bus.send(
            builder,
            iter::empty()
                .chain([AB::Expr::ONE, local.timestamp.into()])
                .chain(
                    local
                        .postimage
                        .chunks(2)
                        .map(|pair| pair[0] + pair[1] * AB::F::from_canonical_u32(256)),
                ),
            is_valid,
        );
    }
}
