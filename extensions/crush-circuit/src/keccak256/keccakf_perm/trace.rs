use std::{
    array::from_fn,
    borrow::BorrowMut,
    sync::{Arc, Mutex},
};


use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    p3_field::{FieldAlgebra, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    p3_maybe_rayon::prelude::*,
    prover::{
        cpu::CpuBackend,
        types::AirProvingContext,
    },
    Chip,
};
use p3_keccak_air::{generate_trace_rows, NUM_KECCAK_COLS, NUM_ROUNDS};

use crate::keccak256::{
    keccakf_op::KeccakfRecord,
    keccakf_perm::{KeccakfPermCols, NUM_KECCAKF_PERM_COLS},
};

#[derive(Clone, derive_new::new)]
pub struct KeccakfPermChip {
    pub(crate) shared_records: Arc<Mutex<Vec<KeccakfRecord>>>,
}

impl<RA, SC> Chip<RA, CpuBackend<SC>> for KeccakfPermChip
where
    SC: StarkGenericConfig,
    Val<SC>: PrimeField32,
{
    fn generate_proving_ctx(&self, _: RA) -> AirProvingContext<CpuBackend<SC>> {
        let records: Vec<_> = std::mem::take(&mut self.shared_records.lock().unwrap());
        let states = records
            .iter()
            .map(|record| {
                // Sequential bytes use standard convention: state[x + 5*y] = A[x, y].
                // p3-keccak-air (539bbc8) expects input[x*5 + y] = A[x, y] (transposed).
                // So we transpose the 5x5 state.
                let sequential: [u64; 25] = from_fn(|i| {
                    u64::from_le_bytes(
                        record.preimage_buffer_bytes[i * 8..i * 8 + 8]
                            .try_into()
                            .unwrap(),
                    )
                });
                // Transpose: sequential[x + 5*y] -> transposed[x*5 + y]
                from_fn(|i| {
                    let x = i / 5;
                    let y = i % 5;
                    sequential[x + 5 * y]
                })
            })
            .collect::<Vec<_>>();

        let p3_trace = generate_trace_rows::<Val<SC>>(states, 0);
        let mut values = Val::<SC>::zero_vec(NUM_KECCAKF_PERM_COLS * p3_trace.height());
        values
            .par_chunks_exact_mut(NUM_KECCAKF_PERM_COLS)
            .zip(p3_trace.values.par_chunks_exact(NUM_KECCAK_COLS))
            .enumerate()
            .for_each(|(row_idx, (row, p3_row))| {
                row[..NUM_KECCAK_COLS].copy_from_slice(p3_row);

                if row_idx % NUM_ROUNDS == (NUM_ROUNDS - 1) {
                    let record_idx = row_idx / NUM_ROUNDS;
                    if let Some(record) = records.get(record_idx) {
                        let local: &mut KeccakfPermCols<_> = row.borrow_mut();
                        local.inner.export = Val::<SC>::ONE;
                        local.timestamp = Val::<SC>::from_canonical_u32(record.timestamp);
                    }
                }
            });
        let matrix = RowMajorMatrix::new(values, NUM_KECCAKF_PERM_COLS);
        AirProvingContext::simple_no_pis(Arc::new(matrix))
    }
}
