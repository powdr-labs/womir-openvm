mod air;
pub mod columns;
mod execution;
/// Preflight and CPU trace generation
pub mod trace;

use std::mem::MaybeUninit;

pub use air::*;
pub use columns::*;
pub use trace::*;

use crate::keccak256::{KECCAK_WIDTH_BYTES, KECCAK_WIDTH_U64S};

pub const NUM_OP_ROWS_PER_INS: usize = 1;

#[derive(derive_new::new, Clone, Copy)]
pub struct KeccakfExecutor {
    pub offset: usize,
    pub pointer_max_bits: usize,
}

#[cfg(target_endian = "little")]
fn copy_bytes_to_u64(src: &[u8; KECCAK_WIDTH_BYTES]) -> [u64; KECCAK_WIDTH_U64S] {
    let mut dst = MaybeUninit::<[u64; KECCAK_WIDTH_U64S]>::uninit();

    unsafe {
        std::ptr::copy_nonoverlapping(
            src.as_ptr(),
            dst.as_mut_ptr() as *mut u8,
            KECCAK_WIDTH_BYTES,
        );
        dst.assume_init()
    }
}

#[cfg(not(target_endian = "little"))]
fn copy_bytes_to_u64(src: &[u8; KECCAK_WIDTH_BYTES]) -> [u64; KECCAK_WIDTH_U64S] {
    let mut dst = [0u64; KECCAK_WIDTH_U64S];
    for (u64_word, chunk) in dst.iter_mut().zip(src.chunks_exact(8)) {
        *u64_word = u64::from_le_bytes(chunk.try_into().unwrap());
    }
    dst
}

#[cfg(target_endian = "little")]
fn transmute_u64_to_bytes(src: [u64; KECCAK_WIDTH_U64S]) -> [u8; KECCAK_WIDTH_BYTES] {
    unsafe { std::mem::transmute::<[u64; KECCAK_WIDTH_U64S], [u8; KECCAK_WIDTH_BYTES]>(src) }
}

#[cfg(not(target_endian = "little"))]
fn transmute_u64_to_bytes(src: [u64; KECCAK_WIDTH_U64S]) -> [u8; KECCAK_WIDTH_BYTES] {
    let mut dst = [0u8; KECCAK_WIDTH_BYTES];
    for (chunk, u64_word) in dst.chunks_exact_mut(8).zip(src.iter()) {
        chunk.copy_from_slice(&u64_word.to_le_bytes());
    }
    dst
}

pub(crate) fn keccakf_postimage_bytes(
    preimage_buffer_bytes: &[u8; KECCAK_WIDTH_BYTES],
) -> [u8; KECCAK_WIDTH_BYTES] {
    let mut state = copy_bytes_to_u64(preimage_buffer_bytes);
    tiny_keccak::keccakf(&mut state);
    transmute_u64_to_bytes(state)
}
