#![no_std]

pub const KECCAK_WIDTH_BYTES: usize = 200;
pub const KECCAK_RATE: usize = 136;
pub const KECCAK_OUTPUT_SIZE: usize = 32;

#[link(wasm_import_module = "env")]
unsafe extern "C" {
    /// XOR `len` bytes from `input` into `buffer`.
    /// Both pointers must be 4-byte aligned and `len` must be a multiple of 4.
    pub unsafe fn __native_xorin(buffer: *mut u8, input: *const u8, len: usize);
    /// Apply keccak-f[1600] permutation in-place to the 200-byte state at `buffer`.
    /// The pointer must be 4-byte aligned.
    pub unsafe fn __native_keccakf(buffer: *mut u8);
}

/// XOR `len` bytes from `input` into `buffer` using the native precompile.
pub fn native_xorin(buffer: &mut [u8], input: &[u8], len: usize) {
    assert!(len <= buffer.len());
    assert!(len <= input.len());
    assert!(len % 4 == 0);
    unsafe {
        __native_xorin(buffer.as_mut_ptr(), input.as_ptr(), len);
    }
}

/// Apply keccak-f[1600] permutation in-place to a 200-byte state buffer.
pub fn native_keccakf(state: &mut [u8; KECCAK_WIDTH_BYTES]) {
    unsafe {
        __native_keccakf(state.as_mut_ptr());
    }
}

/// Compute keccak256 hash of `data` using native xorin + keccakf precompiles.
pub fn keccak256(data: &[u8]) -> [u8; KECCAK_OUTPUT_SIZE] {
    let mut state = [0u8; KECCAK_WIDTH_BYTES];

    // Absorb phase: process full blocks
    let mut offset = 0;
    while offset + KECCAK_RATE <= data.len() {
        native_xorin(&mut state, &data[offset..], KECCAK_RATE);
        native_keccakf(&mut state);
        offset += KECCAK_RATE;
    }

    // Absorb remaining bytes (need padding)
    let remaining = data.len() - offset;
    if remaining > 0 {
        // XOR remaining bytes (must be multiple of 4, pad with zeros)
        let aligned_remaining = ((remaining + 3) / 4) * 4;
        // We need to copy remaining bytes into an aligned buffer for the xorin call
        let mut padded = [0u8; KECCAK_RATE];
        padded[..remaining].copy_from_slice(&data[offset..]);

        // Can only xorin multiples of 4
        if aligned_remaining > 0 {
            native_xorin(&mut state, &padded, aligned_remaining);
        }

        // For any bytes in the range [aligned_remaining..remaining] that we XOR'd with zeros,
        // that's fine since 0 XOR x = x.
    }

    // Apply keccak padding: pad10*1 rule for keccak256
    // First padding byte: 0x01 (for keccak256; SHA3 uses 0x06)
    state[remaining] ^= 0x01;
    // Last padding byte in rate
    state[KECCAK_RATE - 1] ^= 0x80;

    // Final permutation
    native_keccakf(&mut state);

    // Squeeze: extract first 32 bytes
    let mut output = [0u8; KECCAK_OUTPUT_SIZE];
    output.copy_from_slice(&state[..KECCAK_OUTPUT_SIZE]);
    output
}
