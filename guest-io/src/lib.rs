#![no_std]

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use serde::de::DeserializeOwned;

#[link(wasm_import_module = "env")]
unsafe extern "C" {
    /// Prepares the next object to be read from the hint stream.
    pub safe fn __hint_input();
    /// Reads `num_words` 32-bit words from the hint stream into memory at `ptr`.
    pub unsafe fn __hint_buffer(ptr: *mut u8, num_words: u32);
    /// Prints `num_bytes` bytes from memory at `ptr` as a debug message.
    pub unsafe fn __debug_print(ptr: *const u8, num_bytes: u32);
    /// Aborts execution.
    pub safe fn abort() -> !;
}

/// Read a single `u32` from the hint stream.
///
/// Each hint stream item has format `[byte_len, ...data_words]`.
/// For a `u32`, the item is `[4, value]`, so we skip the length word
/// and read the data word.
pub fn read_u32() -> u32 {
    __hint_input();
    let _len = read_word(); // skip length word
    read_word()
}

/// Read and deserialize a value of type `T` from the hint stream.
///
/// Reads a length-prefixed byte blob (one hint item) and deserializes
/// it with `postcard`.
pub fn read<T: DeserializeOwned>() -> T {
    let bytes = read_vec();
    postcard::from_bytes(&bytes).expect("deserialization failed")
}

/// Read a length-prefixed byte vector from the hint stream.
///
/// Each hint stream item has format `[byte_len, ...data_words]`.
/// This reads the length word, then reads that many bytes of data.
pub fn read_vec() -> Vec<u8> {
    __hint_input();
    let len = read_word();
    read_vec_by_len(len as usize)
}

/// Print a debug message.
pub fn debug_print(msg: &str) {
    unsafe {
        __debug_print(msg.as_ptr(), msg.len() as u32);
    }
}

// -- internal helpers --

fn read_word() -> u32 {
    let mut bytes = [0u8; 4];
    unsafe { __hint_buffer(bytes.as_mut_ptr(), 1) }
    u32::from_le_bytes(bytes)
}

fn read_vec_by_len(len: usize) -> Vec<u8> {
    let num_words = len.div_ceil(4);
    let capacity = num_words * 4;
    let mut bytes: Vec<u8> = vec![0; capacity];
    unsafe { __hint_buffer(bytes.as_mut_ptr(), num_words as u32) }
    // SAFETY: We populate a `Vec<u8>` by hintstore-ing `num_words` 4-byte words.
    // We set the length to `len` and ignore the extra `capacity - len` bytes.
    unsafe {
        bytes.set_len(len);
    }
    bytes
}
