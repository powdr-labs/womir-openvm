#![cfg_attr(target_os = "zkvm", no_std)]
#![cfg_attr(target_os = "zkvm", no_main)]

use tiny_keccak::{Hasher, Keccak};

// OpenVM specific
#[cfg(target_os = "zkvm")]
mod platform {
    use crate::main;
    openvm::entry!(main);

    use openvm::io::{read, reveal_u32};

    pub fn read_u32() -> u32 {
        read()
    }

    pub fn finish(first_byte: u8) {
        reveal_u32(first_byte as u32, 0);
    }
}

// WASM specific
#[cfg(not(target_os = "zkvm"))]
mod platform {
    pub fn read_u32() -> u32 {
        womir_guest_io::read_u32()
    }

    pub fn finish(first_byte: u8) {
        // let expected = read_u32();
        // assert_eq!(first_byte as u32, expected);
    }
}

// Shared bench
pub fn main() {
    let n: u32 = platform::read_u32();

    let mut output = [0u8; 32];

    for _ in 0..n {
        let mut hasher = Keccak::v256();
        hasher.update(&output);
        hasher.finalize(&mut output);
    }

    platform::finish(output[0]);
}
