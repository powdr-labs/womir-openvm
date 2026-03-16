#![cfg_attr(target_os = "zkvm", no_std)]
#![cfg_attr(target_os = "zkvm", no_main)]
#![allow(clippy::needless_range_loop)]

use core::array;
use ruint::aliases::U256;

// OpenVM specific
#[cfg(target_os = "zkvm")]
mod platform {
    use crate::main;
    openvm::entry!(main);

    pub fn read_u32() -> u32 {
        openvm::io::read()
    }
}

// WASM specific
#[cfg(not(target_os = "zkvm"))]
mod platform {
    pub fn read_u32() -> u32 {
        crush_guest_io::read_u32()
    }
}

const N: usize = 10;
type Matrix = [[U256; N]; N];

pub fn get_matrix(val: u32) -> Matrix {
    array::from_fn(|_| array::from_fn(|_| U256::from(val)))
}

pub fn mult(a: &Matrix, b: &Matrix) -> Matrix {
    let mut c = get_matrix(0);
    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

pub fn get_identity_matrix() -> Matrix {
    let mut res = get_matrix(0);
    for i in 0..N {
        res[i][i] = U256::from(1u32);
    }
    res
}

pub fn main() {
    let reps = platform::read_u32();

    let a = get_identity_matrix();
    let b = get_matrix(28);

    for _ in 0..reps {
        let c = mult(&a, &b);
        if c != b {
            panic!("Matrix multiplication failed");
        }
    }
}
