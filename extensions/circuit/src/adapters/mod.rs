use openvm_circuit::system::memory::MemoryController;
use openvm_stark_backend::p3_field::PrimeField32;

mod allocate_frame;
mod alu;
mod consts;
mod copy_into_frame;
mod jaaf;
mod jump;
mod loadstore;

pub use allocate_frame::*;
pub use alu::*;
pub use consts::*;
pub use copy_into_frame::*;
pub use jaaf::*;
pub use jump::*;
pub use loadstore::*;
pub use openvm_instructions::riscv::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};

use crate::WomController;

/// Convert the RISC-V register data (32 bits represented as 4 bytes, where each byte is represented
/// as a field element) back into its value as u32.
pub fn compose<F: PrimeField32>(ptr_data: [F; RV32_REGISTER_NUM_LIMBS]) -> u32 {
    let mut val = 0;
    for (i, limb) in ptr_data.map(|x| x.as_canonical_u32()).iter().enumerate() {
        val += limb << (i * 8);
    }
    val
}

/// inverse of `compose`
pub fn decompose<F: PrimeField32>(value: u32) -> [F; RV32_REGISTER_NUM_LIMBS] {
    std::array::from_fn(|i| {
        F::from_canonical_u32((value >> (RV32_CELL_BITS * i)) & ((1 << RV32_CELL_BITS) - 1))
    })
}

/// Read write once register.
pub fn unsafe_read_wom_register<F: PrimeField32>(wom: &WomController<F>, pointer: F) -> u32 {
    // WOM reads don't change state for now, so normal read is fine
    let data = wom.read::<RV32_REGISTER_NUM_LIMBS>(pointer).1;
    compose(data)
}

/// Peeks at the value of a register without updating the memory state or incrementing the
/// timestamp.
pub fn unsafe_read_rv32_register<F: PrimeField32>(memory: &MemoryController<F>, pointer: F) -> u32 {
    let data = memory.unsafe_read::<RV32_REGISTER_NUM_LIMBS>(F::ONE, pointer);
    compose(data)
}
