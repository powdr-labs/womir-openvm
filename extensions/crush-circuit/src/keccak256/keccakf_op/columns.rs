use openvm_circuit::system::memory::offline_checker::{MemoryBaseAuxCols, MemoryReadAuxCols};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::riscv::RV32_REGISTER_NUM_LIMBS;
use struct_reflection::{StructReflection, StructReflectionHelper};

use crate::keccak256::{KECCAK_WIDTH_BYTES, KECCAK_WIDTH_WORDS};

#[repr(C)]
#[derive(Copy, Clone, Debug, AlignedBorrow, StructReflection)]
pub struct KeccakfOpCols<T> {
    /// Program counter
    pub pc: T,
    /// Frame pointer value (read from FP_AS)
    pub fp: T,
    /// True on the row handling execution for an instruction.
    pub is_valid: T,
    /// The starting timestamp for execution in this row.
    /// A single row will do multiple memory accesses.
    pub timestamp: T,
    /// Pointer to address space 1 `rd` register (instruction operand, before FP offset).
    pub rd_ptr: T,
    /// `buffer_ptr <- [fp + rd_ptr:4]_1`.
    /// Limbs of the pointer to address space 2 `buffer`.
    pub buffer_ptr_limbs: [T; RV32_REGISTER_NUM_LIMBS],
    /// The preimage state, to be permuted in the `keccakf` operation.
    pub preimage: [T; KECCAK_WIDTH_BYTES],
    /// The postimage state after `keccakf` permute of `preimage`.
    pub postimage: [T; KECCAK_WIDTH_BYTES],
    /// Auxiliary columns for timestamp checking for the read of FP from FP_AS.
    pub fp_aux: MemoryReadAuxCols<T>,
    /// Auxiliary columns for timestamp checking for the read of `[fp+rd_ptr:4]_1`.
    pub rd_aux: MemoryReadAuxCols<T>,
    /// Auxiliary columns for timestamp checking of the writes to `buffer`. The writes are done one
    /// word at a time, and each write requires a separate previous timestamp.
    pub buffer_word_aux: [MemoryBaseAuxCols<T>; KECCAK_WIDTH_WORDS],
}

pub const NUM_KECCAKF_OP_COLS: usize = size_of::<KeccakfOpCols<u8>>();
