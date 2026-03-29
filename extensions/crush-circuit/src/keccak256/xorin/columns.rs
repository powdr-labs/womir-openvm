use openvm_circuit::system::memory::offline_checker::{MemoryReadAuxCols, MemoryWriteAuxCols};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::riscv::RV32_REGISTER_NUM_LIMBS;
use struct_reflection::{StructReflection, StructReflectionHelper};

use crate::keccak256::{KECCAK_RATE_BYTES, KECCAK_WORD_SIZE};

#[repr(C)]
#[derive(Debug, AlignedBorrow, StructReflection)]
pub struct XorinVmCols<T> {
    pub sponge: XorinSpongeCols<T>,
    pub instruction: XorinInstructionCols<T>,
    pub mem_oc: XorinMemoryCols<T>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, AlignedBorrow, StructReflection, derive_new::new)]
#[allow(clippy::too_many_arguments)]
pub struct XorinInstructionCols<T> {
    pub pc: T,
    /// Frame pointer value (read from FP_AS)
    pub fp: T,
    pub is_enabled: T,
    pub buffer_reg_ptr: T,
    pub input_reg_ptr: T,
    pub len_reg_ptr: T,
    pub buffer_ptr: T,
    pub buffer_ptr_limbs: [T; RV32_REGISTER_NUM_LIMBS],
    pub input_ptr: T,
    pub input_ptr_limbs: [T; RV32_REGISTER_NUM_LIMBS],
    pub len: T,
    pub len_limbs: [T; RV32_REGISTER_NUM_LIMBS],
    pub start_timestamp: T,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, AlignedBorrow, StructReflection)]
pub struct XorinSpongeCols<T> {
    pub is_padding_bytes: [T; KECCAK_RATE_BYTES / KECCAK_WORD_SIZE],
    pub preimage_buffer_bytes: [T; KECCAK_RATE_BYTES],
    pub input_bytes: [T; KECCAK_RATE_BYTES],
    pub postimage_buffer_bytes: [T; KECCAK_RATE_BYTES],
}

#[repr(C)]
#[derive(Clone, Debug, AlignedBorrow, StructReflection)]
pub struct XorinMemoryCols<T> {
    /// FP read aux + 3 register reads = 4 reads
    pub fp_aux: MemoryReadAuxCols<T>,
    pub register_aux_cols: [MemoryReadAuxCols<T>; 3],
    pub input_bytes_read_aux_cols: [MemoryReadAuxCols<T>; KECCAK_RATE_BYTES / KECCAK_WORD_SIZE],
    pub buffer_bytes_read_aux_cols: [MemoryReadAuxCols<T>; KECCAK_RATE_BYTES / KECCAK_WORD_SIZE],
    pub buffer_bytes_write_aux_cols:
        [MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LIMBS>; KECCAK_RATE_BYTES / KECCAK_WORD_SIZE],
}

pub const NUM_XORIN_VM_COLS: usize = size_of::<XorinVmCols<u8>>();
