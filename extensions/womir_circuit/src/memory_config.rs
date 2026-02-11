use openvm_circuit::{
    arch::{ADDR_SPACE_OFFSET, DEFAULT_MAX_NUM_PUBLIC_VALUES, MemoryConfig},
    system::memory::{
        POINTER_MAX_BITS, merkle::public_values::PUBLIC_VALUES_AS, online::GuestMemory,
    },
};
use openvm_instructions::riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS};

/// This address space is only used for execution, to store the frame pointer at address 0.
pub const FP_AS: u32 = 5;

/// Same as `MemoryConfig::default()`, but:
/// - Register address space has the same size as the memory address space.
/// - Removed native address space (only needed in recursion).
/// - Added address space for FP. It has one cell to store the current FP value.
pub fn memory_config_with_fp() -> MemoryConfig {
    let mut addr_spaces =
        MemoryConfig::empty_address_space_configs((1 << 3) + ADDR_SPACE_OFFSET as usize);
    const MAX_CELLS: usize = 1 << 29;
    addr_spaces[RV32_REGISTER_AS as usize].num_cells = MAX_CELLS;
    addr_spaces[RV32_MEMORY_AS as usize].num_cells = MAX_CELLS;
    addr_spaces[PUBLIC_VALUES_AS as usize].num_cells = DEFAULT_MAX_NUM_PUBLIC_VALUES;
    addr_spaces[FP_AS as usize].num_cells = size_of::<u32>();
    MemoryConfig::new(3, addr_spaces, POINTER_MAX_BITS, 29, 17, 32)
}

/// Utility trait to read the frame pointer from memory, used in preflight and execution.
pub trait FpMemory {
    fn fp(&self) -> u32;

    #[allow(dead_code)]
    fn set_fp(&mut self, value: u32);
}

impl FpMemory for GuestMemory {
    fn fp(&self) -> u32 {
        unsafe { self.read::<u32, 1>(FP_AS, 0)[0] }
    }

    fn set_fp(&mut self, value: u32) {
        unsafe { self.write::<u32, 1>(FP_AS, 0, [value]) }
    }
}
