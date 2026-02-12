use openvm_circuit::{
    arch::{
        ADDR_SPACE_OFFSET, AddressSpaceHostConfig, DEFAULT_MAX_NUM_PUBLIC_VALUES, MemoryCellType,
        MemoryConfig,
    },
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
/// - Added address space for FP. It stores the FP value as 4 bytes (U8 cells).
pub fn memory_config_with_fp() -> MemoryConfig {
    let mut addr_spaces =
        MemoryConfig::empty_address_space_configs((1 << 3) + ADDR_SPACE_OFFSET as usize);
    const MAX_CELLS: usize = 1 << 29;
    addr_spaces[RV32_REGISTER_AS as usize].num_cells = MAX_CELLS;
    addr_spaces[RV32_MEMORY_AS as usize].num_cells = MAX_CELLS;
    addr_spaces[PUBLIC_VALUES_AS as usize].num_cells = DEFAULT_MAX_NUM_PUBLIC_VALUES;
    // FP_AS uses U8 cell type (like registers), with 4 cells to store FP as a u32.
    // Using U8 cells with min_block_size=4 makes FP_AS compatible with the memory system
    // for both volatile and persistent memory modes.
    addr_spaces[FP_AS as usize] = AddressSpaceHostConfig::new(4, 4, MemoryCellType::U8);
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
        // FP_AS uses U8 cells, so we read 4 bytes and convert to u32
        let bytes: [u8; 4] = unsafe { self.read(FP_AS, 0) };
        u32::from_le_bytes(bytes)
    }

    fn set_fp(&mut self, value: u32) {
        // FP_AS uses U8 cells, so we write 4 bytes
        unsafe { self.write::<u8, 4>(FP_AS, 0, value.to_le_bytes()) }
    }
}
