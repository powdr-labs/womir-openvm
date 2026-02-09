use openvm_circuit::{arch::MemoryConfig, system::memory::online::GuestMemory};

/// This address space is only used for execution, to store the frame pointer at address 0.
pub const FP_AS: u32 = 5;

/// Same as `MemoryConfig::default()`, but with FP address space.
pub fn memory_config() -> MemoryConfig {
    let mut memory_config = MemoryConfig::default();
    assert_eq!(
        memory_config.addr_spaces[FP_AS as usize].num_cells, 0,
        "FP address space must be unused in default config"
    );
    memory_config.addr_spaces[FP_AS as usize].num_cells = 1;
    memory_config
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
