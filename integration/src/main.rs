use derive_more::From;
use eyre::Result;
use openvm_sdk::{Sdk, StdIn};
use openvm_stark_backend::p3_field::PrimeField32;
use serde::{Deserialize, Serialize};
use std::env::args;
use std::path::Path;

use openvm_circuit::arch::{
    InitFileGenerator, SystemConfig, VmChipComplex, VmConfig, VmInventoryError,
};

use openvm_circuit::circuit_derive::{Chip, ChipUsageGetter};
use openvm_circuit_derive::{AnyEnum, InstructionExecutor};
use openvm_sdk::config::{SdkVmConfig, SdkVmConfigExecutor, SdkVmConfigPeriphery};
type F = openvm_stark_sdk::p3_baby_bear::BabyBear;

mod instruction_builder;
mod instruction_builder_ref;
mod womir_translation;

use openvm_womir_circuit::{self, WomirI, WomirIExecutor, WomirIPeriphery};

#[derive(Serialize, Deserialize, Clone)]
pub struct SpecializedConfig {
    pub sdk_config: SdkVmConfig,
    wom: WomirI,
}

impl SpecializedConfig {
    fn new(sdk_config: SdkVmConfig) -> Self {
        Self {
            sdk_config,
            wom: WomirI,
        }
    }
}

impl InitFileGenerator for SpecializedConfig {
    fn generate_init_file_contents(&self) -> Option<String> {
        self.sdk_config.generate_init_file_contents()
    }

    fn write_to_init_file(
        &self,
        manifest_dir: &Path,
        init_file_name: Option<&str>,
    ) -> eyre::Result<()> {
        self.sdk_config
            .write_to_init_file(manifest_dir, init_file_name)
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(ChipUsageGetter, InstructionExecutor, Chip, From, AnyEnum)]
pub enum SpecializedExecutor<F: PrimeField32> {
    #[any_enum]
    SdkExecutor(SdkVmConfigExecutor<F>),
    #[any_enum]
    WomExecutor(WomirIExecutor<F>),
}

#[derive(From, ChipUsageGetter, Chip, AnyEnum)]
pub enum SpecializedPeriphery<F: PrimeField32> {
    #[any_enum]
    SdkPeriphery(SdkVmConfigPeriphery<F>),
    #[any_enum]
    WomPeriphery(WomirIPeriphery<F>),
}

impl VmConfig<F> for SpecializedConfig {
    type Executor = SpecializedExecutor<F>;
    type Periphery = SpecializedPeriphery<F>;

    fn system(&self) -> &SystemConfig {
        VmConfig::<F>::system(&self.sdk_config)
    }

    fn system_mut(&mut self) -> &mut SystemConfig {
        VmConfig::<F>::system_mut(&mut self.sdk_config)
    }

    fn create_chip_complex(
        &self,
    ) -> Result<VmChipComplex<F, Self::Executor, Self::Periphery>, VmInventoryError> {
        let chip = self.sdk_config.create_chip_complex()?;
        let chip = chip.extend(&self.wom)?;

        Ok(chip)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create VM configuration
    let vm_config = SdkVmConfig::builder()
        .system(Default::default())
        .rv32i(Default::default())
        .rv32m(Default::default())
        .io(Default::default())
        .build();
    let vm_config = SpecializedConfig::new(vm_config);
    let sdk = Sdk::new();

    // Create and execute program
    let mut args = args();
    if args.len() < 3 {
        eprintln!("Usage: {} <wasm_path> <entry_point>", args.next().unwrap());
        return Ok(());
    }
    let wasm_path = args.nth(1).unwrap();
    let entry_point = args.next().unwrap();
    let exe = womir_translation::program_from_wasm::<F>(&wasm_path, &entry_point);
    let stdin = StdIn::default();

    let output = sdk.execute(exe.clone(), vm_config.clone(), stdin.clone())?;
    println!("output: {output:?}");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use instruction_builder as wom;
    use instruction_builder_ref::*;
    use openvm_instructions::{exe::VmExe, instruction::Instruction, program::Program};
    use openvm_sdk::{Sdk, StdIn};
    use openvm_stark_sdk::config::setup_tracing_with_log_level;
    use tracing::Level;

    /// Helper function to run a VM test with given instructions and verify the output
    fn run_vm_test(
        test_name: &str,
        instructions: Vec<Instruction<F>>,
        expected_output: u32,
        stdin: Option<StdIn>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        setup_tracing_with_log_level(Level::WARN);

        // Create VM configuration
        let vm_config = SdkVmConfig::builder()
            .system(Default::default())
            .rv32i(Default::default())
            .rv32m(Default::default())
            .io(Default::default())
            .build();
        let vm_config = SpecializedConfig::new(vm_config);
        let sdk = Sdk::new();

        // Create and execute program
        let program = Program::from_instructions(&instructions);
        let exe = VmExe::new(program);
        let stdin = stdin.unwrap_or_default();

        let output = sdk.execute(exe.clone(), vm_config.clone(), stdin.clone())?;
        println!("{test_name} output: {output:?}");

        // Verify output
        let output_bytes: Vec<_> = output.iter().map(|n| n.as_canonical_u32() as u8).collect();
        let output_0 = u32::from_le_bytes(output_bytes[0..4].try_into().unwrap());
        assert_eq!(
            output_0, expected_output,
            "{test_name} failed: expected {expected_output}, got {output_0}"
        );

        Ok(())
    }

    #[test]
    fn test_basic_wom_operations() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::addi::<F>(8, 0, 666),
            wom::addi::<F>(9, 0, 1),
            wom::add::<F>(10, 8, 9),
            reveal(10, 0),
            halt(),
        ];

        run_vm_test("Basic WOM operations", instructions, 667, None)
    }

    #[test]
    fn test_jaaf_instruction() -> Result<(), Box<dyn std::error::Error>> {
        // Simple test with JAAF instruction
        // We'll set up a value, jump with JAAF, and verify the result
        let instructions = vec![
            wom::addi::<F>(8, 0, 42), // x8 = 42
            wom::addi::<F>(9, 0, 5),  // x9 = 5 (new frame pointer)
            wom::jaaf::<F>(16, 9),    // Jump to PC=16, set FP=x9
            halt(),                   // This should be skipped
            // PC = 16 (byte offset, so instruction at index 4)
            reveal(8, 0), // Reveal x8 (which should still be 42)
            halt(),
        ];

        run_vm_test("JAAF instruction", instructions, 42, None)
    }

    #[test]
    fn test_jaaf_save_instruction() -> Result<(), Box<dyn std::error::Error>> {
        // Test JAAF_SAVE: jump and save FP
        let instructions = vec![
            wom::addi::<F>(8, 0, 99),       // x8 = 99
            wom::addi::<F>(9, 0, 10),       // x9 = 10 (new frame pointer)
            wom::addi::<F>(11, 0, 99),      // x11 = 99 (to show it gets overwritten)
            wom::jaaf_save::<F>(11, 24, 9), // Jump to PC=24, set FP=x9, save old FP to x11
            halt(),                         // This should be skipped
            halt(),                         // This should be skipped too
            // PC = 24 (byte offset, so instruction at index 6)
            reveal(11, 0), // Reveal x11 (should be 0, the old FP)
            halt(),
        ];

        run_vm_test("JAAF_SAVE instruction", instructions, 0, None)
    }

    #[test]
    fn test_ret_instruction() -> Result<(), Box<dyn std::error::Error>> {
        // Test RET: return to saved PC and FP
        let instructions = vec![
            wom::addi::<F>(10, 0, 20), // x10 = 20 (return PC)
            wom::addi::<F>(11, 0, 0),  // x11 = 0 (saved FP)
            wom::addi::<F>(8, 0, 88),  // x8 = 88
            wom::ret::<F>(10, 11),     // Return to PC=x10, FP=x11
            halt(),                    // This should be skipped
            // PC = 20 (where x10 points)
            reveal(8, 0), // Reveal x8 (should be 88)
            halt(),
        ];

        run_vm_test("RET instruction", instructions, 88, None)
    }

    #[test]
    fn test_call_instruction() -> Result<(), Box<dyn std::error::Error>> {
        // Test CALL: save PC and FP, then jump
        let instructions = vec![
            wom::addi::<F>(9, 0, 15),      // x9 = 15 (new FP)
            wom::call::<F>(10, 11, 20, 9), // Call to PC=20, FP=x9, save PC to x10, FP to x11
            wom::addi::<F>(8, 0, 123),     // x8 = 123 (after return) - this should NOT execute
            reveal(8, 0),                  // Reveal x8 - this should NOT execute
            halt(),                        // Padding
            // PC = 20 (function start)
            reveal(10, 0), // Reveal x10 (should be 8, the return address)
            halt(),        // End the test here, don't return
        ];

        run_vm_test("CALL instruction", instructions, 8, None)
    }

    #[test]
    fn test_call_indirect_instruction() -> Result<(), Box<dyn std::error::Error>> {
        // Test CALL_INDIRECT: save PC and FP, jump to register value
        let instructions = vec![
            wom::addi::<F>(12, 0, 28),              // x12 = 28 (target PC)
            wom::addi::<F>(9, 0, 20),               // x9 = 20 (new FP)
            wom::addi::<F>(11, 0, 999),             // x9 = 20 (new FP)
            wom::call_indirect::<F>(10, 11, 12, 9), // Call to PC=x12, FP=x9, save PC to x10, FP to x11
            wom::addi::<F>(8, 0, 456), // x8 = 456 (after return) - this should NOT execute
            reveal(8, 0),              // Reveal x8 - this should NOT execute
            halt(),                    // Padding
            // PC = 28 (function start, where x12 points)
            reveal(11, 0), // Reveal x11 (should be 0, the saved FP)
            halt(),        // End the test here, don't return
        ];

        run_vm_test("CALL_INDIRECT instruction", instructions, 0, None)
    }

    #[test]
    fn test_call_and_return() -> Result<(), Box<dyn std::error::Error>> {
        // Test a complete call and return sequence
        // Note: When FP changes, register addressing changes too
        let instructions = vec![
            wom::addi::<F>(8, 0, 50),      // x8 = 50 (at FP=0)
            wom::addi::<F>(9, 0, 0), // x9 = 0 (new FP for function - using 0 to keep register addressing simple)
            wom::call::<F>(10, 11, 24, 9), // Call function at PC=24, FP=0
            reveal(8, 0),            // Reveal x8 after return (should be 75)
            halt(),
            halt(), // Padding
            // Function at PC = 24
            wom::addi::<F>(8, 8, 25), // x8 = x8 + 25 = 75 (still at FP=0)
            wom::ret::<F>(10, 11),    // Return using saved PC and FP
            halt(),
        ];

        run_vm_test("CALL and RETURN sequence", instructions, 75, None)
    }

    #[test]
    fn test_jump_instruction() -> Result<(), Box<dyn std::error::Error>> {
        // Test unconditional JUMP
        let instructions = vec![
            wom::addi::<F>(8, 0, 42),  // PC=0: x8 = 42
            wom::jump::<F>(20),        // PC=4: Jump to PC=20
            wom::addi::<F>(8, 0, 999), // PC=8: This should be skipped
            reveal(8, 0),              // PC=12: This should be skipped
            halt(),                    // PC=16: Padding
            // PC = 20 (jump target)
            wom::addi::<F>(8, 8, 58), // PC=20: x8 = 42 + 58 = 100
            reveal(8, 0),             // PC=24: Reveal x8 (should be 100)
            halt(),                   // PC=28: End
        ];

        run_vm_test("JUMP instruction", instructions, 100, None)
    }

    #[test]
    fn test_jump_if_instruction() -> Result<(), Box<dyn std::error::Error>> {
        // Test conditional JUMP_IF (condition != 0)
        let instructions = vec![
            wom::addi::<F>(8, 0, 10),  // PC=0: x8 = 10
            wom::addi::<F>(9, 0, 5),   // PC=4: x9 = 5 (condition != 0)
            wom::jump_if::<F>(9, 24),  // PC=8: Jump to PC=24 if x9 != 0 (should jump)
            wom::addi::<F>(8, 0, 999), // PC=12: This should be skipped
            reveal(8, 0),              // PC=16: This should be skipped
            halt(),                    // PC=20: Padding
            // PC = 24 (jump target)
            wom::addi::<F>(8, 8, 15), // PC=24: x8 = 10 + 15 = 25
            reveal(8, 0),             // PC=28: Reveal x8 (should be 25)
            halt(),                   // PC=32: End
        ];

        run_vm_test(
            "JUMP_IF instruction (true condition)",
            instructions,
            25,
            None,
        )
    }

    #[test]
    fn test_jump_if_false_condition() -> Result<(), Box<dyn std::error::Error>> {
        // Test conditional JUMP_IF with false condition (should not jump)
        let instructions = vec![
            wom::addi::<F>(8, 0, 30), // PC=0: x8 = 30
            wom::addi::<F>(9, 0, 0),  // PC=4: x9 = 0 (condition == 0, should not jump)
            wom::jump_if::<F>(9, 28), // PC=8: Jump to PC=28 if x9 != 0 (should NOT jump)
            wom::addi::<F>(8, 8, 20), // PC=12: x8 = 30 + 20 = 50 (this should execute)
            reveal(8, 0),             // PC=16: Reveal x8 (should be 50)
            halt(),                   // PC=20: End
            // PC = 24 (jump target that should not be reached)
            wom::addi::<F>(8, 0, 999), // PC=24: This should not execute
            reveal(8, 0),              // PC=28: This should not execute
            halt(),                    // PC=32: This should not execute
        ];

        run_vm_test(
            "JUMP_IF instruction (false condition)",
            instructions,
            50,
            None,
        )
    }

    #[test]
    fn test_jump_if_zero_instruction() -> Result<(), Box<dyn std::error::Error>> {
        // Test conditional JUMP_IF_ZERO (condition == 0)
        let instructions = vec![
            wom::addi::<F>(8, 0, 77),      // PC=0: x8 = 77
            wom::addi::<F>(9, 0, 0),       // PC=4: x9 = 0 (condition == 0)
            wom::jump_if_zero::<F>(9, 24), // PC=8: Jump to PC=24 if x9 == 0 (should jump)
            wom::addi::<F>(8, 0, 999),     // PC=12: This should be skipped
            reveal(8, 0),                  // PC=16: This should be skipped
            halt(),                        // PC=20: Padding
            // PC = 24 (jump target)
            wom::addi::<F>(8, 8, 23), // PC=24: x8 = 77 + 23 = 100
            reveal(8, 0),             // PC=28: Reveal x8 (should be 100)
            halt(),                   // PC=32: End
        ];

        run_vm_test(
            "JUMP_IF_ZERO instruction (true condition)",
            instructions,
            100,
            None,
        )
    }

    #[test]
    fn test_jump_if_zero_false_condition() -> Result<(), Box<dyn std::error::Error>> {
        // Test conditional JUMP_IF_ZERO with false condition (should not jump)
        let instructions = vec![
            wom::addi::<F>(8, 0, 60),      // PC=0: x8 = 60
            wom::addi::<F>(9, 0, 7),       // PC=4: x9 = 7 (condition != 0, should not jump)
            wom::jump_if_zero::<F>(9, 28), // PC=8: Jump to PC=28 if x9 == 0 (should NOT jump)
            wom::addi::<F>(8, 8, 40),      // PC=12: x8 = 60 + 40 = 100 (this should execute)
            reveal(8, 0),                  // PC=16: Reveal x8 (should be 100)
            halt(),                        // PC=20: End
            // PC = 24 (jump target that should not be reached)
            wom::addi::<F>(8, 0, 999), // PC=24: This should not execute
            reveal(8, 0),              // PC=28: This should not execute
            halt(),                    // PC=32: This should not execute
        ];

        run_vm_test(
            "JUMP_IF_ZERO instruction (false condition)",
            instructions,
            100,
            None,
        )
    }

    #[test]
    fn test_allocate_frame_instruction() -> Result<(), Box<dyn std::error::Error>> {
        // Test ALLOCATE_FRAME instruction
        let instructions = vec![
            wom::allocate_frame_imm::<F>(8, 256), // PC=0: Allocate 256 bytes, store pointer in x8
            reveal(8, 0),                         // PC=4: Reveal x8 (should be allocated pointer)
            halt(),                               // PC=8: End
        ];

        // We expect 4 because the register allocator starts at 4 as convention.
        run_vm_test("ALLOCATE_FRAME instruction", instructions, 4, None)
    }

    #[test]
    fn test_copy_into_frame_instruction() -> Result<(), Box<dyn std::error::Error>> {
        // Test COPY_INTO_FRAME instruction
        // This test verifies that copy_into_frame actually writes to memory
        let instructions = vec![
            wom::addi::<F>(8, 0, 42),            // PC=0: x8 = 42 (value to copy)
            wom::addi::<F>(9, 0, 0x1000),        // PC=4: x9 = 0x1000 (mock frame pointer)
            wom::addi::<F>(10, 0, 0),            // PC=8: x10 = 0 (register to read into)
            wom::copy_into_frame::<F>(10, 8, 9), // PC=12: Copy x8 to [x9[x10]], which writes to address pointed by x10
            wom::jaaf::<F>(20, 9),               // Jump to PC=20, set FP=x9
            // Since copy_into_frame writes x8's value to memory at [x9[x10]],
            // and we activated the frame at x9, x10 should now contain 42.
            // TODO: since `reveal` uses the current loadstore chip that does not take the fp into
            // account, we need to use the absolute register address that we expect.
            // This should go back to `10` once the loadstore chip uses fp.
            reveal(0x1000 / 4 + 10, 0), // PC=20: Reveal x10 (should be 42, the value from x8)
            halt(),                     // PC=24: End
        ];

        run_vm_test("COPY_INTO_FRAME instruction", instructions, 42, None)
    }

    #[test]
    fn test_allocate_and_copy_sequence() -> Result<(), Box<dyn std::error::Error>> {
        // Test sequence: allocate frame, then copy into it
        // This test verifies that copy_into_frame actually writes the value
        let instructions = vec![
            wom::addi::<F>(8, 0, 123),            // PC=0: x8 = 123 (value to store)
            wom::allocate_frame_imm::<F>(9, 128), // PC=4: Allocate 128 bytes, pointer in x9. x9=1
            // by convention on the first allocation.
            wom::addi::<F>(10, 0, 0), // PC=8: x10 = 0 (destination register)
            wom::copy_into_frame::<F>(10, 8, 9), // PC=12: Copy x8 to [x9[x10]]
            // TODO: `reveal` uses the loadstore chip which does not use fp. Change back to 10 once
            // it does.
            reveal(1 + 10, 0), // PC=16: Reveal x10 (should be 123, the value from x8)
            halt(),            // PC=20: End
        ];

        run_vm_test(
            "ALLOCATE_FRAME and COPY_INTO_FRAME sequence",
            instructions,
            123,
            None,
        )
    }

    #[test]
    fn test_const32_simple() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::const_32_imm::<F>(8, 0x1234, 0x5678), // Load 0x56781234 into x8
            reveal(8, 0),
            halt(),
        ];

        run_vm_test("CONST32 simple test", instructions, 0x56781234, None)
    }

    #[test]
    fn test_const32_zero() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::const_32_imm::<F>(10, 0, 0), // Load 0 into x10
            reveal(10, 0),
            halt(),
        ];

        run_vm_test("CONST32 zero test", instructions, 0, None)
    }

    #[test]
    fn test_const32_max_value() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::const_32_imm::<F>(12, 0xFFFF, 0xFFFF), // Load 0xFFFFFFFF into x12
            reveal(12, 0),
            halt(),
        ];

        run_vm_test("CONST32 max value test", instructions, 0xFFFFFFFF, None)
    }

    #[test]
    fn test_const32_multiple_registers() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::const_32_imm::<F>(8, 100, 0), // Load 100 into x8
            wom::const_32_imm::<F>(9, 200, 0), // Load 200 into x9
            wom::add::<F>(11, 8, 9),           // x11 = x8 + x9 = 300
            reveal(11, 0),
            halt(),
        ];

        run_vm_test("CONST32 multiple registers test", instructions, 300, None)
    }

    #[test]
    fn test_const32_with_arithmetic() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::const_32_imm::<F>(8, 1000, 0), // Load 1000 into x8
            wom::const_32_imm::<F>(9, 234, 0),  // Load 234 into x9
            wom::add::<F>(10, 8, 9),            // x10 = x8 + x9 = 1234
            wom::const_32_imm::<F>(11, 34, 0),  // Load 34 into x11
            wom::sub::<F>(12, 10, 11),          // x12 = x10 - x11 = 1200
            reveal(12, 0),
            halt(),
        ];

        run_vm_test("CONST32 with arithmetic test", instructions, 1200, None)
    }
}
