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
            wom: WomirI::default(),
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
    fn test_basic_mul() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::addi::<F>(8, 0, 666),
            wom::addi::<F>(9, 0, 1),
            wom::mul::<F>(10, 8, 9),
            reveal(10, 0),
            halt(),
        ];

        run_vm_test("Basic multiplication", instructions, 666, None)
    }

    #[test]
    fn test_mul_zero() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::addi::<F>(8, 0, 12345),
            wom::addi::<F>(9, 0, 0),
            wom::mul::<F>(10, 8, 9), // 12345 * 0 = 0
            reveal(10, 0),
            halt(),
        ];
        run_vm_test("Multiplication by zero", instructions, 0, None)
    }

    #[test]
    fn test_mul_one() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::addi::<F>(8, 0, 999),
            wom::addi::<F>(9, 0, 1),
            wom::mul::<F>(10, 8, 9), // 999 * 1 = 999
            reveal(10, 0),
            halt(),
        ];
        run_vm_test("Multiplication by one", instructions, 999, None)
    }

    #[test]
    fn test_mul_powers_of_two() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::addi::<F>(8, 0, 7),
            wom::addi::<F>(9, 0, 8), // 2^3
            wom::mul::<F>(10, 8, 9), // 7 * 8 = 56
            reveal(10, 0),
            halt(),
        ];
        run_vm_test("Multiplication by power of 2", instructions, 56, None)
    }

    #[test]
    fn test_mul_large_numbers() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            // Load large numbers
            wom::const_32_imm::<F>(8, 1, 1), // 65537 = 0x10001 (1 << 16 | 1)
            wom::const_32_imm::<F>(9, 65521, 0), // 65521 = 0xFFF1
            wom::mul::<F>(10, 8, 9),         // 65537 * 65521 = 4,294,836,577
            reveal(10, 0),
            halt(),
        ];
        run_vm_test(
            "Multiplication of large numbers",
            instructions,
            4294049777u32,
            None,
        )
    }

    #[test]
    fn test_mul_overflow() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            // Test multiplication that would overflow 32-bit
            wom::const_32_imm::<F>(8, 0, 1), // 2^16 = 65536 (upper=1, lower=0)
            wom::const_32_imm::<F>(9, 1, 1), // 65537 (upper=1, lower=1)
            wom::mul::<F>(10, 8, 9), // 65536 * 65537 = 4,295,032,832 (overflows to 65536 in 32-bit)
            reveal(10, 0),
            halt(),
        ];
        // In 32-bit arithmetic: 4,295,032,832 & 0xFFFFFFFF = 65536
        run_vm_test("Multiplication with overflow", instructions, 65536, None)
    }

    #[test]
    fn test_mul_commutative() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::addi::<F>(8, 0, 13),
            wom::addi::<F>(9, 0, 17),
            wom::mul::<F>(10, 8, 9),   // 13 * 17 = 221
            wom::mul::<F>(11, 9, 8),   // 17 * 13 = 221 (should be same)
            wom::sub::<F>(12, 10, 11), // Should be 0 if commutative
            reveal(12, 0),
            halt(),
        ];
        run_vm_test("Multiplication commutativity", instructions, 0, None)
    }

    #[test]
    fn test_mul_chain() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::addi::<F>(8, 0, 2),
            wom::addi::<F>(9, 0, 3),
            wom::addi::<F>(10, 0, 5),
            wom::mul::<F>(11, 8, 9),   // 2 * 3 = 6
            wom::mul::<F>(12, 11, 10), // 6 * 5 = 30
            reveal(12, 0),
            halt(),
        ];
        run_vm_test("Chained multiplication", instructions, 30, None)
    }

    #[test]
    fn test_mul_max_value() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            // Test with maximum 32-bit value
            wom::const_32_imm::<F>(8, 0xFFFF, 0xFFFF), // 2^32 - 1
            wom::addi::<F>(9, 0, 1),
            wom::mul::<F>(10, 8, 9), // (2^32 - 1) * 1 = 2^32 - 1
            reveal(10, 0),
            halt(),
        ];
        run_vm_test(
            "Multiplication with max value",
            instructions,
            0xFFFFFFFF,
            None,
        )
    }

    #[test]
    fn test_mul_negative_positive() -> Result<(), Box<dyn std::error::Error>> {
        // Test multiplication of negative and positive numbers
        let instructions = vec![
            wom::const_32_imm::<F>(8, 0xFFFB, 0xFFFF), // -5 in two's complement
            wom::addi::<F>(9, 0, 3),
            wom::mul::<F>(10, 8, 9), // -5 * 3 = -15
            reveal(10, 0),
            halt(),
        ];
        // -15 in 32-bit two's complement is 0xFFFFFFF1
        run_vm_test(
            "Multiplication negative * positive",
            instructions,
            0xFFFFFFF1,
            None,
        )
    }

    #[test]
    fn test_mul_positive_negative() -> Result<(), Box<dyn std::error::Error>> {
        // Test multiplication of positive and negative numbers
        let instructions = vec![
            wom::addi::<F>(8, 0, 4),
            wom::const_32_imm::<F>(9, 0xFFFA, 0xFFFF), // -6 in two's complement
            wom::mul::<F>(10, 8, 9),                   // 4 * -6 = -24
            reveal(10, 0),
            halt(),
        ];
        // -24 in 32-bit two's complement is 0xFFFFFFE8
        run_vm_test(
            "Multiplication positive * negative",
            instructions,
            0xFFFFFFE8,
            None,
        )
    }

    #[test]
    fn test_mul_both_negative() -> Result<(), Box<dyn std::error::Error>> {
        // Test multiplication of two negative numbers
        let instructions = vec![
            wom::const_32_imm::<F>(8, 0xFFF9, 0xFFFF), // -7 in two's complement
            wom::const_32_imm::<F>(9, 0xFFFD, 0xFFFF), // -3 in two's complement
            wom::mul::<F>(10, 8, 9),                   // -7 * -3 = 21
            reveal(10, 0),
            halt(),
        ];
        run_vm_test("Multiplication both negative", instructions, 21, None)
    }

    #[test]
    fn test_mul_negative_one() -> Result<(), Box<dyn std::error::Error>> {
        // Test multiplication by -1
        let instructions = vec![
            wom::addi::<F>(8, 0, 42),
            wom::const_32_imm::<F>(9, 0xFFFF, 0xFFFF), // -1 in two's complement
            wom::mul::<F>(10, 8, 9),                   // 42 * -1 = -42
            reveal(10, 0),
            halt(),
        ];
        // -42 in 32-bit two's complement is 0xFFFFFFD6
        run_vm_test(
            "Multiplication by negative one",
            instructions,
            0xFFFFFFD6,
            None,
        )
    }

    #[test]
    fn test_mul_negative_overflow() -> Result<(), Box<dyn std::error::Error>> {
        // Test multiplication that would overflow with signed numbers
        let instructions = vec![
            wom::const_32_imm::<F>(8, 0x0000, 0x8000), // -2147483648 (INT32_MIN)
            wom::const_32_imm::<F>(9, 0xFFFF, 0xFFFF), // -1
            wom::mul::<F>(10, 8, 9),                   // INT32_MIN * -1 = INT32_MIN (overflow)
            reveal(10, 0),
            halt(),
        ];
        // INT32_MIN * -1 overflows back to INT32_MIN (0x80000000)
        run_vm_test(
            "Multiplication negative overflow",
            instructions,
            0x80000000,
            None,
        )
    }

    #[test]
    fn test_basic_div() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::addi::<F>(8, 0, 100),
            wom::addi::<F>(9, 0, 10),
            wom::div::<F>(10, 8, 9), // 100 / 10 = 10
            reveal(10, 0),
            halt(),
        ];
        run_vm_test("Basic division", instructions, 10, None)
    }

    #[test]
    fn test_div_by_one() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::addi::<F>(8, 0, 999),
            wom::addi::<F>(9, 0, 1),
            wom::div::<F>(10, 8, 9), // 999 / 1 = 999
            reveal(10, 0),
            halt(),
        ];
        run_vm_test("Division by one", instructions, 999, None)
    }

    #[test]
    fn test_div_equal_numbers() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::addi::<F>(8, 0, 42),
            wom::addi::<F>(9, 0, 42),
            wom::div::<F>(10, 8, 9), // 42 / 42 = 1
            reveal(10, 0),
            halt(),
        ];
        run_vm_test("Division of equal numbers", instructions, 1, None)
    }

    #[test]
    fn test_div_with_remainder() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::addi::<F>(8, 0, 17),
            wom::addi::<F>(9, 0, 5),
            wom::div::<F>(10, 8, 9), // 17 / 5 = 3 (integer division)
            reveal(10, 0),
            halt(),
        ];
        run_vm_test("Division with remainder", instructions, 3, None)
    }

    #[test]
    fn test_div_zero_dividend() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::addi::<F>(8, 0, 0),
            wom::addi::<F>(9, 0, 100),
            wom::div::<F>(10, 8, 9), // 0 / 100 = 0
            reveal(10, 0),
            halt(),
        ];
        run_vm_test("Division of zero", instructions, 0, None)
    }

    #[test]
    fn test_div_large_numbers() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::const_32_imm::<F>(8, 0, 1000), // 65536000
            wom::const_32_imm::<F>(9, 256, 0),  // 256
            wom::div::<F>(10, 8, 9),            // 65536000 / 256 = 256000
            reveal(10, 0),
            halt(),
        ];
        run_vm_test("Division of large numbers", instructions, 256000, None)
    }

    #[test]
    fn test_div_powers_of_two() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::addi::<F>(8, 0, 128),
            wom::addi::<F>(9, 0, 8), // 2^3
            wom::div::<F>(10, 8, 9), // 128 / 8 = 16
            reveal(10, 0),
            halt(),
        ];
        run_vm_test("Division by power of 2", instructions, 16, None)
    }

    #[test]
    fn test_div_chain() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::addi::<F>(8, 0, 120),
            wom::addi::<F>(9, 0, 2),
            wom::addi::<F>(10, 0, 3),
            wom::div::<F>(11, 8, 9),   // 120 / 2 = 60
            wom::div::<F>(12, 11, 10), // 60 / 3 = 20
            reveal(12, 0),
            halt(),
        ];
        run_vm_test("Chained division", instructions, 20, None)
    }

    #[test]
    fn test_div_negative_signed() -> Result<(), Box<dyn std::error::Error>> {
        // Testing signed division with negative numbers
        let instructions = vec![
            wom::const_32_imm::<F>(8, 0xFFF6, 0xFFFF), // -10 in two's complement
            wom::addi::<F>(9, 0, 2),
            wom::div::<F>(10, 8, 9), // -10 / 2 = -5
            reveal(10, 0),
            halt(),
        ];
        // -5 in 32-bit two's complement is 0xFFFFFFFB
        run_vm_test(
            "Signed division with negative dividend",
            instructions,
            0xFFFFFFFB,
            None,
        )
    }

    #[test]
    fn test_div_both_negative() -> Result<(), Box<dyn std::error::Error>> {
        // Testing signed division with both numbers negative
        let instructions = vec![
            wom::const_32_imm::<F>(8, 0xFFEC, 0xFFFF), // -20 in two's complement
            wom::const_32_imm::<F>(9, 0xFFFB, 0xFFFF), // -5 in two's complement
            wom::div::<F>(10, 8, 9),                   // -20 / -5 = 4
            reveal(10, 0),
            halt(),
        ];
        run_vm_test("Signed division with both negative", instructions, 4, None)
    }

    #[test]
    fn test_div_and_mul_inverse() -> Result<(), Box<dyn std::error::Error>> {
        // Test that (a / b) * b ≈ a (with integer truncation)
        let instructions = vec![
            wom::addi::<F>(8, 0, 100),
            wom::addi::<F>(9, 0, 7),
            wom::div::<F>(10, 8, 9),  // 100 / 7 = 14
            wom::mul::<F>(11, 10, 9), // 14 * 7 = 98 (not 100 due to truncation)
            reveal(11, 0),
            halt(),
        ];
        run_vm_test(
            "Division and multiplication relationship",
            instructions,
            98,
            None,
        )
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

    #[test]
    fn test_lt_u_true() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::const_32_imm::<F>(8, 100, 0), // Load 100 into x8
            wom::const_32_imm::<F>(9, 200, 0), // Load 200 into x9
            wom::lt_u::<F>(10, 8, 9),          // x10 = (x8 < x9) = (100 < 200) = 1
            reveal(10, 0),
            halt(),
        ];

        run_vm_test("SLTU true test", instructions, 1, None)
    }

    #[test]
    fn test_lt_u_false() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::const_32_imm::<F>(8, 200, 0), // Load 200 into x8
            wom::const_32_imm::<F>(9, 100, 0), // Load 100 into x9
            wom::lt_u::<F>(10, 8, 9),          // x10 = (x8 < x9) = (200 < 100) = 0
            reveal(10, 0),
            halt(),
        ];

        run_vm_test("SLTU false test", instructions, 0, None)
    }

    #[test]
    fn test_lt_u_equal() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::const_32_imm::<F>(8, 150, 0), // Load 150 into x8
            wom::const_32_imm::<F>(9, 150, 0), // Load 150 into x9
            wom::lt_u::<F>(10, 8, 9),          // x10 = (x8 < x9) = (150 < 150) = 0
            reveal(10, 0),
            halt(),
        ];

        run_vm_test("SLTU equal test", instructions, 0, None)
    }

    #[test]
    fn test_lt_s_positive() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::const_32_imm::<F>(8, 50, 0),  // Load 50 into x8
            wom::const_32_imm::<F>(9, 100, 0), // Load 100 into x9
            wom::lt_s::<F>(10, 8, 9),          // x10 = (x8 < x9) = (50 < 100) = 1
            reveal(10, 0),
            halt(),
        ];

        run_vm_test("SLT positive numbers test", instructions, 1, None)
    }

    #[test]
    fn test_lt_s_negative() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::const_32_imm::<F>(8, 0xFFFF, 0xFFFF), // Load -1 into x8
            wom::const_32_imm::<F>(9, 5, 0),           // Load 5 into x9
            wom::lt_s::<F>(10, 8, 9),                  // x10 = (x8 < x9) = (-1 < 5) = 1
            reveal(10, 0),
            halt(),
        ];

        run_vm_test("SLT negative vs positive test", instructions, 1, None)
    }

    #[test]
    fn test_lt_s_both_negative() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::const_32_imm::<F>(8, 0xFFFE, 0xFFFF), // Load -2 into x8
            wom::const_32_imm::<F>(9, 0xFFFC, 0xFFFF), // Load -4 into x9
            wom::lt_s::<F>(10, 8, 9),                  // x10 = (x8 < x9) = (-2 < -4) = 0
            reveal(10, 0),
            halt(),
        ];

        run_vm_test("SLT both negative test", instructions, 0, None)
    }

    #[test]
    fn test_lt_comparison_chain() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::const_32_imm::<F>(8, 10, 0),  // x8 = 10
            wom::const_32_imm::<F>(9, 20, 0),  // x9 = 20
            wom::const_32_imm::<F>(10, 30, 0), // x10 = 30
            wom::lt_u::<F>(11, 8, 9),          // x11 = (10 < 20) = 1
            wom::lt_u::<F>(12, 9, 10),         // x12 = (20 < 30) = 1
            wom::and::<F>(13, 11, 12),         // x13 = x11 & x12 = 1 & 1 = 1
            reveal(13, 0),
            halt(),
        ];

        run_vm_test("Less than comparison chain test", instructions, 1, None)
    }

    #[test]
    fn test_gt_u_true() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::const_32_imm::<F>(8, 200, 0), // Load 200 into x8
            wom::const_32_imm::<F>(9, 100, 0), // Load 100 into x9
            wom::gt_u::<F>(10, 8, 9),          // x10 = (x8 > x9) = (200 > 100) = 1
            reveal(10, 0),
            halt(),
        ];

        run_vm_test("GT_U true test", instructions, 1, None)
    }

    #[test]
    fn test_gt_u_false() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::const_32_imm::<F>(8, 100, 0), // Load 100 into x8
            wom::const_32_imm::<F>(9, 200, 0), // Load 200 into x9
            wom::gt_u::<F>(10, 8, 9),          // x10 = (x8 > x9) = (100 > 200) = 0
            reveal(10, 0),
            halt(),
        ];

        run_vm_test("GT_U false test", instructions, 0, None)
    }

    #[test]
    fn test_gt_u_equal() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::const_32_imm::<F>(8, 150, 0), // Load 150 into x8
            wom::const_32_imm::<F>(9, 150, 0), // Load 150 into x9
            wom::gt_u::<F>(10, 8, 9),          // x10 = (x8 > x9) = (150 > 150) = 0
            reveal(10, 0),
            halt(),
        ];

        run_vm_test("GT_U equal test", instructions, 0, None)
    }

    #[test]
    fn test_gt_s_positive() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::const_32_imm::<F>(8, 100, 0), // Load 100 into x8
            wom::const_32_imm::<F>(9, 50, 0),  // Load 50 into x9
            wom::gt_s::<F>(10, 8, 9),          // x10 = (x8 > x9) = (100 > 50) = 1
            reveal(10, 0),
            halt(),
        ];

        run_vm_test("GT_S positive numbers test", instructions, 1, None)
    }

    #[test]
    fn test_gt_s_negative() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::const_32_imm::<F>(8, 5, 0),           // Load 5 into x8
            wom::const_32_imm::<F>(9, 0xFFFF, 0xFFFF), // Load -1 into x9
            wom::gt_s::<F>(10, 8, 9),                  // x10 = (x8 > x9) = (5 > -1) = 1
            reveal(10, 0),
            halt(),
        ];

        run_vm_test("GT_S positive vs negative test", instructions, 1, None)
    }

    #[test]
    fn test_gt_s_both_negative() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::const_32_imm::<F>(8, 0xFFFE, 0xFFFF), // Load -2 into x8
            wom::const_32_imm::<F>(9, 0xFFFC, 0xFFFF), // Load -4 into x9
            wom::gt_s::<F>(10, 8, 9),                  // x10 = (x8 > x9) = (-2 > -4) = 1
            reveal(10, 0),
            halt(),
        ];

        run_vm_test("GT_S both negative test", instructions, 1, None)
    }

    #[test]
    fn test_gt_edge_cases() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            // Test max unsigned value
            wom::const_32_imm::<F>(8, 0xFFFF, 0xFFFF), // Load 0xFFFFFFFF (max u32) into x8
            wom::const_32_imm::<F>(9, 0, 0),           // Load 0 into x9
            wom::gt_u::<F>(10, 8, 9),                  // x10 = (max > 0) = 1
            // Test with max signed positive
            wom::const_32_imm::<F>(11, 0xFFFF, 0x7FFF), // Load 0x7FFFFFFF (max positive) into x11
            wom::const_32_imm::<F>(12, 0, 0),           // Load 0 into x12
            wom::gt_s::<F>(13, 11, 12),                 // x13 = (max_pos > 0) = 1
            // Combine results
            wom::and::<F>(14, 10, 13), // x14 = 1 & 1 = 1
            reveal(14, 0),
            halt(),
        ];

        run_vm_test("GT edge cases test", instructions, 1, None)
    }

    #[test]
    fn test_comparison_equivalence() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::const_32_imm::<F>(8, 25, 0), // x8 = 25
            wom::const_32_imm::<F>(9, 10, 0), // x9 = 10
            // Test that (a > b) == !(a <= b) == !((a < b) || (a == b))
            wom::gt_u::<F>(10, 8, 9), // x10 = (25 > 10) = 1
            wom::lt_u::<F>(11, 9, 8), // x11 = (10 < 25) = 1 (equivalent)
            // Test that gt_u and lt_u with swapped operands are equivalent
            wom::xor::<F>(12, 10, 11), // x12 = x10 XOR x11 = 1 XOR 1 = 0 (should be 0 if equivalent)
            reveal(12, 0),
            halt(),
        ];

        run_vm_test("Comparison equivalence test", instructions, 0, None)
    }

    #[test]
    fn test_mixed_comparisons() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::const_32_imm::<F>(8, 0xFFFE, 0xFFFF), // x8 = -2 (signed)
            wom::const_32_imm::<F>(9, 2, 0),           // x9 = 2
            // Unsigned comparison: 0xFFFFFFFE > 2
            wom::gt_u::<F>(10, 8, 9), // x10 = 1 (large unsigned > small)
            // Signed comparison: -2 > 2
            wom::gt_s::<F>(11, 8, 9), // x11 = 0 (negative < positive)
            // Show the difference
            wom::sub::<F>(12, 10, 11), // x12 = 1 - 0 = 1
            reveal(12, 0),
            halt(),
        ];

        run_vm_test(
            "Mixed signed/unsigned comparison test",
            instructions,
            1,
            None,
        )
    }

    #[test]
    fn test_input_hint() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::pre_read_u32::<F>(),
            wom::read_u32::<F>(10),
            reveal(10, 0),
            halt(),
        ];
        let mut stdin = StdIn::default();
        stdin.write(&42u32);

        run_vm_test("Input hint", instructions, 42, Some(stdin))
    }

    #[test]
    fn test_input_hint_with_frame_jump_and_xor() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            // Read first value into r8
            wom::pre_read_u32::<F>(),
            wom::read_u32::<F>(8),
            wom::allocate_frame_imm::<F>(9, 64), // Allocate frame, pointer in 99
            wom::copy_into_frame::<F>(2, 8, 9),  // Copy r8 to frame[2]
            // Jump to new frame
            wom::jaaf::<F>(24, 9), // Jump to PC=24, activate frame at r9
            // This should be skipped
            halt(),
            // Read second value into r3
            wom::pre_read_u32::<F>(),
            wom::read_u32::<F>(3),
            // Xor the two read values
            wom::xor::<F>(4, 2, 3),
            // TODO: register 5 below is the absolute value for local register 4 used above,
            // due to the `loadstore` chip not being fp relative yet.
            // The allocated frame is 4 (first allocation). Registers are 4-aligned,
            // so in order to access local reg 4 we need (fp / 4 + local reg) = 4/4+1.
            reveal(5, 0),
            halt(),
        ];

        let mut stdin = StdIn::default();
        stdin.write(&170u32); // First value: 170 in decimal
        stdin.write(&204u32); // Second value: 204 in decimal

        run_vm_test(
            "Input hint with frame jump and XOR",
            instructions,
            102,
            Some(stdin),
        )
    }
}
