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
        eprintln!(
            "Usage: {} <wasm_path> <entry_point> [<32_bit_args>...]",
            args.next().unwrap()
        );
        return Ok(());
    }
    let wasm_path = args.nth(1).unwrap();
    let entry_point = args.next().unwrap();
    let exe = womir_translation::program_from_wasm::<F>(&wasm_path, &entry_point);

    let inputs = args
        .flat_map(|arg| {
            let val = arg.parse::<u32>().unwrap();
            val.to_le_bytes().into_iter()
        })
        .collect::<Vec<_>>();

    let stdin = StdIn::from_bytes(&inputs);

    let output = sdk.execute(exe.clone(), vm_config.clone(), stdin.clone())?;
    println!("output: {output:?}");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use instruction_builder as wom;
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
            wom::reveal(10, 0),
            wom::halt(),
        ];

        run_vm_test("Basic WOM operations", instructions, 667, None)
    }

    #[test]
    fn test_basic_mul() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::addi::<F>(8, 0, 666),
            wom::addi::<F>(9, 0, 1),
            wom::mul::<F>(10, 8, 9),
            wom::reveal(10, 0),
            wom::halt(),
        ];

        run_vm_test("Basic multiplication", instructions, 666, None)
    }

    #[test]
    fn test_mul_zero() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::addi::<F>(8, 0, 12345),
            wom::addi::<F>(9, 0, 0),
            wom::mul::<F>(10, 8, 9), // 12345 * 0 = 0
            wom::reveal(10, 0),
            wom::halt(),
        ];
        run_vm_test("Multiplication by zero", instructions, 0, None)
    }

    #[test]
    fn test_mul_one() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::addi::<F>(8, 0, 999),
            wom::addi::<F>(9, 0, 1),
            wom::mul::<F>(10, 8, 9), // 999 * 1 = 999
            wom::reveal(10, 0),
            wom::halt(),
        ];
        run_vm_test("Multiplication by one", instructions, 999, None)
    }

    #[test]
    fn test_mul_powers_of_two() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::addi::<F>(8, 0, 7),
            wom::addi::<F>(9, 0, 8), // 2^3
            wom::mul::<F>(10, 8, 9), // 7 * 8 = 56
            wom::reveal(10, 0),
            wom::halt(),
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
            wom::reveal(10, 0),
            wom::halt(),
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
            wom::reveal(10, 0),
            wom::halt(),
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
            wom::reveal(12, 0),
            wom::halt(),
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
            wom::reveal(12, 0),
            wom::halt(),
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
            wom::reveal(10, 0),
            wom::halt(),
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
            wom::reveal(10, 0),
            wom::halt(),
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
            wom::reveal(10, 0),
            wom::halt(),
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
            wom::reveal(10, 0),
            wom::halt(),
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
            wom::reveal(10, 0),
            wom::halt(),
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
            wom::reveal(10, 0),
            wom::halt(),
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
            wom::reveal(10, 0),
            wom::halt(),
        ];
        run_vm_test("Basic division", instructions, 10, None)
    }

    #[test]
    fn test_div_by_one() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::addi::<F>(8, 0, 999),
            wom::addi::<F>(9, 0, 1),
            wom::div::<F>(10, 8, 9), // 999 / 1 = 999
            wom::reveal(10, 0),
            wom::halt(),
        ];
        run_vm_test("Division by one", instructions, 999, None)
    }

    #[test]
    fn test_div_equal_numbers() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::addi::<F>(8, 0, 42),
            wom::addi::<F>(9, 0, 42),
            wom::div::<F>(10, 8, 9), // 42 / 42 = 1
            wom::reveal(10, 0),
            wom::halt(),
        ];
        run_vm_test("Division of equal numbers", instructions, 1, None)
    }

    #[test]
    fn test_div_with_remainder() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::addi::<F>(8, 0, 17),
            wom::addi::<F>(9, 0, 5),
            wom::div::<F>(10, 8, 9), // 17 / 5 = 3 (integer division)
            wom::reveal(10, 0),
            wom::halt(),
        ];
        run_vm_test("Division with remainder", instructions, 3, None)
    }

    #[test]
    fn test_div_zero_dividend() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::addi::<F>(8, 0, 0),
            wom::addi::<F>(9, 0, 100),
            wom::div::<F>(10, 8, 9), // 0 / 100 = 0
            wom::reveal(10, 0),
            wom::halt(),
        ];
        run_vm_test("Division of zero", instructions, 0, None)
    }

    #[test]
    fn test_div_large_numbers() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::const_32_imm::<F>(8, 0, 1000), // 65536000
            wom::const_32_imm::<F>(9, 256, 0),  // 256
            wom::div::<F>(10, 8, 9),            // 65536000 / 256 = 256000
            wom::reveal(10, 0),
            wom::halt(),
        ];
        run_vm_test("Division of large numbers", instructions, 256000, None)
    }

    #[test]
    fn test_div_powers_of_two() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::addi::<F>(8, 0, 128),
            wom::addi::<F>(9, 0, 8), // 2^3
            wom::div::<F>(10, 8, 9), // 128 / 8 = 16
            wom::reveal(10, 0),
            wom::halt(),
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
            wom::reveal(12, 0),
            wom::halt(),
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
            wom::reveal(10, 0),
            wom::halt(),
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
            wom::reveal(10, 0),
            wom::halt(),
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
            wom::reveal(11, 0),
            wom::halt(),
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
            wom::addi::<F>(8, 0, 42),            // x8 = 42
            wom::addi::<F>(9, 0, 5),             // x9 = 5 (new frame pointer)
            wom::copy_into_frame::<F>(10, 8, 9), // PC=12: Copy x8 to [x9[x10]], which writes to address pointed by x10
            wom::jaaf::<F>(20, 9),               // Jump to PC=16, set FP=x9
            wom::halt(),                         // This should be skipped
            // PC = 20 (byte offset, so instruction at index 4)
            wom::reveal(10, 0), // wom::reveal x8 (which should still be 42)
            wom::halt(),
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
            wom::halt(),                    // This should be skipped
            wom::halt(),                    // This should be skipped too
            // PC = 24 (byte offset, so instruction at index 6)
            wom::reveal(11, 0), // wom::reveal x11 (should be 0, the old FP)
            wom::halt(),
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
            wom::halt(),               // This should be skipped
            // PC = 20 (where x10 points)
            wom::reveal(8, 0), // wom::reveal x8 (should be 88)
            wom::halt(),
        ];

        run_vm_test("RET instruction", instructions, 88, None)
    }

    #[test]
    fn test_call_instruction() -> Result<(), Box<dyn std::error::Error>> {
        // Test CALL: save PC and FP, then jump
        let instructions = vec![
            wom::addi::<F>(9, 0, 16),      // x9 = 15 (new FP)
            wom::call::<F>(10, 11, 20, 9), // Call to PC=20, FP=x9, save PC to x10, FP to x11
            wom::addi::<F>(8, 0, 123),     // x8 = 123 (after return) - this should NOT execute
            wom::reveal(8, 0),             // wom::reveal x8 - this should NOT execute
            wom::halt(),                   // Padding
            // PC = 20 (function start)
            wom::reveal(10, 0), // wom::reveal x10 (should be 8, the return address)
            wom::halt(),        // End the test here, don't return
        ];

        run_vm_test("CALL instruction", instructions, 8, None)
    }

    #[test]
    fn test_call_indirect_instruction() -> Result<(), Box<dyn std::error::Error>> {
        // Test CALL_INDIRECT: save PC and FP, jump to register value
        let instructions = vec![
            wom::addi::<F>(12, 0, 28),              // x12 = 28 (target PC)
            wom::addi::<F>(9, 0, 20),               // x9 = 20 (new FP)
            wom::addi::<F>(11, 0, 999),             // x11 = 999
            wom::call_indirect::<F>(10, 11, 12, 9), // Call to PC=x12, FP=x9, save PC to x10, FP to x11
            wom::addi::<F>(8, 0, 456), // x8 = 456 (after return) - this should NOT execute
            wom::reveal(8, 0),         // wom::reveal x8 - this should NOT execute
            wom::halt(),               // Padding
            // PC = 28 (function start, where x12 points)
            wom::reveal(5 + 11, 0), // wom::reveal x11 (should be 0, the saved FP)
            wom::halt(),            // End the test here, don't return
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
            wom::reveal(8, 0),       // wom::reveal x8 after return (should be 75)
            wom::halt(),
            wom::halt(), // Padding
            // Function at PC = 24
            wom::addi::<F>(8, 8, 25), // x8 = x8 + 25 = 75 (still at FP=0)
            wom::ret::<F>(10, 11),    // Return using saved PC and FP
            wom::halt(),
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
            wom::reveal(8, 0),         // PC=12: This should be skipped
            wom::halt(),               // PC=16: Padding
            // PC = 20 (jump target)
            wom::addi::<F>(8, 8, 58), // PC=20: x8 = 42 + 58 = 100
            wom::reveal(8, 0),        // PC=24: wom::reveal x8 (should be 100)
            wom::halt(),              // PC=28: End
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
            wom::reveal(8, 0),         // PC=16: This should be skipped
            wom::halt(),               // PC=20: Padding
            // PC = 24 (jump target)
            wom::addi::<F>(8, 8, 15), // PC=24: x8 = 10 + 15 = 25
            wom::reveal(8, 0),        // PC=28: wom::reveal x8 (should be 25)
            wom::halt(),              // PC=32: End
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
            wom::reveal(8, 0),        // PC=16: wom::reveal x8 (should be 50)
            wom::halt(),              // PC=20: End
            // PC = 24 (jump target that should not be reached)
            wom::addi::<F>(8, 0, 999), // PC=24: This should not execute
            wom::reveal(8, 0),         // PC=28: This should not execute
            wom::halt(),               // PC=32: This should not execute
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
            wom::reveal(8, 0),             // PC=16: This should be skipped
            wom::halt(),                   // PC=20: Padding
            // PC = 24 (jump target)
            wom::addi::<F>(8, 8, 23), // PC=24: x8 = 77 + 23 = 100
            wom::reveal(8, 0),        // PC=28: wom::reveal x8 (should be 100)
            wom::halt(),              // PC=32: End
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
            wom::reveal(8, 0),             // PC=16: wom::reveal x8 (should be 100)
            wom::halt(),                   // PC=20: End
            // PC = 24 (jump target that should not be reached)
            wom::addi::<F>(8, 0, 999), // PC=24: This should not execute
            wom::reveal(8, 0),         // PC=28: This should not execute
            wom::halt(),               // PC=32: This should not execute
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
            wom::reveal(8, 0), // PC=4: wom::reveal x8 (should be allocated pointer)
            wom::halt(),       // PC=8: End
        ];

        // We expect 4 because the register allocator starts at 4 as convention.
        run_vm_test("ALLOCATE_FRAME instruction", instructions, 8, None)
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
            wom::reveal(10, 0), // PC=20: wom::reveal x10 (should be 42, the value from x8)
            wom::halt(),        // PC=24: End
        ];

        run_vm_test("COPY_INTO_FRAME instruction", instructions, 42, None)
    }

    #[test]
    fn test_allocate_and_copy_sequence() -> Result<(), Box<dyn std::error::Error>> {
        // Test sequence: allocate frame, then copy into it
        // This test verifies that copy_into_frame actually writes the value
        let instructions = vec![
            wom::addi::<F>(8, 0, 123),            // PC=0: x8 = 123 (value to store)
            wom::allocate_frame_imm::<F>(9, 128), // PC=4: Allocate 128 bytes, pointer in x9. x9=2
            // by convention on the first allocation.
            wom::addi::<F>(10, 0, 0), // PC=8: x10 = 0 (destination register)
            wom::copy_into_frame::<F>(10, 8, 9), // PC=12: Copy x8 to [x9[x10]]
            wom::jaaf::<F>(24, 9),    // Jump to PC=20, set FP=x9
            wom::halt(),              // Should be skipped
            wom::reveal(10, 0),       // PC=24: wom::reveal x10 (should be 123, the value from x8)
            wom::halt(),              // PC=28: End
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
            wom::reveal(8, 0),
            wom::halt(),
        ];

        run_vm_test("CONST32 simple test", instructions, 0x56781234, None)
    }

    #[test]
    fn test_const32_zero() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::const_32_imm::<F>(10, 0, 0), // Load 0 into x10
            wom::reveal(10, 0),
            wom::halt(),
        ];

        run_vm_test("CONST32 zero test", instructions, 0, None)
    }

    #[test]
    fn test_const32_max_value() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::const_32_imm::<F>(12, 0xFFFF, 0xFFFF), // Load 0xFFFFFFFF into x12
            wom::reveal(12, 0),
            wom::halt(),
        ];

        run_vm_test("CONST32 max value test", instructions, 0xFFFFFFFF, None)
    }

    #[test]
    fn test_const32_multiple_registers() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::const_32_imm::<F>(8, 100, 0), // Load 100 into x8
            wom::const_32_imm::<F>(9, 200, 0), // Load 200 into x9
            wom::add::<F>(11, 8, 9),           // x11 = x8 + x9 = 300
            wom::reveal(11, 0),
            wom::halt(),
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
            wom::reveal(12, 0),
            wom::halt(),
        ];

        run_vm_test("CONST32 with arithmetic test", instructions, 1200, None)
    }

    #[test]
    fn test_lt_u_true() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::const_32_imm::<F>(8, 100, 0), // Load 100 into x8
            wom::const_32_imm::<F>(9, 200, 0), // Load 200 into x9
            wom::lt_u::<F>(10, 8, 9),          // x10 = (x8 < x9) = (100 < 200) = 1
            wom::reveal(10, 0),
            wom::halt(),
        ];

        run_vm_test("SLTU true test", instructions, 1, None)
    }

    #[test]
    fn test_lt_u_false() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::const_32_imm::<F>(8, 200, 0), // Load 200 into x8
            wom::const_32_imm::<F>(9, 100, 0), // Load 100 into x9
            wom::lt_u::<F>(10, 8, 9),          // x10 = (x8 < x9) = (200 < 100) = 0
            wom::reveal(10, 0),
            wom::halt(),
        ];

        run_vm_test("SLTU false test", instructions, 0, None)
    }

    #[test]
    fn test_lt_u_equal() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::const_32_imm::<F>(8, 150, 0), // Load 150 into x8
            wom::const_32_imm::<F>(9, 150, 0), // Load 150 into x9
            wom::lt_u::<F>(10, 8, 9),          // x10 = (x8 < x9) = (150 < 150) = 0
            wom::reveal(10, 0),
            wom::halt(),
        ];

        run_vm_test("SLTU equal test", instructions, 0, None)
    }

    #[test]
    fn test_lt_s_positive() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::const_32_imm::<F>(8, 50, 0),  // Load 50 into x8
            wom::const_32_imm::<F>(9, 100, 0), // Load 100 into x9
            wom::lt_s::<F>(10, 8, 9),          // x10 = (x8 < x9) = (50 < 100) = 1
            wom::reveal(10, 0),
            wom::halt(),
        ];

        run_vm_test("SLT positive numbers test", instructions, 1, None)
    }

    #[test]
    fn test_lt_s_negative() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::const_32_imm::<F>(8, 0xFFFF, 0xFFFF), // Load -1 into x8
            wom::const_32_imm::<F>(9, 5, 0),           // Load 5 into x9
            wom::lt_s::<F>(10, 8, 9),                  // x10 = (x8 < x9) = (-1 < 5) = 1
            wom::reveal(10, 0),
            wom::halt(),
        ];

        run_vm_test("SLT negative vs positive test", instructions, 1, None)
    }

    #[test]
    fn test_lt_s_both_negative() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::const_32_imm::<F>(8, 0xFFFE, 0xFFFF), // Load -2 into x8
            wom::const_32_imm::<F>(9, 0xFFFC, 0xFFFF), // Load -4 into x9
            wom::lt_s::<F>(10, 8, 9),                  // x10 = (x8 < x9) = (-2 < -4) = 0
            wom::reveal(10, 0),
            wom::halt(),
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
            wom::reveal(13, 0),
            wom::halt(),
        ];

        run_vm_test("Less than comparison chain test", instructions, 1, None)
    }

    #[test]
    fn test_gt_u_true() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::const_32_imm::<F>(8, 200, 0), // Load 200 into x8
            wom::const_32_imm::<F>(9, 100, 0), // Load 100 into x9
            wom::gt_u::<F>(10, 8, 9),          // x10 = (x8 > x9) = (200 > 100) = 1
            wom::reveal(10, 0),
            wom::halt(),
        ];

        run_vm_test("GT_U true test", instructions, 1, None)
    }

    #[test]
    fn test_gt_u_false() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::const_32_imm::<F>(8, 100, 0), // Load 100 into x8
            wom::const_32_imm::<F>(9, 200, 0), // Load 200 into x9
            wom::gt_u::<F>(10, 8, 9),          // x10 = (x8 > x9) = (100 > 200) = 0
            wom::reveal(10, 0),
            wom::halt(),
        ];

        run_vm_test("GT_U false test", instructions, 0, None)
    }

    #[test]
    fn test_gt_u_equal() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::const_32_imm::<F>(8, 150, 0), // Load 150 into x8
            wom::const_32_imm::<F>(9, 150, 0), // Load 150 into x9
            wom::gt_u::<F>(10, 8, 9),          // x10 = (x8 > x9) = (150 > 150) = 0
            wom::reveal(10, 0),
            wom::halt(),
        ];

        run_vm_test("GT_U equal test", instructions, 0, None)
    }

    #[test]
    fn test_gt_s_positive() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::const_32_imm::<F>(8, 100, 0), // Load 100 into x8
            wom::const_32_imm::<F>(9, 50, 0),  // Load 50 into x9
            wom::gt_s::<F>(10, 8, 9),          // x10 = (x8 > x9) = (100 > 50) = 1
            wom::reveal(10, 0),
            wom::halt(),
        ];

        run_vm_test("GT_S positive numbers test", instructions, 1, None)
    }

    #[test]
    fn test_gt_s_negative() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::const_32_imm::<F>(8, 5, 0),           // Load 5 into x8
            wom::const_32_imm::<F>(9, 0xFFFF, 0xFFFF), // Load -1 into x9
            wom::gt_s::<F>(10, 8, 9),                  // x10 = (x8 > x9) = (5 > -1) = 1
            wom::reveal(10, 0),
            wom::halt(),
        ];

        run_vm_test("GT_S positive vs negative test", instructions, 1, None)
    }

    #[test]
    fn test_gt_s_both_negative() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            wom::const_32_imm::<F>(8, 0xFFFE, 0xFFFF), // Load -2 into x8
            wom::const_32_imm::<F>(9, 0xFFFC, 0xFFFF), // Load -4 into x9
            wom::gt_s::<F>(10, 8, 9),                  // x10 = (x8 > x9) = (-2 > -4) = 1
            wom::reveal(10, 0),
            wom::halt(),
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
            wom::reveal(14, 0),
            wom::halt(),
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
            wom::reveal(12, 0),
            wom::halt(),
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
            wom::reveal(12, 0),
            wom::halt(),
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
            wom::reveal(10, 0),
            wom::halt(),
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
            wom::allocate_frame_imm::<F>(9, 64), // Allocate frame, pointer in r9
            wom::copy_into_frame::<F>(2, 8, 9),  // Copy r8 to frame[2]
            // Jump to new frame
            wom::jaaf::<F>(24, 9), // Jump to PC=24, activate frame at r9
            // This should be skipped
            wom::halt(),
            // Read second value into r3
            wom::pre_read_u32::<F>(),
            wom::read_u32::<F>(3),
            // Xor the two read values
            wom::xor::<F>(4, 2, 3),
            wom::reveal(4, 0),
            wom::halt(),
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

    #[test]
    fn test_loadw_basic() -> Result<(), Box<dyn std::error::Error>> {
        // Test basic LOADW instruction
        let instructions = vec![
            wom::addi::<F>(8, 0, 100), // x8 = 100 (base address)
            wom::addi::<F>(9, 0, 42),  // x9 = 42 (value to store)
            wom::storew::<F>(9, 8, 0), // MEM[x8 + 0] = x9 (store 42 at address 100)
            wom::addi::<F>(10, 0, 0),  // x10 = 0 (clear register)
            wom::loadw::<F>(10, 8, 0), // x10 = MEM[x8 + 0] (load from address 100)
            wom::reveal(10, 0),        // wom::reveal x10 (should be 42)
            wom::halt(),
        ];

        run_vm_test("LOADW basic test", instructions, 42, None)
    }

    #[test]
    fn test_storew_with_offset() -> Result<(), Box<dyn std::error::Error>> {
        // Test STOREW with positive offset
        let instructions = vec![
            wom::addi::<F>(8, 0, 200),  // x8 = 200 (base address)
            wom::addi::<F>(9, 0, 111),  // x9 = 111 (first value)
            wom::addi::<F>(10, 0, 222), // x10 = 222 (second value)
            wom::storew::<F>(9, 8, 0),  // MEM[x8 + 0] = 111
            wom::storew::<F>(10, 8, 4), // MEM[x8 + 4] = 222
            wom::addi::<F>(11, 0, 0),   // x11 = 0 (clear register)
            wom::addi::<F>(12, 0, 0),   // x12 = 0 (clear register)
            wom::loadw::<F>(11, 8, 0),  // x11 = MEM[x8 + 0] (should be 111)
            wom::loadw::<F>(12, 8, 4),  // x12 = MEM[x8 + 4] (should be 222)
            // Test that we loaded the correct values
            wom::add::<F>(13, 11, 12), // x13 = x11 + x12 = 111 + 222 = 333
            wom::reveal(13, 0),        // wom::reveal x13 (should be 333)
            wom::halt(),
        ];

        run_vm_test("STOREW with offset test", instructions, 333, None)
    }

    #[test]
    fn test_loadbu_basic() -> Result<(), Box<dyn std::error::Error>> {
        // Test LOADBU instruction (load byte unsigned)
        let instructions = vec![
            wom::addi::<F>(8, 0, 300),  // x8 = 300 (base address)
            wom::addi::<F>(9, 0, 0xFF), // x9 = 255 (max byte value)
            wom::storeb::<F>(9, 8, 0),  // MEM[x8 + 0] = 255 (store as byte)
            wom::addi::<F>(10, 0, 0),   // x10 = 0 (clear register)
            wom::loadbu::<F>(10, 8, 0), // x10 = MEM[x8 + 0] (load byte unsigned)
            wom::reveal(10, 0),         // Reveal x10 (should be 255)
            wom::halt(),
        ];
        run_vm_test("LOADBU basic test", instructions, 255, None)
    }

    #[test]
    fn test_loadhu_basic() -> Result<(), Box<dyn std::error::Error>> {
        // Test LOADHU instruction (load halfword unsigned)
        let instructions = vec![
            wom::addi::<F>(8, 0, 400),            // x8 = 400 (base address)
            wom::const_32_imm::<F>(9, 0xABCD, 0), // x9 = 0xABCD (43981)
            wom::storeh::<F>(9, 8, 0),            // MEM[x8 + 0] = 0xABCD (store as halfword)
            wom::addi::<F>(10, 0, 0),             // x10 = 0 (clear register)
            wom::loadhu::<F>(10, 8, 0),           // x10 = MEM[x8 + 0] (load halfword unsigned)
            wom::reveal(10, 0),                   // Reveal x10 (should be 0xABCD = 43981)
            wom::halt(),
        ];
        run_vm_test("LOADHU basic test", instructions, 0xABCD, None)
    }

    #[test]
    fn test_storeb_with_offset() -> Result<(), Box<dyn std::error::Error>> {
        // Test STOREB with offset and masking
        let instructions = vec![
            wom::addi::<F>(8, 0, 500),            // x8 = 500 (base address)
            wom::const_32_imm::<F>(9, 0x1234, 0), // x9 = 0x1234 (only lowest byte 0x34 will be stored)
            wom::storeb::<F>(9, 8, 0),            // MEM[x8 + 0] = 0x34 (store lowest byte)
            wom::storeb::<F>(9, 8, 1),            // MEM[x8 + 1] = 0x34 (store at offset 1)
            wom::addi::<F>(10, 0, 0),             // x10 = 0
            wom::addi::<F>(11, 0, 0),             // x11 = 0
            wom::loadbu::<F>(10, 8, 0),           // x10 = MEM[x8 + 0] (should be 0x34 = 52)
            wom::loadbu::<F>(11, 8, 1),           // x11 = MEM[x8 + 1] (should be 0x34 = 52)
            wom::add::<F>(12, 10, 11),            // x12 = x10 + x11 = 52 + 52 = 104
            wom::reveal(12, 0),                   // Reveal x12 (should be 104)
            wom::halt(),
        ];
        run_vm_test("STOREB with offset test", instructions, 104, None)
    }

    #[test]
    fn test_storeh_with_offset() -> Result<(), Box<dyn std::error::Error>> {
        // Test STOREH with offset
        let instructions = vec![
            wom::addi::<F>(8, 0, 600),             // x8 = 600 (base address)
            wom::const_32_imm::<F>(9, 0x1111, 0),  // x9 = 0x1111
            wom::const_32_imm::<F>(10, 0x2222, 0), // x10 = 0x2222
            wom::storeh::<F>(9, 8, 0),             // MEM[x8 + 0] = 0x1111 (store halfword)
            wom::storeh::<F>(10, 8, 2),            // MEM[x8 + 2] = 0x2222 (store at offset 2)
            wom::addi::<F>(11, 0, 0),              // x11 = 0
            wom::addi::<F>(12, 0, 0),              // x12 = 0
            wom::loadhu::<F>(11, 8, 0),            // x11 = MEM[x8 + 0] (should be 0x1111 = 4369)
            wom::loadhu::<F>(12, 8, 2),            // x12 = MEM[x8 + 2] (should be 0x2222 = 8738)
            wom::add::<F>(13, 11, 12),             // x13 = 4369 + 8738 = 13107
            wom::reveal(13, 0),                    // Reveal x13 (should be 13107)
            wom::halt(),
        ];
        run_vm_test("STOREH with offset test", instructions, 13107, None)
    }
}

#[cfg(test)]
mod wast_tests {
    use super::*;
    use openvm_sdk::{Sdk, StdIn};
    use openvm_stark_sdk::config::setup_tracing_with_log_level;
    use serde::Deserialize;
    use serde_json::Value;
    use std::fs;
    use std::path::Path;
    use std::process::Command;
    use tracing::Level;

    type TestCase = (String, Vec<u32>, Vec<u32>);
    type TestModule = (String, u32, Vec<TestCase>);

    #[derive(Debug, Deserialize)]
    struct TestFile {
        commands: Vec<CommandEntry>,
    }

    #[derive(Debug, Deserialize)]
    struct CommandEntry {
        #[serde(rename = "type")]
        cmd_type: String,
        filename: Option<String>,
        line: Option<u32>,
        action: Option<Action>,
        expected: Option<Vec<Expected>>,
    }

    #[derive(Debug, Deserialize)]
    struct Action {
        #[serde(rename = "type")]
        action_type: String,
        field: Option<String>,
        args: Option<Vec<Value>>,
        #[allow(dead_code)]
        module: Option<String>,
    }

    #[derive(Debug, Deserialize)]
    struct Expected {
        #[serde(rename = "type")]
        #[allow(dead_code)]
        expected_type: String,
        #[allow(dead_code)]
        lane: Option<String>,
        value: Option<String>,
    }

    fn extract_wast_test_info(
        wast_file: &str,
    ) -> Result<Vec<TestModule>, Box<dyn std::error::Error>> {
        // Convert .wast to .json using wast2json
        let wast_path = Path::new(wast_file);
        let json_path = wast_path.with_extension("json");
        let _output_dir = wast_path.parent().unwrap_or(Path::new("."));

        let output = Command::new("wast2json")
            .arg(wast_file)
            .arg("-o")
            .arg(&json_path)
            .arg("--debug-names")
            .output()?;

        if !output.status.success() {
            return Err(format!(
                "wast2json failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )
            .into());
        }

        // Parse the JSON file
        let json_content = fs::read_to_string(&json_path)?;
        let test_file: TestFile = serde_json::from_str(&json_content)?;

        let mut test_cases = Vec::new();
        let mut current_module = None;
        let mut current_line = 0;
        let mut assert_cases = Vec::new();

        for cmd in test_file.commands {
            match cmd.cmd_type.as_str() {
                "module" => {
                    if let Some(module) = current_module.take() {
                        if !assert_cases.is_empty() {
                            test_cases.push((module, current_line, assert_cases.clone()));
                            assert_cases.clear();
                        }
                    }
                    current_module = cmd.filename;
                    current_line = cmd.line.unwrap_or(0);
                }
                "assert_return" => {
                    if let (Some(action), Some(expected)) = (cmd.action, cmd.expected) {
                        if action.action_type == "invoke" {
                            if let (Some(field), Some(args)) = (action.field, action.args) {
                                let args_u32: Vec<u32> = args
                                    .iter()
                                    .filter_map(|v| {
                                        if let Value::Object(obj) = v {
                                            if let Some(Value::String(val_str)) = obj.get("value") {
                                                val_str.parse::<u32>().ok()
                                            } else {
                                                None
                                            }
                                        } else {
                                            None
                                        }
                                    })
                                    .collect();

                                let expected_u32: Vec<u32> = expected
                                    .iter()
                                    .filter_map(|e| {
                                        e.value.as_ref().and_then(|v| v.parse::<u32>().ok())
                                    })
                                    .collect();

                                assert_cases.push((field, args_u32, expected_u32));
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        if let Some(module) = current_module {
            if !assert_cases.is_empty() {
                test_cases.push((module, current_line, assert_cases));
            }
        }

        // Clean up JSON file
        let _ = fs::remove_file(&json_path);

        Ok(test_cases)
    }

    #[allow(dead_code)]
    fn parse_val(s: &str) -> Result<u32, Box<dyn std::error::Error>> {
        if s.starts_with("i32.const ") {
            let val_str = s.trim_start_matches("i32.const ").trim();
            if val_str.starts_with("0x") {
                u32::from_str_radix(val_str.trim_start_matches("0x"), 16).map_err(|e| e.into())
            } else if val_str.starts_with("-0x") {
                u32::from_str_radix(val_str.trim_start_matches("-0x"), 16)
                    .map(|v| (!v).wrapping_add(1))
                    .map_err(|e| e.into())
            } else if val_str.starts_with("-") {
                val_str
                    .parse::<i32>()
                    .map(|v| v as u32)
                    .map_err(|e| e.into())
            } else {
                val_str.parse::<u32>().map_err(|e| e.into())
            }
        } else {
            Err("Unsupported value format".into())
        }
    }

    fn run_single_wast_test(
        module_path: &str,
        function: &str,
        args: &[u32],
        expected: &[u32],
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

        // Load and execute the module
        let exe = womir_translation::program_from_wasm::<F>(module_path, function);

        // Prepare input
        let mut stdin = StdIn::default();
        for &arg in args {
            stdin.write(&arg);
        }

        let output = sdk.execute(exe.clone(), vm_config.clone(), stdin.clone())?;

        // Verify output
        if !expected.is_empty() {
            let output_bytes: Vec<_> = output.iter().map(|n| n.as_canonical_u32() as u8).collect();
            let output_0 = u32::from_le_bytes(output_bytes[0..4].try_into().unwrap());
            assert_eq!(
                output_0, expected[0],
                "Test failed for {function}({args:?}): expected {expected:?}, got {output_0:?}"
            );
        }

        Ok(())
    }

    #[test]
    fn test_i32() -> Result<(), Box<dyn std::error::Error>> {
        // Load test cases
        let test_cases = extract_wast_test_info("../wasm_tests/i32.wast")?;

        // Run all test cases
        for (module_path, _line, cases) in &test_cases {
            // Prepend ../ to the module path since we're running from integration directory
            let full_module_path = format!("../wasm_tests/{module_path}");

            for (function, args, expected) in cases {
                run_single_wast_test(&full_module_path, function, args, expected)?;
            }
        }

        Ok(())
    }
}
