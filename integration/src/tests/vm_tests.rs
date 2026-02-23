use crate::womir_translation::ERROR_CODE_OFFSET;
use crate::*;
use instruction_builder as wom;
use openvm_circuit::{
    arch::{ExecutionError, VmExecutor},
    system::memory::merkle::public_values::extract_public_values,
};
use openvm_instructions::{exe::VmExe, instruction::Instruction, program::Program};
use openvm_sdk::{
    StdIn,
    config::{AppConfig, DEFAULT_APP_LOG_BLOWUP},
    keygen::AppProvingKey,
    prover::AppProver,
};
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_stark_sdk::config::{FriParameters, baby_bear_poseidon2::BabyBearPoseidon2Engine};
use tracing::Level;
use womir_circuit::WomirCpuBuilder;

/// Helper function to run a VM test with given instructions and return the error or
/// verify the output on success.
fn run_vm_test_with_result(
    test_name: &str,
    instructions: Vec<Instruction<F>>,
    expected_output: u32,
    stdin: Option<StdIn>,
) -> Result<(), ExecutionError> {
    setup_tracing_with_log_level(Level::WARN);

    // Create and execute program
    let program = Program::from_instructions(&instructions);
    let exe = VmExe::new(program);
    let stdin = stdin.unwrap_or_default();

    let vm_config = WomirConfig::default();
    let vm = VmExecutor::new(vm_config.clone()).unwrap();
    let instance = vm.instance(&exe).unwrap();
    let final_state = instance.execute(stdin, None)?;
    let output = extract_public_values(
        vm_config.system.num_public_values,
        &final_state.memory.memory,
    );

    println!("{test_name} output: {output:?}");

    // Verify output
    let output_0 = u32::from_le_bytes(output[0..4].try_into().unwrap());
    assert_eq!(
        output_0, expected_output,
        "{test_name} failed: expected {expected_output}, got {output_0}"
    );

    Ok(())
}

/// Helper function to run a VM test with given instructions and verify the output
fn run_vm_test(
    test_name: &str,
    instructions: Vec<Instruction<F>>,
    expected_output: u32,
    stdin: Option<StdIn>,
) -> Result<(), Box<dyn std::error::Error>> {
    run_vm_test_with_result(test_name, instructions, expected_output, stdin)?;
    Ok(())
}

fn run_vm_test_proof_with_result(
    test_name: &str,
    instructions: Vec<Instruction<F>>,
    expected_output: u32,
    stdin: Option<StdIn>,
) -> Result<(), ExecutionError> {
    setup_tracing_with_log_level(Level::WARN);

    // Create and execute program
    let program = Program::from_instructions(&instructions);
    let exe = VmExe::new(program);
    let stdin = stdin.unwrap_or_default();

    let vm_config = WomirConfig::default();

    // Set app configuration
    let app_fri_params =
        FriParameters::standard_with_100_bits_conjectured_security(DEFAULT_APP_LOG_BLOWUP);
    let app_config = AppConfig::new(app_fri_params, vm_config.clone());
    let app_pk = AppProvingKey::keygen(app_config.clone()).expect("app_keygen failed");

    let mut app_prover = AppProver::<BabyBearPoseidon2Engine, WomirCpuBuilder>::new(
        WomirCpuBuilder,
        &app_pk.app_vm_pk,
        exe.clone().into(),
        app_pk.leaf_verifier_program_commit(),
    )
    .expect("app_prover failed");

    tracing::info!("Generating app proof...");
    let start = std::time::Instant::now();
    let app_proof = app_prover.prove(stdin.clone()).expect("App proof failed");
    tracing::info!("App proof took {:?}", start.elapsed());

    tracing::info!("Public values: {:?}", app_proof.user_public_values);

    let output = app_proof.user_public_values.public_values;

    println!("{test_name} output: {output:?}");

    // Verify output - convert field elements to bytes
    let output_bytes: Vec<u8> = output.iter().map(|f| f.as_canonical_u32() as u8).collect();
    let output_0 = u32::from_le_bytes(output_bytes[0..4].try_into().unwrap());
    // TODO bring this back once LoadStore is supported properly for proofs.
    assert_eq!(
        output_0, expected_output,
        "{test_name} failed: expected {expected_output}, got {output_0}"
    );

    Ok(())
}

fn run_vm_test_proof(
    test_name: &str,
    instructions: Vec<Instruction<F>>,
    expected_output: u32,
    stdin: Option<StdIn>,
) -> Result<(), Box<dyn std::error::Error>> {
    run_vm_test_proof_with_result(test_name, instructions, expected_output, stdin)?;
    Ok(())
}

#[test]
fn test_basic_add() -> Result<(), Box<dyn std::error::Error>> {
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::add_imm(8, 0, 666_i16),
        wom::add_imm(9, 0, 1_i16),
        wom::add(10, 8, 9),
        wom::reveal(10, 0),
        wom::halt(),
    ];

    run_vm_test("Basic WOM operations", instructions, 667, None)
}

#[test]
fn test_basic_add_proof() -> Result<(), Box<dyn std::error::Error>> {
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::add_imm(8, 0, 666_i16),
        wom::add_imm(9, 0, 1_i16),
        wom::add(10, 8, 9),
        wom::reveal(10, 0),
        wom::halt(),
    ];

    run_vm_test_proof("Basic WOM operations", instructions, 667, None)
}

#[test]
fn test_basic_wom_operations() -> Result<(), Box<dyn std::error::Error>> {
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::add_imm(8, 0, 666_i16),
        wom::add_imm(9, 0, 1_i16),
        wom::add(10, 8, 9),
        wom::reveal(10, 0),
        wom::halt(),
    ];

    run_vm_test("Basic WOM operations", instructions, 667, None)
}

#[test]
fn test_trap() -> Result<(), Box<dyn std::error::Error>> {
    let instructions = vec![wom::trap(42), wom::trap(8), wom::halt()];

    let err = run_vm_test_with_result("Trap instruction", instructions, 0, None).unwrap_err();
    if let ExecutionError::FailedWithExitCode(code) = err {
        assert_eq!(code, ERROR_CODE_OFFSET + 42);
    } else {
        panic!("Unexpected error: {err:?}");
    }
    Ok(())
}

#[test]
fn test_basic_addi_64() {
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::const_32_imm(1, 0, 0),
        wom::add_imm_64(8, 0, 666_i16),
        wom::add_imm_64(10, 8, 1_i16),
        wom::reveal(10, 0),
        wom::halt(),
    ];

    run_vm_test("Basic addi_64", instructions, 667, None).unwrap()
}

#[test]
fn test_basic_mul() {
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::add_imm(8, 0, 666_i16),
        wom::add_imm(9, 0, 1_i16),
        wom::mul(10, 8, 9),
        wom::reveal(10, 0),
        wom::halt(),
    ];

    run_vm_test("Basic multiplication", instructions, 666, None).unwrap()
}

#[test]
fn test_mul_zero() {
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::add_imm(8, 0, 12345_i16),
        wom::add_imm(9, 0, 0_i16),
        wom::mul(10, 8, 9), // 12345 * 0 = 0
        wom::reveal(10, 0),
        wom::halt(),
    ];
    run_vm_test("Multiplication by zero", instructions, 0, None).unwrap()
}

#[test]
fn test_mul_one() {
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::add_imm(8, 0, 999_i16),
        wom::add_imm(9, 0, 1_i16),
        wom::mul(10, 8, 9), // 999 * 1 = 999
        wom::reveal(10, 0),
        wom::halt(),
    ];
    run_vm_test("Multiplication by one", instructions, 999, None).unwrap()
}

#[test]
fn test_skip() {
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        // Sets to skip 5 instructions.
        wom::const_32_imm(8, 5, 0),
        wom::skip(8),
        //// SKIPPED BLOCK ////
        wom::halt(),
        wom::const_32_imm(10, 666, 0),
        wom::reveal(10, 0),
        wom::halt(),
        wom::halt(),
        ///////////////////////
        wom::const_32_imm(10, 42, 0),
        wom::reveal(10, 0),
        wom::halt(),
    ];
    run_vm_test("Skipping 5 instructions", instructions, 42, None).unwrap()
}

#[test]
fn test_mul_powers_of_two() {
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::add_imm(8, 0, 7_i16),
        wom::add_imm(9, 0, 8_i16), // 2^3
        wom::mul(10, 8, 9),        // 7 * 8 = 56
        wom::reveal(10, 0),
        wom::halt(),
    ];
    run_vm_test("Multiplication by power of 2", instructions, 56, None).unwrap()
}

#[test]
fn test_mul_large_numbers() {
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        // Load large numbers
        wom::const_32_imm(8, 1, 1),     // 65537 = 0x10001 (1 << 16 | 1)
        wom::const_32_imm(9, 65521, 0), // 65521 = 0xFFF1
        wom::mul(10, 8, 9),             // 65537 * 65521 = 4,294,836,577
        wom::reveal(10, 0),
        wom::halt(),
    ];
    run_vm_test(
        "Multiplication of large numbers",
        instructions,
        4294049777u32,
        None,
    )
    .unwrap()
}

#[test]
fn test_mul_overflow() {
    let instructions = vec![
        // Test multiplication that would overflow 32-bit
        wom::const_32_imm(0, 0, 0),
        wom::const_32_imm(8, 0, 1), // 2^16 = 65536 (upper=1, lower=0)
        wom::const_32_imm(9, 1, 1), // 65537 (upper=1, lower=1)
        wom::mul(10, 8, 9),         // 65536 * 65537 = 4,295,032,832 (overflows to 65536 in 32-bit)
        wom::reveal(10, 0),
        wom::halt(),
    ];
    // In 32-bit arithmetic: 4,295,032,832 & 0xFFFFFFFF = 65536
    run_vm_test("Multiplication with overflow", instructions, 65536, None).unwrap()
}

#[test]
fn test_mul_commutative() {
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::add_imm(8, 0, 13_i16),
        wom::add_imm(9, 0, 17_i16),
        wom::mul(10, 8, 9),   // 13 * 17 = 221
        wom::mul(11, 9, 8),   // 17 * 13 = 221 (should be same)
        wom::sub(12, 10, 11), // Should be 0 if commutative
        wom::reveal(12, 0),
        wom::halt(),
    ];
    run_vm_test("Multiplication commutativity", instructions, 0, None).unwrap()
}

#[test]
fn test_mul_chain() {
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::add_imm(8, 0, 2_i16),
        wom::add_imm(9, 0, 3_i16),
        wom::add_imm(10, 0, 5_i16),
        wom::mul(11, 8, 9),   // 2 * 3 = 6
        wom::mul(12, 11, 10), // 6 * 5 = 30
        wom::reveal(12, 0),
        wom::halt(),
    ];
    run_vm_test("Chained multiplication", instructions, 30, None).unwrap()
}

#[test]
fn test_mul_max_value() {
    let instructions = vec![
        // Test with maximum 32-bit value
        wom::const_32_imm(0, 0, 0),
        wom::const_32_imm(8, 0xFFFF, 0xFFFF), // 2^32 - 1
        wom::add_imm(9, 0, 1_i16),
        wom::mul(10, 8, 9), // (2^32 - 1) * 1 = 2^32 - 1
        wom::reveal(10, 0),
        wom::halt(),
    ];
    run_vm_test(
        "Multiplication with max value",
        instructions,
        0xFFFFFFFF,
        None,
    )
    .unwrap()
}

#[test]
fn test_mul_negative_positive() {
    // Test multiplication of negative and positive numbers
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::const_32_imm(8, 0xFFFB, 0xFFFF), // -5 in two's complement
        wom::add_imm(9, 0, 3_i16),
        wom::mul(10, 8, 9), // -5 * 3 = -15
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
    .unwrap()
}

#[test]
fn test_mul_positive_negative() {
    // Test multiplication of positive and negative numbers
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::add_imm(8, 0, 4_i16),
        wom::const_32_imm(9, 0xFFFA, 0xFFFF), // -6 in two's complement
        wom::mul(10, 8, 9),                   // 4 * -6 = -24
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
    .unwrap()
}

#[test]
fn test_mul_both_negative() {
    // Test multiplication of two negative numbers
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::const_32_imm(8, 0xFFF9, 0xFFFF), // -7 in two's complement
        wom::const_32_imm(9, 0xFFFD, 0xFFFF), // -3 in two's complement
        wom::mul(10, 8, 9),                   // -7 * -3 = 21
        wom::reveal(10, 0),
        wom::halt(),
    ];
    run_vm_test("Multiplication both negative", instructions, 21, None).unwrap()
}

#[test]
fn test_mul_negative_one() {
    // Test multiplication by -1
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::add_imm(8, 0, 42_i16),
        wom::const_32_imm(9, 0xFFFF, 0xFFFF), // -1 in two's complement
        wom::mul(10, 8, 9),                   // 42 * -1 = -42
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
    .unwrap()
}

#[test]
fn test_mul_negative_overflow() {
    // Test multiplication that would overflow with signed numbers
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::const_32_imm(8, 0x0000, 0x8000), // -2147483648 (INT32_MIN)
        wom::const_32_imm(9, 0xFFFF, 0xFFFF), // -1
        wom::mul(10, 8, 9),                   // INT32_MIN * -1 = INT32_MIN (overflow)
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
    .unwrap()
}

#[test]
fn test_basic_div() {
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::add_imm(8, 0, 100_i16),
        wom::add_imm(9, 0, 10_i16),
        wom::div(10, 8, 9), // 100 / 10 = 10
        wom::reveal(10, 0),
        wom::halt(),
    ];
    run_vm_test("Basic division", instructions, 10, None).unwrap()
}

#[test]
fn test_div_by_one() {
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::add_imm(8, 0, 999_i16),
        wom::add_imm(9, 0, 1_i16),
        wom::div(10, 8, 9), // 999 / 1 = 999
        wom::reveal(10, 0),
        wom::halt(),
    ];
    run_vm_test("Division by one", instructions, 999, None).unwrap()
}

#[test]
fn test_div_equal_numbers() {
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::add_imm(8, 0, 42_i16),
        wom::add_imm(9, 0, 42_i16),
        wom::div(10, 8, 9), // 42 / 42 = 1
        wom::reveal(10, 0),
        wom::halt(),
    ];
    run_vm_test("Division of equal numbers", instructions, 1, None).unwrap()
}

#[test]
fn test_div_with_remainder() {
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::add_imm(8, 0, 17_i16),
        wom::add_imm(9, 0, 5_i16),
        wom::div(10, 8, 9), // 17 / 5 = 3 (integer division)
        wom::reveal(10, 0),
        wom::halt(),
    ];
    run_vm_test("Division with remainder", instructions, 3, None).unwrap()
}

#[test]
fn test_div_zero_dividend() {
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::add_imm(8, 0, 0_i16),
        wom::add_imm(9, 0, 100_i16),
        wom::div(10, 8, 9), // 0 / 100 = 0
        wom::reveal(10, 0),
        wom::halt(),
    ];
    run_vm_test("Division of zero", instructions, 0, None).unwrap()
}

#[test]
fn test_div_large_numbers() {
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::const_32_imm(8, 0, 1000), // 65536000
        wom::const_32_imm(9, 256, 0),  // 256
        wom::div(10, 8, 9),            // 65536000 / 256 = 256000
        wom::reveal(10, 0),
        wom::halt(),
    ];
    run_vm_test("Division of large numbers", instructions, 256000, None).unwrap()
}

#[test]
fn test_div_powers_of_two() {
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::add_imm(8, 0, 128_i16),
        wom::add_imm(9, 0, 8_i16), // 2^3
        wom::div(10, 8, 9),        // 128 / 8 = 16
        wom::reveal(10, 0),
        wom::halt(),
    ];
    run_vm_test("Division by power of 2", instructions, 16, None).unwrap()
}

#[test]
fn test_div_chain() {
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::add_imm(8, 0, 120_i16),
        wom::add_imm(9, 0, 2_i16),
        wom::add_imm(10, 0, 3_i16),
        wom::div(11, 8, 9),   // 120 / 2 = 60
        wom::div(12, 11, 10), // 60 / 3 = 20
        wom::reveal(12, 0),
        wom::halt(),
    ];
    run_vm_test("Chained division", instructions, 20, None).unwrap()
}

#[test]
fn test_div_negative_signed() {
    // Testing signed division with negative numbers
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::const_32_imm(8, 0xFFF6, 0xFFFF), // -10 in two's complement
        wom::add_imm(9, 0, 2_i16),
        wom::div(10, 8, 9), // -10 / 2 = -5
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
    .unwrap()
}

#[test]
fn test_div_both_negative() {
    // Testing signed division with both numbers negative
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::const_32_imm(8, 0xFFEC, 0xFFFF), // -20 in two's complement
        wom::const_32_imm(9, 0xFFFB, 0xFFFF), // -5 in two's complement
        wom::div(10, 8, 9),                   // -20 / -5 = 4
        wom::reveal(10, 0),
        wom::halt(),
    ];
    run_vm_test("Signed division with both negative", instructions, 4, None).unwrap()
}

#[test]
fn test_div_and_mul_inverse() {
    // Test that (a / b) * b ≈ a (with integer truncation)
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::add_imm(8, 0, 100_i16),
        wom::add_imm(9, 0, 7_i16),
        wom::div(10, 8, 9),  // 100 / 7 = 14
        wom::mul(11, 10, 9), // 14 * 7 = 98 (not 100 due to truncation)
        wom::reveal(11, 0),
        wom::halt(),
    ];
    run_vm_test(
        "Division and multiplication relationship",
        instructions,
        98,
        None,
    )
    .unwrap()
}

#[test]
fn test_ret_instruction() {
    // Test RET: return to saved PC and FP
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),  // PC=0
        wom::add_imm(10, 0, 24_i16), // PC=4: x10 = 24 (return PC)
        wom::add_imm(11, 0, 0_i16),  // PC=8: x11 = 0 (saved FP)
        wom::add_imm(8, 0, 88_i16),  // PC=12: x8 = 88
        wom::ret(10, 11),            // PC=16: Return to PC=x10, FP=x11
        wom::halt(),                 // PC=20: This should be skipped
        // PC = 24 (where x10 points)
        wom::reveal(8, 0), // PC=24: reveal x8 (should be 88)
        wom::halt(),       // PC=28
    ];

    run_vm_test("RET instruction", instructions, 88, None).unwrap()
}

#[test]
fn test_call_instruction() {
    // Test CALL: save PC and FP, sets a new frame then jump
    // CALL saves return PC (pc+4=8) into x10 in the new frame, then jumps to PC=16
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),  // PC=0
        wom::call(10, 11, 16, 100),  // PC=4: CALL to PC=16, FP offset=100
        wom::add_imm(8, 0, 123_i16), // PC=8: should NOT execute
        wom::halt(),                 // PC=12: padding
        // New frame starts here (PC=16), FP = old_fp + 100
        wom::reveal(10, 0), // PC=16: reveal x10 (should be 8, the return address)
        wom::halt(),        // PC=20: End
    ];

    run_vm_test("CALL instruction", instructions, 8, None).unwrap()
}

#[test]
fn test_call_indirect_instruction() {
    // Test CALL_INDIRECT: save PC and FP, jump to register value
    // x12 holds the target PC, CALL_INDIRECT saves old FP into x11 in the new frame
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),          // PC=0
        wom::add_imm(12, 0, 16_i16),         // PC=4: x12 = 16 (target PC)
        wom::call_indirect(10, 11, 12, 100), // PC=8: CALL_INDIRECT to PC=x12, FP offset=100
        wom::halt(),                         // PC=12: padding
        // New frame starts here (PC=16), FP = old_fp + 100
        wom::reveal(11, 0), // PC=16: reveal x11 (should be 0, the saved old FP)
        wom::halt(),        // PC=20: End
    ];

    run_vm_test("CALL_INDIRECT instruction", instructions, 0, None).unwrap()
}

#[test]
fn test_call_and_return() {
    // Test a complete call and return sequence
    // Note: When FP changes, register addressing changes too
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::add_imm(8, 0, 50_i16), // x8 = 50 (at FP=0)
        wom::call(10, 11, 20, 100), // Call function at PC=20, FP offset=100
        wom::reveal(8, 0),          // wom::reveal x8 after return (should be 50)
        wom::halt(),
        // Function at PC = 20
        wom::const_32_imm(8, 1, 0), // x8 = 1 in new frame
        wom::ret(10, 11),           // Return using saved PC and FP
        wom::halt(),
    ];

    run_vm_test("CALL and RETURN sequence", instructions, 50, None).unwrap()
}

#[test]
fn test_jump_instruction() {
    // Test unconditional JUMP
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),  // PC=0:
        wom::jump(20),               // PC=4: Jump to PC=20
        wom::add_imm(9, 0, 999_i16), // PC=8: This should be skipped
        wom::reveal(9, 0),           // PC=12: This should be skipped
        wom::halt(),                 // PC=16: Padding
        // PC = 20 (jump target)
        wom::add_imm(9, 0, 58_i16), // PC=20: x8 = 42 + 58 = 100
        wom::reveal(9, 0),          // PC=24: wom::reveal x8 (should be 100)
        wom::halt(),                // PC=28: End
    ];

    run_vm_test("JUMP instruction", instructions, 58, None).unwrap()
}

#[test]
fn test_jump_if_instruction() {
    // Test conditional JUMP_IF (condition != 0)
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),  // PC=0
        wom::add_imm(9, 0, 5_i16),   // PC=4: x9 = 5 (condition != 0)
        wom::jump_if(9, 24),         // PC=8: Jump to PC=24 if x9 != 0 (should jump)
        wom::add_imm(8, 0, 999_i16), // PC=12: This should be skipped
        wom::reveal(8, 0),           // PC=16: This should be skipped
        wom::halt(),                 // PC=20: Padding
        // PC = 24 (jump target)
        wom::add_imm(8, 0, 15_i16), // PC=24: x8 = 15
        wom::reveal(8, 0),          // PC=28: wom::reveal x8 (should be 25)
        wom::halt(),                // PC=32: End
    ];

    run_vm_test(
        "JUMP_IF instruction (true condition)",
        instructions,
        15,
        None,
    )
    .unwrap()
}

#[test]
fn test_jump_if_false_condition() {
    // Test conditional JUMP_IF with false condition (should not jump)
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::add_imm(9, 0, 0_i16), // PC=4: x9 = 0 (condition == 0, should not jump)
        wom::jump_if(9, 28),       // PC=8: Jump to PC=28 if x9 != 0 (should NOT jump)
        wom::add_imm(8, 0, 20_i16), // PC=12: x8 = 30 + 20 = 50 (this should execute)
        wom::reveal(8, 0),         // PC=16: wom::reveal x8 (should be 50)
        wom::halt(),               // PC=20: End
        // PC = 24 (jump target that should not be reached)
        wom::add_imm(8, 0, 999_i16), // PC=24: This should not execute
        wom::reveal(8, 0),           // PC=28: This should not execute
        wom::halt(),                 // PC=32: This should not execute
    ];

    run_vm_test(
        "JUMP_IF instruction (false condition)",
        instructions,
        20,
        None,
    )
    .unwrap()
}

#[test]
fn test_jump_if_zero_instruction() {
    // Test conditional JUMP_IF_ZERO (condition == 0)
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::add_imm(9, 0, 0_i16),   // PC=4: x9 = 0 (condition == 0)
        wom::jump_if_zero(9, 24),    // PC=8: Jump to PC=24 if x9 == 0 (should jump)
        wom::add_imm(8, 0, 999_i16), // PC=12: This should be skipped
        wom::reveal(8, 0),           // PC=16: This should be skipped
        wom::halt(),                 // PC=20: Padding
        // PC = 24 (jump target)
        wom::add_imm(8, 0, 23_i16), // PC=24: x8 = 23
        wom::reveal(8, 0),          // PC=28: wom::reveal x8 (should be 100)
        wom::halt(),                // PC=32: End
    ];

    run_vm_test(
        "JUMP_IF_ZERO instruction (true condition)",
        instructions,
        23,
        None,
    )
    .unwrap()
}

#[test]
fn test_jump_if_zero_false_condition() {
    // Test conditional JUMP_IF_ZERO with false condition (should not jump)
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::add_imm(9, 0, 7_i16), // PC=4: x9 = 7 (condition != 0, should not jump)
        wom::jump_if_zero(9, 28),  // PC=8: Jump to PC=28 if x9 == 0 (should NOT jump)
        wom::add_imm(8, 0, 40_i16), // PC=12: x8 = 40 (this should execute)
        wom::reveal(8, 0),         // PC=16: wom::reveal x8 (should be 100)
        wom::halt(),               // PC=20: End
        // PC = 24 (jump target that should not be reached)
        wom::add_imm(8, 0, 999_i16), // PC=24: This should not execute
        wom::reveal(8, 0),           // PC=28: This should not execute
        wom::halt(),                 // PC=32: This should not execute
    ];

    run_vm_test(
        "JUMP_IF_ZERO instruction (false condition)",
        instructions,
        40,
        None,
    )
    .unwrap()
}

#[test]
fn test_const32_simple() {
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::const_32_imm(8, 0x1234, 0x5678), // Load 0x56781234 into x8
        wom::reveal(8, 0),
        wom::halt(),
    ];

    run_vm_test("CONST32 simple test", instructions, 0x56781234, None).unwrap()
}

#[test]
fn test_const32_zero() {
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::const_32_imm(10, 0, 0), // Load 0 into x10
        wom::reveal(10, 0),
        wom::halt(),
    ];

    run_vm_test("CONST32 zero test", instructions, 0, None).unwrap()
}

#[test]
fn test_const32_max_value() {
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::const_32_imm(12, 0xFFFF, 0xFFFF), // Load 0xFFFFFFFF into x12
        wom::reveal(12, 0),
        wom::halt(),
    ];

    run_vm_test("CONST32 max value test", instructions, 0xFFFFFFFF, None).unwrap()
}

#[test]
fn test_const32_multiple_registers() {
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::const_32_imm(8, 100, 0), // Load 100 into x8
        wom::const_32_imm(9, 200, 0), // Load 200 into x9
        wom::add(11, 8, 9),           // x11 = x8 + x9 = 300
        wom::reveal(11, 0),
        wom::halt(),
    ];

    run_vm_test("CONST32 multiple registers test", instructions, 300, None).unwrap()
}

#[test]
fn test_const32_with_arithmetic() {
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::const_32_imm(8, 1000, 0), // Load 1000 into x8
        wom::const_32_imm(9, 234, 0),  // Load 234 into x9
        wom::add(10, 8, 9),            // x10 = x8 + x9 = 1234
        wom::const_32_imm(11, 34, 0),  // Load 34 into x11
        wom::sub(12, 10, 11),          // x12 = x10 - x11 = 1200
        wom::reveal(12, 0),
        wom::halt(),
    ];

    run_vm_test("CONST32 with arithmetic test", instructions, 1200, None).unwrap()
}

#[test]
fn test_lt_u_true() {
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::const_32_imm(8, 100, 0), // Load 100 into x8
        wom::const_32_imm(9, 200, 0), // Load 200 into x9
        wom::lt_u(10, 8, 9),          // x10 = (x8 < x9) = (100 < 200) = 1
        wom::reveal(10, 0),
        wom::halt(),
    ];

    run_vm_test("SLTU true test", instructions, 1, None).unwrap()
}

#[test]
fn test_lt_u_false() {
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::const_32_imm(8, 200, 0), // Load 200 into x8
        wom::const_32_imm(9, 100, 0), // Load 100 into x9
        wom::lt_u(10, 8, 9),          // x10 = (x8 < x9) = (200 < 100) = 0
        wom::reveal(10, 0),
        wom::halt(),
    ];

    run_vm_test("SLTU false test", instructions, 0, None).unwrap()
}

#[test]
fn test_lt_u_equal() {
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::const_32_imm(8, 150, 0), // Load 150 into x8
        wom::const_32_imm(9, 150, 0), // Load 150 into x9
        wom::lt_u(10, 8, 9),          // x10 = (x8 < x9) = (150 < 150) = 0
        wom::reveal(10, 0),
        wom::halt(),
    ];

    run_vm_test("SLTU equal test", instructions, 0, None).unwrap()
}

#[test]
fn test_lt_s_positive() {
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::const_32_imm(8, 50, 0),  // Load 50 into x8
        wom::const_32_imm(9, 100, 0), // Load 100 into x9
        wom::lt_s(10, 8, 9),          // x10 = (x8 < x9) = (50 < 100) = 1
        wom::reveal(10, 0),
        wom::halt(),
    ];

    run_vm_test("SLT positive numbers test", instructions, 1, None).unwrap()
}

#[test]
fn test_lt_s_negative() {
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::const_32_imm(8, 0xFFFF, 0xFFFF), // Load -1 into x8
        wom::const_32_imm(9, 5, 0),           // Load 5 into x9
        wom::lt_s(10, 8, 9),                  // x10 = (x8 < x9) = (-1 < 5) = 1
        wom::reveal(10, 0),
        wom::halt(),
    ];

    run_vm_test("SLT negative vs positive test", instructions, 1, None).unwrap()
}

#[test]
fn test_lt_s_both_negative() {
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::const_32_imm(8, 0xFFFE, 0xFFFF), // Load -2 into x8
        wom::const_32_imm(9, 0xFFFC, 0xFFFF), // Load -4 into x9
        wom::lt_s(10, 8, 9),                  // x10 = (x8 < x9) = (-2 < -4) = 0
        wom::reveal(10, 0),
        wom::halt(),
    ];

    run_vm_test("SLT both negative test", instructions, 0, None).unwrap()
}

#[test]
fn test_lt_comparison_chain() {
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::const_32_imm(8, 10, 0),  // x8 = 10
        wom::const_32_imm(9, 20, 0),  // x9 = 20
        wom::const_32_imm(10, 30, 0), // x10 = 30
        wom::lt_u(11, 8, 9),          // x11 = (10 < 20) = 1
        wom::lt_u(12, 9, 10),         // x12 = (20 < 30) = 1
        wom::and(13, 11, 12),         // x13 = x11 & x12 = 1 & 1 = 1
        wom::reveal(13, 0),
        wom::halt(),
    ];

    run_vm_test("Less than comparison chain test", instructions, 1, None).unwrap()
}

#[test]
fn test_gt_u_true() {
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::const_32_imm(8, 200, 0), // Load 200 into x8
        wom::const_32_imm(9, 100, 0), // Load 100 into x9
        wom::gt_u(10, 8, 9),          // x10 = (x8 > x9) = (200 > 100) = 1
        wom::reveal(10, 0),
        wom::halt(),
    ];

    run_vm_test("GT_U true test", instructions, 1, None).unwrap()
}

#[test]
fn test_gt_u_false() {
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::const_32_imm(8, 100, 0), // Load 100 into x8
        wom::const_32_imm(9, 200, 0), // Load 200 into x9
        wom::gt_u(10, 8, 9),          // x10 = (x8 > x9) = (100 > 200) = 0
        wom::reveal(10, 0),
        wom::halt(),
    ];

    run_vm_test("GT_U false test", instructions, 0, None).unwrap()
}

#[test]
fn test_gt_u_equal() {
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::const_32_imm(8, 150, 0), // Load 150 into x8
        wom::const_32_imm(9, 150, 0), // Load 150 into x9
        wom::gt_u(10, 8, 9),          // x10 = (x8 > x9) = (150 > 150) = 0
        wom::reveal(10, 0),
        wom::halt(),
    ];

    run_vm_test("GT_U equal test", instructions, 0, None).unwrap()
}

#[test]
fn test_gt_s_positive() {
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::const_32_imm(8, 100, 0), // Load 100 into x8
        wom::const_32_imm(9, 50, 0),  // Load 50 into x9
        wom::gt_s(10, 8, 9),          // x10 = (x8 > x9) = (100 > 50) = 1
        wom::reveal(10, 0),
        wom::halt(),
    ];

    run_vm_test("GT_S positive numbers test", instructions, 1, None).unwrap()
}

#[test]
fn test_gt_s_negative() {
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::const_32_imm(8, 5, 0),           // Load 5 into x8
        wom::const_32_imm(9, 0xFFFF, 0xFFFF), // Load -1 into x9
        wom::gt_s(10, 8, 9),                  // x10 = (x8 > x9) = (5 > -1) = 1
        wom::reveal(10, 0),
        wom::halt(),
    ];

    run_vm_test("GT_S positive vs negative test", instructions, 1, None).unwrap()
}

#[test]
fn test_gt_s_both_negative() {
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::const_32_imm(8, 0xFFFE, 0xFFFF), // Load -2 into x8
        wom::const_32_imm(9, 0xFFFC, 0xFFFF), // Load -4 into x9
        wom::gt_s(10, 8, 9),                  // x10 = (x8 > x9) = (-2 > -4) = 1
        wom::reveal(10, 0),
        wom::halt(),
    ];

    run_vm_test("GT_S both negative test", instructions, 1, None).unwrap()
}

#[test]
fn test_gt_edge_cases() {
    let instructions = vec![
        // Test max unsigned value
        wom::const_32_imm(0, 0, 0),
        wom::const_32_imm(8, 0xFFFF, 0xFFFF), // Load 0xFFFFFFFF (max u32) into x8
        wom::const_32_imm(9, 0, 0),           // Load 0 into x9
        wom::gt_u(10, 8, 9),                  // x10 = (max > 0) = 1
        // Test with max signed positive
        wom::const_32_imm(11, 0xFFFF, 0x7FFF), // Load 0x7FFFFFFF (max positive) into x11
        wom::const_32_imm(12, 0, 0),           // Load 0 into x12
        wom::gt_s(13, 11, 12),                 // x13 = (max_pos > 0) = 1
        // Combine results
        wom::and(14, 10, 13), // x14 = 1 & 1 = 1
        wom::reveal(14, 0),
        wom::halt(),
    ];

    run_vm_test("GT edge cases test", instructions, 1, None).unwrap()
}

#[test]
fn test_comparison_equivalence() {
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::const_32_imm(8, 25, 0), // x8 = 25
        wom::const_32_imm(9, 10, 0), // x9 = 10
        // Test that (a > b) == !(a <= b) == !((a < b) || (a == b))
        wom::gt_u(10, 8, 9), // x10 = (25 > 10) = 1
        wom::lt_u(11, 9, 8), // x11 = (10 < 25) = 1 (equivalent)
        // Test that gt_u and lt_u with swapped operands are equivalent
        wom::xor(12, 10, 11), // x12 = x10 XOR x11 = 1 XOR 1 = 0 (should be 0 if equivalent)
        wom::reveal(12, 0),
        wom::halt(),
    ];

    run_vm_test("Comparison equivalence test", instructions, 0, None).unwrap()
}

#[test]
fn test_mixed_comparisons() {
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::const_32_imm(8, 0xFFFE, 0xFFFF), // x8 = -2 (signed)
        wom::const_32_imm(9, 2, 0),           // x9 = 2
        // Unsigned comparison: 0xFFFFFFFE > 2
        wom::gt_u(10, 8, 9), // x10 = 1 (large unsigned > small)
        // Signed comparison: -2 > 2
        wom::gt_s(11, 8, 9), // x11 = 0 (negative < positive)
        // Show the difference
        wom::sub(12, 10, 11), // x12 = 1 - 0 = 1
        wom::reveal(12, 0),
        wom::halt(),
    ];

    run_vm_test(
        "Mixed signed/unsigned comparison test",
        instructions,
        1,
        None,
    )
    .unwrap()
}

#[test]
fn test_input_hint() {
    // hint_storew writes to memory AS, so we use a scratch register pointing to MEM[0].
    // Each prepare_read + hint_storew(skip len) + hint_storew(data) + loadw pattern
    // reads one u32 from stdin into a register.
    let scratch = 5; // scratch register pointing to MEM[0]
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::const_32_imm(scratch, 0, 0), // scratch_reg = 0 (points to MEM[0])
        wom::prepare_read(),
        wom::hint_storew(scratch),  // skip length word
        wom::hint_storew(scratch),  // write data to MEM[0]
        wom::loadw(10, scratch, 0), // load MEM[0] → r10
        wom::reveal(10, 0),
        wom::halt(),
    ];
    let mut stdin = StdIn::default();
    stdin.write(&42u32);

    run_vm_test("Input hint", instructions, 42, Some(stdin)).unwrap()
}

#[test]
fn test_input_hint_with_frame_jump_and_xor() {
    let scratch = 5; // scratch register pointing to MEM[0]
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),       // PC=0
        wom::const_32_imm(scratch, 0, 0), // PC=4: scratch = 0
        // Read first value into r8
        wom::prepare_read(),       // PC=8
        wom::hint_storew(scratch), // PC=12: skip length
        wom::hint_storew(scratch), // PC=16: write data to MEM[0]
        wom::loadw(8, scratch, 0), // PC=20: load MEM[0] → r8
        // Store r8 to memory address 100 so the new frame can access it
        wom::add_imm(6, 0, 100_i16), // PC=24: r6 = 100
        wom::storew(8, 6, 0),        // PC=28: MEM[100] = r8
        // Call into a new frame
        wom::call(10, 11, 40, 200), // PC=32: jump to PC=40, FP += 200
        wom::halt(),                // PC=36: padding (skipped)
        // === New frame (PC=40), FP = old_FP + 200 ===
        wom::const_32_imm(0, 0, 0),       // PC=40
        wom::const_32_imm(scratch, 0, 0), // PC=44: scratch = 0
        // Load the first value from memory into r2
        wom::add_imm(6, 0, 100_i16), // PC=48: r6 = 100
        wom::loadw(2, 6, 0),         // PC=52: r2 = MEM[100]
        // Read second value into r3
        wom::prepare_read(),       // PC=56
        wom::hint_storew(scratch), // PC=60: skip length
        wom::hint_storew(scratch), // PC=64: write data to MEM[0]
        wom::loadw(3, scratch, 0), // PC=68: load MEM[0] → r3
        // XOR the two values
        wom::xor(4, 2, 3), // PC=72
        wom::reveal(4, 0), // PC=76
        wom::halt(),       // PC=80
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
    .unwrap()
}

#[test]
fn test_loadw_basic() {
    // Test basic LOADW instruction
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::add_imm(8, 0, 100_i16), // x8 = 100 (base address)
        wom::add_imm(9, 0, 42_i16),  // x9 = 42 (value to store)
        wom::storew(9, 8, 0),        // MEM[x8 + 0] = x9 (store 42 at address 100)
        wom::loadw(10, 8, 0),        // x10 = MEM[x8 + 0] (load from address 100)
        wom::reveal(10, 0),          // wom::reveal x10 (should be 42)
        wom::halt(),
    ];

    run_vm_test("LOADW basic test", instructions, 42, None).unwrap()
}

#[test]
fn test_storew_with_offset() {
    // Test STOREW with positive offset
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::add_imm(8, 0, 200_i16),  // x8 = 200 (base address)
        wom::add_imm(9, 0, 111_i16),  // x9 = 111 (first value)
        wom::add_imm(10, 0, 222_i16), // x10 = 222 (second value)
        wom::storew(9, 8, 0),         // MEM[x8 + 0] = 111
        wom::storew(10, 8, 4),        // MEM[x8 + 4] = 222
        wom::loadw(11, 8, 0),         // x11 = MEM[x8 + 0] (should be 111)
        wom::loadw(12, 8, 4),         // x12 = MEM[x8 + 4] (should be 222)
        // Test that we loaded the correct values
        wom::add(13, 11, 12), // x13 = x11 + x12 = 111 + 222 = 333
        wom::reveal(13, 0),   // wom::reveal x13 (should be 333)
        wom::halt(),
    ];

    run_vm_test("STOREW with offset test", instructions, 333, None).unwrap()
}

#[test]
fn test_loadbu_basic() {
    // Test LOADBU instruction (load byte unsigned)
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::add_imm(8, 0, 300_i16),  // x8 = 300 (base address)
        wom::add_imm(9, 0, 0xFF_i16), // x9 = 255 (max byte value)
        wom::storeb(9, 8, 0),         // MEM[x8 + 0] = 255 (store as byte)
        wom::loadbu(10, 8, 0),        // x10 = MEM[x8 + 0] (load byte unsigned)
        wom::reveal(10, 0),           // Reveal x10 (should be 255)
        wom::halt(),
    ];
    run_vm_test("LOADBU basic test", instructions, 255, None).unwrap()
}

#[test]
fn test_loadhu_basic() {
    // Test LOADHU instruction (load halfword unsigned)
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::add_imm(8, 0, 400_i16),     // x8 = 400 (base address)
        wom::const_32_imm(9, 0xABCD, 0), // x9 = 0xABCD (43981)
        wom::storeh(9, 8, 0),            // MEM[x8 + 0] = 0xABCD (store as halfword)
        wom::loadhu(10, 8, 0),           // x10 = MEM[x8 + 0] (load halfword unsigned)
        wom::reveal(10, 0),              // Reveal x10 (should be 0xABCD = 43981)
        wom::halt(),
    ];
    run_vm_test("LOADHU basic test", instructions, 0xABCD, None).unwrap()
}

#[test]
fn test_storeb_with_offset() {
    // Test STOREB with offset and masking
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::add_imm(8, 0, 500_i16),     // x8 = 500 (base address)
        wom::const_32_imm(9, 0x1234, 0), // x9 = 0x1234 (only lowest byte 0x34 will be stored)
        wom::storeb(9, 8, 0),            // MEM[x8 + 0] = 0x34 (store lowest byte)
        wom::storeb(9, 8, 1),            // MEM[x8 + 1] = 0x34 (store at offset 1)
        wom::loadbu(10, 8, 0),           // x10 = MEM[x8 + 0] (should be 0x34 = 52)
        wom::loadbu(11, 8, 1),           // x11 = MEM[x8 + 1] (should be 0x34 = 52)
        wom::add(12, 10, 11),            // x12 = x10 + x11 = 52 + 52 = 104
        wom::reveal(12, 0),              // Reveal x12 (should be 104)
        wom::halt(),
    ];
    run_vm_test("STOREB with offset test", instructions, 104, None).unwrap()
}

#[test]
fn test_storeh_with_offset() {
    // Test STOREH with offset
    let instructions = vec![
        wom::const_32_imm(0, 0, 0),
        wom::add_imm(8, 0, 600_i16),      // x8 = 600 (base address)
        wom::const_32_imm(9, 0x1111, 0),  // x9 = 0x1111
        wom::const_32_imm(10, 0x2222, 0), // x10 = 0x2222
        wom::storeh(9, 8, 0),             // MEM[x8 + 0] = 0x1111 (store halfword)
        wom::storeh(10, 8, 2),            // MEM[x8 + 2] = 0x2222 (store at offset 2)
        wom::loadhu(11, 8, 0),            // x11 = MEM[x8 + 0] (should be 0x1111 = 4369)
        wom::loadhu(12, 8, 2),            // x12 = MEM[x8 + 2] (should be 0x2222 = 8738)
        wom::add(13, 11, 12),             // x13 = 4369 + 8738 = 13107
        wom::reveal(13, 0),               // Reveal x13 (should be 13107)
        wom::halt(),
    ];
    run_vm_test("STOREH with offset test", instructions, 13107, None).unwrap()
}
