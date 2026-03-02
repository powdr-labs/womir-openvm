//! CUDA GPU proving tests for WOMIR.

use openvm_circuit::arch::VmState;
use openvm_instructions::{exe::VmExe, program::Program};
use womir_circuit::WomirConfig;

use crate::instruction_builder::halt;
use crate::proving::mock_prove_gpu;

type F = openvm_stark_sdk::p3_baby_bear::BabyBear;

/// Test that a simple halt() program can be proven on GPU.
#[test]
fn test_gpu_halt() {
    // Build a minimal program: just halt
    let instructions = vec![halt()];
    let program = Program::from_instructions(&instructions);
    let exe = VmExe::<F>::new(program);

    let vm_config = WomirConfig::default();
    let input: Vec<Vec<F>> = vec![];
    let init_state = VmState::initial(&vm_config.system, &exe.init_memory, exe.pc_start, input);

    // Run mock proof with GPU engine
    mock_prove_gpu(&exe, init_state).expect("GPU mock proof failed");
}
