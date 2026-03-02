//! CUDA GPU proving tests for WOMIR.

use std::sync::Arc;

use openvm_circuit::{
    arch::{ContinuationVmProver, SystemConfig, VirtualMachine, VmCircuitConfig, VmInstance},
    system::cuda::extensions::SystemGpuBuilder,
};
use openvm_instructions::{exe::VmExe, program::Program};
use openvm_stark_backend::prover::hal::DeviceDataTransporter;
use openvm_stark_sdk::engine::StarkEngine;
use womir_circuit::memory_config::memory_config_with_fp;

use crate::instruction_builder::halt;
use crate::proving::gpu_engine;

type F = openvm_stark_sdk::p3_baby_bear::BabyBear;

/// Test that a simple halt() program can be proven on GPU.
/// Uses system-only config (no WOMIR extension) since WOMIR GPU tracegen is not yet implemented.
#[test]
#[ignore] // Requires CUDA hardware; run with: cargo test --features cuda test_gpu_halt -- --ignored
fn test_gpu_halt() {
    // Build a minimal program: just halt
    let instructions = vec![halt()];
    let program = Program::from_instructions(&instructions);
    let exe = Arc::new(VmExe::<F>::new(program));

    // Use system-only config with WOMIR memory layout
    let system_config = SystemConfig::default_from_memory(memory_config_with_fp());
    let engine = gpu_engine();

    // Generate proving key for system-only config
    let circuit = system_config
        .create_airs()
        .expect("failed to create AIR inventory for keygen");
    let pk = circuit.keygen(&engine);
    let d_pk = engine.device().transport_pk_to_device(&pk);

    // Create VM with system GPU builder (no extensions)
    let vm =
        VirtualMachine::<_, SystemGpuBuilder>::new(engine, SystemGpuBuilder, system_config, d_pk)
            .expect("failed to create VM");

    // Commit program and create VM instance
    let cached_program_trace = vm.commit_program_on_device(&exe.program);
    let mut vm_instance =
        VmInstance::new(vm, exe, cached_program_trace).expect("failed to create VM instance");

    // Prove the program on GPU (empty input stream)
    let input: Vec<Vec<F>> = vec![];
    let proof = vm_instance.prove(input).expect("GPU proof generation");

    // Verify we got a proof
    assert!(
        !proof.per_segment.is_empty(),
        "proof should have at least one segment"
    );
}
