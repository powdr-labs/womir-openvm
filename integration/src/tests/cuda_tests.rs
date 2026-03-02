//! CUDA GPU proving tests for WOMIR.

use std::sync::Arc;

use openvm_circuit::{
    arch::{ContinuationVmProver, SystemConfig, VirtualMachine, VmCircuitConfig, VmInstance},
    system::cuda::extensions::SystemGpuBuilder,
};
use openvm_instructions::{exe::VmExe, program::Program};
use openvm_stark_backend::prover::hal::DeviceDataTransporter;
use openvm_stark_sdk::engine::StarkEngine;

use crate::instruction_builder::halt;
use crate::proving::gpu_engine;

type F = openvm_stark_sdk::p3_baby_bear::BabyBear;

/// Test that a simple halt() program can be proven on GPU.
/// Uses system-only config (no WOMIR extension) since WOMIR GPU tracegen is not yet implemented.
/// Note: We use SystemConfig::default() instead of memory_config_with_fp() because the CUDA
/// merkle tree kernel only supports address spaces 0-3, but FP_AS = 5 would require index 4.
#[test]
#[ignore] // Requires CUDA hardware; run with: cargo test --features cuda test_gpu_halt -- --ignored
fn test_gpu_halt() {
    // Build a minimal program: just halt
    let instructions = vec![halt()];
    let program = Program::from_instructions(&instructions);
    let exe = Arc::new(VmExe::<F>::new(program));

    // Use default system config (without FP_AS) since CUDA merkle tree only supports AS 0-3
    let system_config = SystemConfig::default();
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
