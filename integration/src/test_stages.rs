//! Test-only stages: metered execution and preflight.
//! For mock proof, use `crate::proving::mock_prove` directly.

use openvm_circuit::arch::{VirtualMachine, VmState, execution_mode::Segment};
use openvm_instructions::exe::VmExe;
use openvm_stark_sdk::{
    engine::StarkEngine, openvm_stark_backend::prover::hal::DeviceDataTransporter,
};
use womir_circuit::{WomirConfig, WomirCpuBuilder};

use crate::proving::{self, F, default_engine, vm_proving_key};

/// Stage 2: Metered execution. Returns (segments, final_state).
pub fn test_metered_execution(
    exe: &VmExe<F>,
    make_state: impl Fn() -> VmState<F>,
) -> Result<(Vec<Segment>, VmState<F>), Box<dyn std::error::Error>> {
    let engine = default_engine();
    let pk = vm_proving_key();
    let d_pk = engine.device().transport_pk_to_device(pk);
    let vm_config = WomirConfig::default();
    let vm = VirtualMachine::<_, WomirCpuBuilder>::new(engine, WomirCpuBuilder, vm_config, d_pk)?;

    let metered_ctx = vm.build_metered_ctx(exe);
    let metered_instance = vm.metered_interpreter(exe)?;
    let (segments, final_state) =
        metered_instance.execute_metered_from_state(make_state(), metered_ctx)?;

    Ok((segments, final_state))
}

/// Stage 3: Preflight (all segments). Returns the final state after the last segment.
pub fn test_preflight(
    exe: &VmExe<F>,
    make_state: impl Fn() -> VmState<F>,
) -> Result<VmState<F>, Box<dyn std::error::Error>> {
    let engine = default_engine();
    let pk = vm_proving_key();
    let d_pk = engine.device().transport_pk_to_device(pk);
    let vm_config = WomirConfig::default();
    let vm = VirtualMachine::<_, WomirCpuBuilder>::new(engine, WomirCpuBuilder, vm_config, d_pk)?;

    // Run metered execution to discover segments.
    let metered_ctx = vm.build_metered_ctx(exe);
    let metered_instance = vm.metered_interpreter(exe)?;
    let (segments, _) = metered_instance.execute_metered_from_state(make_state(), metered_ctx)?;

    // Preflight each segment.
    let mut preflight_interpreter = vm.preflight_interpreter(exe)?;
    let mut state = make_state();
    for segment in &segments {
        let output = vm.execute_preflight(
            &mut preflight_interpreter,
            state,
            Some(segment.num_insns),
            &segment.trace_heights,
        )?;
        state = output.to_state;
    }

    Ok(state)
}

/// Stage 4: Mock proof with constraint verification (all segments).
/// Delegates to `crate::proving::mock_prove`.
pub fn test_prove(
    exe: &VmExe<F>,
    make_state: impl Fn() -> VmState<F>,
) -> Result<(), Box<dyn std::error::Error>> {
    proving::mock_prove(exe, make_state)
}
