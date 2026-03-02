//! Proving infrastructure: engine setup, cached proving key, mock proof, and real proof.

use std::sync::OnceLock;

use openvm_circuit::arch::{
    Executor, MeteredExecutor, PreflightExecutor, VirtualMachine, VmBuilder, VmCircuitConfig,
    VmExecutionConfig, VmState, debug_proving_ctx,
};
use openvm_instructions::exe::VmExe;
use openvm_native_circuit::NativeCpuBuilder;
use openvm_sdk::GenericSdk;
use openvm_sdk::StdIn;
use openvm_sdk::config::{AppConfig, DEFAULT_APP_LOG_BLOWUP};
use openvm_sdk::prover::verify_app_proof;
use openvm_stark_backend::{config::Val, p3_field::PrimeField32};
use openvm_stark_sdk::{
    config::{
        FriParameters,
        baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
    },
    engine::{StarkEngine, StarkFriEngine},
    openvm_stark_backend::{
        keygen::types::MultiStarkProvingKey, prover::hal::DeviceDataTransporter,
    },
};
use womir_circuit::{WomirConfig, WomirCpuBuilder};

pub type F = openvm_stark_sdk::p3_baby_bear::BabyBear;
type SC = BabyBearPoseidon2Config;

static VM_PROVING_KEY: OnceLock<MultiStarkProvingKey<SC>> = OnceLock::new();

/// Proving backend selection.
/// Used by test infrastructure to run tests on different backends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub enum Backend {
    Cpu,
    #[cfg(feature = "cuda")]
    Gpu,
}

#[allow(dead_code)]
impl Backend {
    /// Run mock proof on this backend, returning the final state.
    pub fn mock_prove(
        self,
        exe: &VmExe<F>,
        init_state: VmState<F>,
    ) -> Result<VmState<F>, Box<dyn std::error::Error>> {
        match self {
            Backend::Cpu => mock_prove(exe, init_state),
            #[cfg(feature = "cuda")]
            Backend::Gpu => mock_prove_gpu(exe, init_state),
        }
    }

    /// Name for display/logging.
    pub fn name(self) -> &'static str {
        match self {
            Backend::Cpu => "CPU",
            #[cfg(feature = "cuda")]
            Backend::Gpu => "GPU",
        }
    }
}

/// All available backends for the current build configuration.
/// GPU uses WomirPreparingGpuConfig which supports: BaseAlu32 + system instructions.
#[allow(dead_code)]
pub const ALL_BACKENDS: &[Backend] = &[
    Backend::Cpu,
    #[cfg(feature = "cuda")]
    Backend::Gpu,
];

/// CPU-only backend (useful for tests that don't support GPU yet).
#[allow(dead_code)]
pub const CPU_ONLY: &[Backend] = &[Backend::Cpu];

/// Create a CPU engine with default FRI parameters.
pub fn cpu_engine() -> BabyBearPoseidon2Engine {
    let fri_params =
        FriParameters::standard_with_100_bits_conjectured_security(DEFAULT_APP_LOG_BLOWUP);
    BabyBearPoseidon2Engine::new(fri_params)
}

/// Alias for backwards compatibility.
#[allow(dead_code)]
pub fn default_engine() -> BabyBearPoseidon2Engine {
    cpu_engine()
}

/// Create a GPU engine with default FRI parameters.
#[cfg(feature = "cuda")]
#[allow(dead_code)]
pub fn gpu_engine() -> openvm_cuda_backend::engine::GpuBabyBearPoseidon2Engine {
    let fri_params =
        FriParameters::standard_with_100_bits_conjectured_security(DEFAULT_APP_LOG_BLOWUP);
    openvm_cuda_backend::engine::GpuBabyBearPoseidon2Engine::new(fri_params)
}

pub fn vm_proving_key() -> &'static MultiStarkProvingKey<SC> {
    VM_PROVING_KEY.get_or_init(|| {
        let config = WomirConfig::default();
        let engine = cpu_engine();
        let circuit = config
            .create_airs()
            .expect("failed to create AIR inventory for keygen");
        circuit.keygen(&engine)
    })
}

type WomirSdk = GenericSdk<BabyBearPoseidon2Engine, WomirCpuBuilder, NativeCpuBuilder>;

/// Generate and verify a real cryptographic proof, with optional recursion.
pub fn prove(
    exe: &VmExe<F>,
    stdin: StdIn,
    recursion: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let vm_config = WomirConfig::default();
    let app_fri_params =
        FriParameters::standard_with_100_bits_conjectured_security(DEFAULT_APP_LOG_BLOWUP);
    let app_config = AppConfig::new(app_fri_params, vm_config);
    let sdk = WomirSdk::new_without_transpiler(app_config)?;

    let mut app_prover = sdk.app_prover(exe.clone())?;
    let app_proof = app_prover.prove(stdin)?;

    let app_vk = sdk.app_pk().get_app_vk();
    verify_app_proof(&app_vk, &app_proof)?;

    if recursion {
        let mut agg_prover = sdk.prover(exe.clone())?.agg_prover;

        // Note that this proof is not verified. We assume that any valid app proof
        // (verified above) also leads to a valid aggregation proof.
        // If this was not the case, it would be a completeness bug in OpenVM.
        let start = std::time::Instant::now();
        let _ = agg_prover.generate_root_verifier_input(app_proof)?;
        tracing::info!("Agg proof (inner recursion) took {:?}", start.elapsed());
    }

    Ok(())
}

/// Mock proof with constraint verification (all segments) using a specific engine and builder.
/// Returns the final state after all segments have been processed.
pub fn mock_prove_with<E, VB>(
    engine: E,
    builder: VB,
    exe: &VmExe<F>,
    init_state: VmState<F>,
) -> Result<VmState<F>, Box<dyn std::error::Error>>
where
    E: StarkEngine<SC = SC>,
    VB: VmBuilder<E, VmConfig = WomirConfig> + Clone,
    Val<E::SC>: PrimeField32,
    <WomirConfig as VmExecutionConfig<Val<E::SC>>>::Executor: Executor<Val<E::SC>>
        + MeteredExecutor<Val<E::SC>>
        + PreflightExecutor<Val<E::SC>, VB::RecordArena>,
{
    let pk = vm_proving_key();
    let d_pk = engine.device().transport_pk_to_device(pk);
    let vm_config = WomirConfig::default();
    let mut vm = VirtualMachine::<_, VB>::new(engine, builder, vm_config, d_pk)?;

    // Run metered execution to discover segments.
    let metered_ctx = vm.build_metered_ctx(exe);
    let metered_instance = vm.metered_interpreter(exe)?;
    let (segments, _) =
        metered_instance.execute_metered_from_state(init_state.clone(), metered_ctx)?;

    let cached_program_trace = vm.commit_program_on_device(&exe.program);
    vm.load_program(cached_program_trace);

    // Preflight + constraint verification per segment.
    let mut preflight_interpreter = vm.preflight_interpreter(exe)?;
    let mut state = init_state;
    for segment in &segments {
        vm.transport_init_memory_to_device(&state.memory);
        let preflight_output = vm.execute_preflight(
            &mut preflight_interpreter,
            state,
            Some(segment.num_insns),
            &segment.trace_heights,
        )?;
        state = preflight_output.to_state;

        let ctx = vm.generate_proving_ctx(
            preflight_output.system_records,
            preflight_output.record_arenas,
        )?;
        debug_proving_ctx(&vm, pk, &ctx);
    }

    Ok(state)
}

/// Mock proof with constraint verification (all segments) using CPU engine.
/// Returns the final state after all segments have been processed.
pub fn mock_prove(
    exe: &VmExe<F>,
    init_state: VmState<F>,
) -> Result<VmState<F>, Box<dyn std::error::Error>> {
    mock_prove_with(cpu_engine(), WomirCpuBuilder, exe, init_state)
}

/// Mock proof with constraint verification (all segments) using GPU engine.
/// Uses WomirPreparingGpuConfig which includes only GPU-ready chips.
/// Currently supports: BaseAlu32 + system instructions.
/// Returns the final state after all segments have been processed.
///
/// TODO: Once all WOMIR GPU chips are implemented, switch to WomirConfig+WomirGpuBuilder.
#[cfg(feature = "cuda")]
#[allow(dead_code)]
pub fn mock_prove_gpu(
    exe: &VmExe<F>,
    init_state: VmState<F>,
) -> Result<VmState<F>, Box<dyn std::error::Error>> {
    use womir_circuit::{WomirPreparingGpuBuilder, WomirPreparingGpuConfig};

    let engine = gpu_engine();
    let vm_config = WomirPreparingGpuConfig::default();
    let circuit = vm_config
        .create_airs()
        .expect("failed to create AIR inventory for keygen");
    let pk = circuit.keygen(&engine);
    let d_pk = engine.device().transport_pk_to_device(&pk);

    let mut vm = VirtualMachine::<_, WomirPreparingGpuBuilder>::new(
        engine,
        WomirPreparingGpuBuilder,
        vm_config,
        d_pk,
    )?;

    // Run metered execution to discover segments.
    let metered_ctx = vm.build_metered_ctx(exe);
    let metered_instance = vm.metered_interpreter(exe)?;
    let (segments, _) =
        metered_instance.execute_metered_from_state(init_state.clone(), metered_ctx)?;

    let cached_program_trace = vm.commit_program_on_device(&exe.program);
    vm.load_program(cached_program_trace);

    // Preflight + constraint verification per segment.
    let mut preflight_interpreter = vm.preflight_interpreter(exe)?;
    let mut state = init_state;
    for segment in &segments {
        vm.transport_init_memory_to_device(&state.memory);
        let preflight_output = vm.execute_preflight(
            &mut preflight_interpreter,
            state,
            Some(segment.num_insns),
            &segment.trace_heights,
        )?;
        state = preflight_output.to_state;

        let ctx = vm.generate_proving_ctx(
            preflight_output.system_records,
            preflight_output.record_arenas,
        )?;
        debug_proving_ctx(&vm, &pk, &ctx);
    }

    Ok(state)
}
