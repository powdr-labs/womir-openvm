//! Proving infrastructure: engine setup, cached proving key, mock proof, and real proof.

use std::sync::OnceLock;

use openvm_circuit::arch::{VirtualMachine, VmCircuitConfig, VmState, debug_proving_ctx};
use openvm_instructions::exe::VmExe;
use openvm_sdk::StdIn;
use openvm_sdk::config::{AppConfig, DEFAULT_APP_LOG_BLOWUP};
use openvm_sdk::keygen::AppProvingKey;
use openvm_sdk::prover::{AppProver, verify_app_proof};
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

pub fn default_engine() -> BabyBearPoseidon2Engine {
    let fri_params =
        FriParameters::standard_with_100_bits_conjectured_security(DEFAULT_APP_LOG_BLOWUP);
    BabyBearPoseidon2Engine::new(fri_params)
}

pub fn vm_proving_key() -> &'static MultiStarkProvingKey<SC> {
    VM_PROVING_KEY.get_or_init(|| {
        let config = WomirConfig::default();
        let engine = default_engine();
        let circuit = config
            .create_airs()
            .expect("failed to create AIR inventory for keygen");
        circuit.keygen(&engine)
    })
}

/// Generate and verify a real cryptographic proof.
pub fn prove(exe: &VmExe<F>, stdin: StdIn) -> Result<(), Box<dyn std::error::Error>> {
    let vm_config = WomirConfig::default();
    let app_fri_params =
        FriParameters::standard_with_100_bits_conjectured_security(DEFAULT_APP_LOG_BLOWUP);
    let app_config = AppConfig::new(app_fri_params, vm_config);
    let app_pk = AppProvingKey::keygen(app_config)?;
    let mut app_prover = AppProver::<BabyBearPoseidon2Engine, WomirCpuBuilder>::new(
        WomirCpuBuilder,
        &app_pk.app_vm_pk,
        exe.clone().into(),
        app_pk.leaf_verifier_program_commit(),
    )?;
    let app_proof = app_prover.prove(stdin)?;

    let app_vk = app_pk.get_app_vk();
    verify_app_proof(&app_vk, &app_proof)?;
    Ok(())
}

/// Mock proof with constraint verification (all segments).
pub fn mock_prove(
    exe: &VmExe<F>,
    make_state: impl Fn() -> VmState<F>,
) -> Result<(), Box<dyn std::error::Error>> {
    let engine = default_engine();
    let pk = vm_proving_key();
    let d_pk = engine.device().transport_pk_to_device(pk);
    let vm_config = WomirConfig::default();
    let mut vm =
        VirtualMachine::<_, WomirCpuBuilder>::new(engine, WomirCpuBuilder, vm_config, d_pk)?;

    // Run metered execution to discover segments.
    let metered_ctx = vm.build_metered_ctx(exe);
    let metered_instance = vm.metered_interpreter(exe)?;
    let (segments, _) = metered_instance.execute_metered_from_state(make_state(), metered_ctx)?;

    let cached_program_trace = vm.commit_program_on_device(&exe.program);
    vm.load_program(cached_program_trace);

    // Preflight + constraint verification per segment.
    let mut preflight_interpreter = vm.preflight_interpreter(exe)?;
    let mut state = make_state();
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

    Ok(())
}
