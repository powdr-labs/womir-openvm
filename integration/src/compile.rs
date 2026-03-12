//! Compile-to-disk pipelines for WOMIR and RISC-V.

use std::path::Path;

use autoprecompiles::WomirISA;
use openvm_sdk::StdIn;
use openvm_sdk::config::{AppConfig, DEFAULT_APP_LOG_BLOWUP};
use openvm_stark_sdk::config::FriParameters;
use powdr_autoprecompiles::{
    empirical_constraints::EmpiricalConstraints,
    pgo::{CellPgo, NonePgo},
};
use powdr_openvm::{
    customize_exe::{OpenVmApcCandidate, customize},
    default_powdr_openvm_config, execution_profile_from_guest,
    program::OriginalCompiledProgram,
};

use crate::proving::{AGG_PK_FILE, APP_PK_FILE, COMPILED_PROGRAM_FILE, RiscvSdk, WomirSdk};

/// Compile a WOMIR program: load WASM, PGO, APC generation, keygen.
/// Saves the compiled program and proving keys to `output_dir`.
pub fn compile_womir_to_disk(
    original_program: OriginalCompiledProgram<WomirISA>,
    stdin: StdIn,
    apc_count: u64,
    output_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(output_dir)?;

    let apc_start = std::time::Instant::now();
    let compiled = if apc_count > 0 {
        let execution_profile = execution_profile_from_guest(&original_program, stdin);
        customize(
            original_program,
            default_powdr_openvm_config(apc_count, 0),
            CellPgo::<_, OpenVmApcCandidate<WomirISA>>::with_pgo_data_and_max_columns(
                execution_profile,
                None,
            ),
            EmpiricalConstraints::default(),
        )
    } else {
        customize(
            original_program,
            default_powdr_openvm_config(apc_count, 0),
            NonePgo::default(),
            EmpiricalConstraints::default(),
        )
    };
    tracing::info!("APC generation took {:?}", apc_start.elapsed());

    // Serialize compiled program
    tracing::info!("Serializing compiled program...");
    let compiled_bytes = rmp_serde::to_vec(&compiled)?;
    std::fs::write(output_dir.join(COMPILED_PROGRAM_FILE), &compiled_bytes)?;
    tracing::info!(
        "Wrote compiled_program ({:.1} MB)",
        compiled_bytes.len() as f64 / 1e6
    );

    // Keygen
    let app_fri_params =
        FriParameters::standard_with_100_bits_conjectured_security(DEFAULT_APP_LOG_BLOWUP);
    let app_config = AppConfig::new(app_fri_params, compiled.vm_config.clone());
    let sdk = WomirSdk::new_without_transpiler(app_config)?;

    let keygen_start = std::time::Instant::now();

    tracing::info!("Generating app proving key...");
    let app_pk = sdk.app_pk();
    let app_pk_bytes = rmp_serde::to_vec(app_pk)?;
    std::fs::write(output_dir.join(APP_PK_FILE), &app_pk_bytes)?;
    tracing::info!("Wrote app_pk ({:.1} MB)", app_pk_bytes.len() as f64 / 1e6);

    tracing::info!("Generating aggregation proving key...");
    let agg_pk = sdk.agg_pk();
    let agg_pk_bytes = rmp_serde::to_vec(agg_pk)?;
    std::fs::write(output_dir.join(AGG_PK_FILE), &agg_pk_bytes)?;
    tracing::info!("Wrote agg_pk ({:.1} MB)", agg_pk_bytes.len() as f64 / 1e6);

    tracing::info!("Keygen took {:?}", keygen_start.elapsed());

    Ok(())
}

/// Compile a RISC-V program: build, PGO, APC generation, keygen.
/// Saves the compiled program and proving keys to `output_dir`.
pub fn compile_riscv_to_disk(
    program: &str,
    stdin: StdIn,
    apc_count: u64,
    output_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(output_dir)?;

    let program_abs = std::fs::canonicalize(program).expect("Failed to resolve program path");
    let program_str = program_abs.to_str().unwrap();

    let original = powdr_openvm_riscv::compile_openvm(
        program_str,
        powdr_openvm_riscv::GuestOptions::default(),
    )
    .map_err(|e| eyre::eyre!("{e}"))?;

    let apc_start = std::time::Instant::now();
    let config = powdr_openvm_riscv::default_powdr_openvm_config(apc_count, 0);
    let pgo_config = if apc_count > 0 {
        let execution_profile =
            powdr_openvm::execution_profile_from_guest(&original, stdin.clone());
        powdr_openvm_riscv::PgoConfig::Cell(execution_profile, None)
    } else {
        powdr_openvm_riscv::PgoConfig::None
    };
    let compiled = powdr_openvm_riscv::compile_exe(
        original,
        config,
        pgo_config,
        powdr_autoprecompiles::empirical_constraints::EmpiricalConstraints::default(),
    )
    .map_err(|e| eyre::eyre!("{e}"))?;
    tracing::info!("APC generation took {:?}", apc_start.elapsed());

    // Serialize compiled program
    tracing::info!("Serializing compiled RISC-V program...");
    let compiled_bytes = rmp_serde::to_vec(&compiled)?;
    std::fs::write(output_dir.join(COMPILED_PROGRAM_FILE), &compiled_bytes)?;
    tracing::info!(
        "Wrote compiled_program ({:.1} MB)",
        compiled_bytes.len() as f64 / 1e6
    );

    // Keygen
    let app_fri_params =
        FriParameters::standard_with_100_bits_conjectured_security(DEFAULT_APP_LOG_BLOWUP);
    let app_config = AppConfig::new(app_fri_params, compiled.vm_config.clone());
    let sdk = RiscvSdk::new_without_transpiler(app_config)?;

    let keygen_start = std::time::Instant::now();

    tracing::info!("Generating app proving key...");
    let app_pk = sdk.app_pk();
    let app_pk_bytes = rmp_serde::to_vec(app_pk)?;
    std::fs::write(output_dir.join(APP_PK_FILE), &app_pk_bytes)?;
    tracing::info!("Wrote app_pk ({:.1} MB)", app_pk_bytes.len() as f64 / 1e6);

    tracing::info!("Generating aggregation proving key...");
    let agg_pk = sdk.agg_pk();
    let agg_pk_bytes = rmp_serde::to_vec(agg_pk)?;
    std::fs::write(output_dir.join(AGG_PK_FILE), &agg_pk_bytes)?;
    tracing::info!("Wrote agg_pk ({:.1} MB)", agg_pk_bytes.len() as f64 / 1e6);

    tracing::info!("Keygen took {:?}", keygen_start.elapsed());

    Ok(())
}
