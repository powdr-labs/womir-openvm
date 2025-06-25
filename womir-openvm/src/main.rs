use std::sync::Arc;

use eyre::Result;
use openvm_instructions::{exe::VmExe, program::Program};
use openvm_sdk::{
    config::{AppConfig, SdkVmConfig},
    Sdk, StdIn,
};
use openvm_stark_sdk::config::FriParameters;
type F = openvm_stark_sdk::p3_baby_bear::BabyBear;

mod instruction_builder;
use instruction_builder::add;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vm_config = SdkVmConfig::builder()
        .system(Default::default())
        .rv32i(Default::default())
        .rv32m(Default::default())
        .io(Default::default())
        .build();
    let sdk = Sdk::new();

    let instructions = vec![add::<F>(2, 0, 0)];
    let program = Program::from_instructions(&instructions);
    let exe = VmExe::new(program);

    let stdin = StdIn::default();

    let output = sdk.execute(exe.clone(), vm_config.clone(), stdin.clone())?;
    println!("public values output: {output:?}");

    // let app_log_blowup = 2;
    // let app_fri_params = FriParameters::standard_with_100_bits_conjectured_security(app_log_blowup);
    // let app_config = AppConfig::new(app_fri_params, vm_config);
    //
    // let app_committed_exe = sdk.commit_app_exe(app_fri_params, exe)?;
    //
    // let app_pk = Arc::new(sdk.app_keygen(app_config)?);
    //
    // let proof = sdk.generate_app_proof(app_pk.clone(), app_committed_exe.clone(), stdin.clone())?;
    //
    // let app_vk = app_pk.get_app_vk();
    // sdk.verify_app_proof(&app_vk, &proof)?;

    Ok(())
}
