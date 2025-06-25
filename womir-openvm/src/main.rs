use derive_more::From;
use eyre::Result;
use openvm_stark_backend::config::StarkGenericConfig;
use openvm_stark_backend::p3_field::PrimeField32;
use serde::{Deserialize, Serialize};
use std::{path::Path, sync::Arc};

use openvm_circuit::arch::{
    InitFileGenerator, SystemConfig, VmChipComplex, VmConfig, VmInventoryError,
};

use openvm_circuit::{
    arch::{VmExtension, VmInventory},
    circuit_derive::{Chip, ChipUsageGetter},
    system::phantom::PhantomChip,
};
use openvm_circuit_derive::{AnyEnum, InstructionExecutor, VmConfig};
use openvm_instructions::{exe::VmExe, program::Program};
use openvm_sdk::{
    config::{AggStarkConfig, AppConfig, SdkVmConfig, SdkVmConfigExecutor, SdkVmConfigPeriphery},
    keygen::AggStarkProvingKey,
    prover::AggStarkProver,
    Sdk, StdIn,
};
use openvm_stark_sdk::config::FriParameters;
type F = openvm_stark_sdk::p3_baby_bear::BabyBear;

mod instruction_builder;
use instruction_builder::{add, add_wom};

use openvm_rv32im_wom_circuit::{self, Rv32I, Rv32IExecutor, Rv32IPeriphery};

#[derive(Serialize, Deserialize, Clone)]
pub struct SpecializedConfig {
    pub sdk_config: SdkVmConfig,
    wom: Rv32I,
}

impl SpecializedConfig {
    fn new(sdk_config: SdkVmConfig) -> Self {
        Self {
            sdk_config,
            wom: Rv32I,
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
    WomExecutor(Rv32IExecutor<F>),
}

#[derive(From, ChipUsageGetter, Chip, AnyEnum)]
pub enum SpecializedPeriphery<F: PrimeField32> {
    #[any_enum]
    SdkPeriphery(SdkVmConfigPeriphery<F>),
    #[any_enum]
    WomPeriphery(Rv32IPeriphery<F>),
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
    let vm_config = SdkVmConfig::builder()
        .system(Default::default())
        .rv32i(Default::default())
        .rv32m(Default::default())
        .io(Default::default())
        .build();
    let vm_config = SpecializedConfig::new(vm_config);
    let sdk = Sdk::new();

    let instructions = vec![add::<F>(2, 0, 0), add_wom::<F>(3, 0, 0)];
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
