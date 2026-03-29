use openvm_circuit::{
    arch::{
        AirInventory, ChipInventoryError, InitFileGenerator, MatrixRecordArena, SystemConfig,
        VmBuilder, VmChipComplex, VmProverExtension,
    },
    system::{SystemChipInventory, SystemCpuBuilder, SystemExecutor},
};
use openvm_circuit_derive::VmConfig;
use openvm_sdk::config::TranspilerConfig;
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    engine::StarkEngine,
    p3_field::PrimeField32,
    prover::cpu::{CpuBackend, CpuDevice},
};
use openvm_transpiler::transpiler::Transpiler;
use powdr_openvm::{SpecializedExecutor, isa::OpenVmISA};
use serde::{Deserialize, Serialize};

use crate::{
    extension::{Crush, CrushCpuProverExt, CrushExecutor},
    keccak256::{Keccak256, Keccak256CpuProverExt, Keccak256Executor},
    system_config,
};

/// Config for a VM with crush base extension + keccak256 precompile.
#[derive(Clone, Debug, derive_new::new, VmConfig, Serialize, Deserialize)]
pub struct CrushKeccakConfig {
    #[config(executor = "SystemExecutor<F>")]
    pub system: SystemConfig,
    #[extension]
    pub base: Crush,
    #[extension]
    pub keccak: Keccak256,
}

impl<F: PrimeField32, ISA: OpenVmISA<Executor<F> = CrushKeccakConfigExecutor<F>>>
    From<CrushKeccakConfigExecutor<F>> for SpecializedExecutor<F, ISA>
{
    fn from(value: CrushKeccakConfigExecutor<F>) -> Self {
        Self::OriginalExecutor(value)
    }
}

impl InitFileGenerator for CrushKeccakConfig {}

impl Default for CrushKeccakConfig {
    fn default() -> Self {
        let system = system_config();
        Self {
            system,
            base: Default::default(),
            keccak: Default::default(),
        }
    }
}

impl CrushKeccakConfig {
    pub fn with_public_values(public_values: usize) -> Self {
        let system = system_config().with_public_values(public_values);
        Self {
            system,
            base: Default::default(),
            keccak: Default::default(),
        }
    }
}

impl<F: PrimeField32> TranspilerConfig<F> for CrushKeccakConfig {
    fn transpiler(&self) -> Transpiler<F> {
        Transpiler::default()
    }
}

#[derive(Clone, Default)]
pub struct CrushKeccakCpuBuilder;

impl<E, SC> VmBuilder<E> for CrushKeccakCpuBuilder
where
    SC: StarkGenericConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    Val<SC>: PrimeField32,
{
    type VmConfig = CrushKeccakConfig;
    type SystemChipInventory = SystemChipInventory<SC>;
    type RecordArena = MatrixRecordArena<Val<SC>>;

    fn create_chip_complex(
        &self,
        config: &CrushKeccakConfig,
        circuit: AirInventory<SC>,
    ) -> Result<
        VmChipComplex<SC, Self::RecordArena, E::PB, Self::SystemChipInventory>,
        ChipInventoryError,
    > {
        let mut chip_complex =
            VmBuilder::<E>::create_chip_complex(&SystemCpuBuilder, &config.system, circuit)?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<E, _, _>::extend_prover(&CrushCpuProverExt, &config.base, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(
            &Keccak256CpuProverExt,
            &config.keccak,
            inventory,
        )?;
        Ok(chip_complex)
    }
}
