#![cfg_attr(feature = "tco", allow(incomplete_features))]
#![cfg_attr(feature = "tco", feature(explicit_tail_calls))]
#![cfg_attr(feature = "tco", allow(internal_features))]
#![cfg_attr(feature = "tco", feature(core_intrinsics))]
use crate::memory_config::memory_config_with_fp;
use openvm_circuit::{
    arch::{
        AirInventory, ChipInventoryError, InitFileGenerator, MatrixRecordArena, SystemConfig,
        VmBuilder, VmChipComplex, VmProverExtension,
    },
    system::{SystemChipInventory, SystemCpuBuilder, SystemExecutor},
};
use openvm_circuit_derive::{Executor, MeteredExecutor, VmConfig};
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    engine::StarkEngine,
    p3_field::PrimeField32,
    prover::cpu::{CpuBackend, CpuDevice},
};
use serde::{Deserialize, Serialize};

pub mod execution;

pub mod adapters;
mod base_alu;
mod const32;
mod load_sign_extend;
mod loadstore;

pub use base_alu::*;
pub use const32::*;
pub use load_sign_extend::*;
pub use loadstore::*;

mod extension;
pub use extension::*;

mod memory_config;

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        use openvm_circuit::arch::DenseRecordArena;
        use openvm_circuit::system::cuda::{extensions::SystemGpuBuilder, SystemChipInventoryGPU};
        use openvm_cuda_backend::{engine::GpuBabyBearPoseidon2Engine, prover_backend::GpuBackend};
        use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
        pub(crate) mod cuda_abi;
        pub use self::{
            WomirGpuBuilder as WomirBuilder,
        };
    } else {
        pub use self::{
            WomirCpuBuilder as WomirBuilder,
        };
    }
}

#[cfg(any(test, feature = "test-utils"))]
mod test_utils;

// Config for a VM with base extension and IO extension
#[derive(Clone, Debug, derive_new::new, VmConfig, Serialize, Deserialize)]
pub struct WomirConfig {
    #[config(executor = "SystemExecutor<F>")]
    pub system: SystemConfig,
    #[extension]
    pub base: Womir,
}

// Default implementation uses no init file
impl InitFileGenerator for WomirConfig {}

impl Default for WomirConfig {
    fn default() -> Self {
        let system = system_config();
        Self {
            system,
            base: Default::default(),
        }
    }
}

impl WomirConfig {
    pub fn with_public_values(public_values: usize) -> Self {
        let system = system_config().with_public_values(public_values);
        Self {
            system,
            base: Default::default(),
        }
    }

    pub fn with_public_values_and_segment_len(public_values: usize, segment_len: usize) -> Self {
        let system = system_config()
            .with_public_values(public_values)
            .with_max_segment_len(segment_len);
        Self {
            system,
            base: Default::default(),
        }
    }
}

#[derive(Clone)]
pub struct WomirCpuBuilder;

impl<E, SC> VmBuilder<E> for WomirCpuBuilder
where
    SC: StarkGenericConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    Val<SC>: PrimeField32,
{
    type VmConfig = WomirConfig;
    type SystemChipInventory = SystemChipInventory<SC>;
    type RecordArena = MatrixRecordArena<Val<SC>>;

    fn create_chip_complex(
        &self,
        config: &WomirConfig,
        circuit: AirInventory<SC>,
    ) -> Result<
        VmChipComplex<SC, Self::RecordArena, E::PB, Self::SystemChipInventory>,
        ChipInventoryError,
    > {
        let mut chip_complex =
            VmBuilder::<E>::create_chip_complex(&SystemCpuBuilder, &config.system, circuit)?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<E, _, _>::extend_prover(&WomirCpuProverExt, &config.base, inventory)?;
        Ok(chip_complex)
    }
}

pub fn system_config() -> SystemConfig {
    SystemConfig::default_from_memory(memory_config_with_fp())
}
