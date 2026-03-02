use std::collections::HashSet;

use openvm_circuit::arch::{
    AirInventory, ChipInventoryError, InitFileGenerator, MatrixRecordArena, SystemConfig,
    VmBuilder, VmCircuitConfig, VmCircuitExtension, VmExecutionConfig, VmProverExtension,
};
use openvm_circuit::system::{SystemChipInventory, SystemCpuBuilder};
use openvm_circuit_derive::{AnyEnum, Executor, MeteredExecutor, PreflightExecutor};
use openvm_circuit_primitives::Chip;
use openvm_instructions::{
    LocalOpcode, VmOpcode, instruction::Instruction, program::DEFAULT_PC_STEP,
};
use openvm_sdk::config::{SdkVmConfig, TranspilerConfig};
use openvm_stark_backend::{config::Val, p3_field::PrimeField32, prover::cpu::CpuBackend};
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::BabyBearPoseidon2Engine, p3_baby_bear::BabyBear,
};
use openvm_transpiler::transpiler::Transpiler;
use openvm_womir_transpiler::{
    BaseAlu64Opcode, BaseAluOpcode, CallOpcode, ConstOpcodes, DivRem64Opcode, DivRemOpcode,
    Eq64Opcode, EqOpcode, HintStoreOpcode, JumpOpcode, LessThan64Opcode, LessThanOpcode,
    LoadStoreOpcode, Mul64Opcode, MulOpcode, Shift64Opcode, ShiftOpcode,
};
use powdr_openvm_common::{
    isa::{OpenVmISA, OriginalCpuChipComplex, OriginalCpuChipInventory},
    trace_generator::cpu::periphery::{SharedPeripheryChipsCpu, SharedPeripheryChipsCpuProverExt},
    vm::PowdrExtensionExecutor,
};
use serde::{Deserialize, Serialize};
use strum::IntoEnumIterator;
use womir_circuit::{WomirConfig, WomirConfigExecutor, WomirCpuBuilder, WomirProverExt};

#[derive(Clone, Default)]
pub struct WomirISA;

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OpenVmRegisterAddress(u8);

#[allow(clippy::large_enum_variant)]
#[derive(AnyEnum, Chip, Executor, MeteredExecutor, PreflightExecutor)]
pub enum SpecializedExecutor {
    Base(WomirConfigExecutor<BabyBear>),
    Powdr(PowdrExtensionExecutor<WomirISA>),
}

impl From<WomirConfigExecutor<BabyBear>> for SpecializedExecutor {
    fn from(value: WomirConfigExecutor<BabyBear>) -> Self {
        Self::Base(value)
    }
}

impl From<PowdrExtensionExecutor<WomirISA>> for SpecializedExecutor {
    fn from(value: PowdrExtensionExecutor<WomirISA>) -> Self {
        Self::Powdr(value)
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct WomirOpenVmConfig {
    pub sdk: SdkVmConfig,
    pub womir: WomirConfig,
}

impl TranspilerConfig<BabyBear> for WomirOpenVmConfig {
    fn transpiler(&self) -> Transpiler<BabyBear> {
        self.sdk.transpiler()
    }
}

impl InitFileGenerator for WomirOpenVmConfig {}

impl VmCircuitConfig<powdr_openvm_common::BabyBearSC> for WomirOpenVmConfig {
    fn create_airs(
        &self,
    ) -> Result<
        AirInventory<powdr_openvm_common::BabyBearSC>,
        openvm_circuit::arch::AirInventoryError,
    > {
        self.womir.create_airs()
    }
}

impl VmExecutionConfig<BabyBear> for WomirOpenVmConfig {
    type Executor = WomirConfigExecutor<BabyBear>;

    fn create_executors(
        &self,
    ) -> Result<
        openvm_circuit::arch::ExecutorInventory<Self::Executor>,
        openvm_circuit::arch::ExecutorInventoryError,
    > {
        self.womir.create_executors()
    }
}

impl AsRef<SystemConfig> for WomirOpenVmConfig {
    fn as_ref(&self) -> &SystemConfig {
        self.womir.as_ref()
    }
}

impl AsMut<SystemConfig> for WomirOpenVmConfig {
    fn as_mut(&mut self) -> &mut SystemConfig {
        self.womir.as_mut()
    }
}

#[derive(Clone, Default)]
pub struct WomirDummyBuilder;

impl VmBuilder<BabyBearPoseidon2Engine> for WomirDummyBuilder {
    type VmConfig = WomirOpenVmConfig;
    type SystemChipInventory = SystemChipInventory<powdr_openvm_common::BabyBearSC>;
    type RecordArena = MatrixRecordArena<Val<powdr_openvm_common::BabyBearSC>>;

    fn create_chip_complex(
        &self,
        config: &WomirOpenVmConfig,
        circuit: AirInventory<powdr_openvm_common::BabyBearSC>,
    ) -> Result<
        openvm_circuit::arch::VmChipComplex<
            powdr_openvm_common::BabyBearSC,
            Self::RecordArena,
            CpuBackend<powdr_openvm_common::BabyBearSC>,
            Self::SystemChipInventory,
        >,
        ChipInventoryError,
    > {
        <WomirCpuBuilder as VmBuilder<BabyBearPoseidon2Engine>>::create_chip_complex(
            &WomirCpuBuilder,
            &config.womir,
            circuit,
        )
    }
}

fn vm_opcode_set() -> HashSet<VmOpcode> {
    let mut set = HashSet::new();
    set.extend(BaseAluOpcode::iter().map(|x| x.global_opcode()));
    set.extend(BaseAlu64Opcode::iter().map(|x| x.global_opcode()));
    set.extend(MulOpcode::iter().map(|x| x.global_opcode()));
    set.extend(Mul64Opcode::iter().map(|x| x.global_opcode()));
    set.extend(LessThanOpcode::iter().map(|x| x.global_opcode()));
    set.extend(LessThan64Opcode::iter().map(|x| x.global_opcode()));
    set.extend(DivRemOpcode::iter().map(|x| x.global_opcode()));
    set.extend(DivRem64Opcode::iter().map(|x| x.global_opcode()));
    set.extend(EqOpcode::iter().map(|x| x.global_opcode()));
    set.extend(Eq64Opcode::iter().map(|x| x.global_opcode()));
    set.extend(ShiftOpcode::iter().map(|x| x.global_opcode()));
    set.extend(Shift64Opcode::iter().map(|x| x.global_opcode()));
    set.extend(
        LoadStoreOpcode::iter()
            .take(LoadStoreOpcode::STOREB as usize + 1)
            .map(|x| x.global_opcode()),
    );
    set.extend([LoadStoreOpcode::LOADB, LoadStoreOpcode::LOADH].map(|x| x.global_opcode()));
    set.extend(JumpOpcode::iter().map(|x| x.global_opcode()));
    set.extend(CallOpcode::iter().map(|x| x.global_opcode()));
    set.extend(ConstOpcodes::iter().map(|x| x.global_opcode()));
    set.extend(HintStoreOpcode::iter().map(|x| x.global_opcode()));
    set
}

fn branch_opcodes() -> HashSet<VmOpcode> {
    let mut set = HashSet::new();
    set.extend(JumpOpcode::iter().map(|x| x.global_opcode()));
    set.extend(CallOpcode::iter().map(|x| x.global_opcode()));
    set
}

impl OpenVmISA for WomirISA {
    const DEFAULT_PC_STEP: u32 = DEFAULT_PC_STEP;

    type RegisterAddress = OpenVmRegisterAddress;
    type DummyExecutor = WomirConfigExecutor<BabyBear>;
    type DummyConfig = WomirOpenVmConfig;
    type DummyBuilder = WomirDummyBuilder;
    type Executor = SpecializedExecutor;
    type OriginalConfig = WomirOpenVmConfig;

    fn lower(original: Self::OriginalConfig) -> Self::DummyConfig {
        original
    }

    fn create_original_chip_complex(
        config: &Self::OriginalConfig,
        airs: AirInventory<powdr_openvm_common::BabyBearSC>,
    ) -> Result<OriginalCpuChipComplex, ChipInventoryError> {
        <WomirCpuBuilder as VmBuilder<BabyBearPoseidon2Engine>>::create_chip_complex(
            &WomirCpuBuilder,
            &config.womir,
            airs,
        )
    }

    fn create_dummy_inventory(
        config: &Self::OriginalConfig,
        context: SharedPeripheryChipsCpu<Self>,
    ) -> OriginalCpuChipInventory {
        let dummy_config = Self::lower(config.clone());
        let mut airs = dummy_config
            .womir
            .system
            .create_airs()
            .expect("failed to create system AIR inventory for dummy config");
        airs.start_new_extension();
        VmCircuitExtension::extend_circuit(&context, &mut airs)
            .expect("failed to extend dummy AIRs with shared periphery");
        VmCircuitExtension::extend_circuit(&dummy_config.womir.base, &mut airs)
            .expect("failed to extend dummy AIRs with womir extension");

        let mut chip_complex = VmBuilder::<BabyBearPoseidon2Engine>::create_chip_complex(
            &SystemCpuBuilder,
            &dummy_config.womir.system,
            airs,
        )
        .expect("failed to create dummy chip complex");

        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<BabyBearPoseidon2Engine, _, _>::extend_prover(
            &SharedPeripheryChipsCpuProverExt,
            &context,
            inventory,
        )
        .expect("failed to preload shared periphery chips into dummy inventory");
        VmProverExtension::<BabyBearPoseidon2Engine, _, _>::extend_prover(
            &WomirProverExt,
            &dummy_config.womir.base,
            inventory,
        )
        .expect("failed to extend dummy inventory with womir chips");

        chip_complex.inventory
    }

    fn is_allowed(opcode: VmOpcode) -> bool {
        vm_opcode_set().contains(&opcode)
    }

    fn is_branching(opcode: VmOpcode) -> bool {
        branch_opcodes().contains(&opcode)
    }

    fn instruction_allowlist() -> HashSet<VmOpcode> {
        vm_opcode_set()
    }

    fn extra_targets() -> HashSet<VmOpcode> {
        branch_opcodes()
    }

    fn get_register_value(_: &Self::RegisterAddress) -> u32 {
        unimplemented!("execution constraints are currently unused")
    }

    fn value_limb(_: u32, _: usize) -> u32 {
        unimplemented!("execution constraints are currently unused")
    }

    fn format<F: PrimeField32>(instruction: &Instruction<F>) -> String {
        format!("{instruction:?}")
    }
}
