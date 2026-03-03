use std::collections::{BTreeMap, HashSet};

use openvm_circuit::arch::{
    AirInventory, ChipInventoryError, MatrixRecordArena, VmBuilder, VmCircuitExtension,
    VmCircuitConfig, VmProverExtension,
};
use openvm_circuit::system::{SystemChipInventory, SystemCpuBuilder};
use openvm_circuit_derive::{AnyEnum, Executor, MeteredExecutor, PreflightExecutor};
use openvm_circuit_primitives::Chip;
use openvm_instructions::{LocalOpcode, VmOpcode, instruction::Instruction};
use openvm_stark_backend::{config::Val, p3_field::PrimeField32, prover::cpu::CpuBackend};
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::BabyBearPoseidon2Engine, p3_baby_bear::BabyBear,
};
use openvm_womir_transpiler::{
    BaseAlu64Opcode, BaseAluOpcode, CallOpcode, ConstOpcodes, DivRem64Opcode, DivRemOpcode,
    Eq64Opcode, EqOpcode, HintStoreOpcode, JumpOpcode, LessThan64Opcode, LessThanOpcode,
    LoadStoreOpcode, Mul64Opcode, MulOpcode, Shift64Opcode, ShiftOpcode,
};
use powdr_openvm_common::program::OriginalCompiledProgram;
use powdr_openvm_common::{
    isa::{OpenVmISA, OriginalCpuChipComplex, OriginalCpuChipInventory},
    trace_generator::cpu::periphery::{SharedPeripheryChipsCpu, SharedPeripheryChipsCpuProverExt},
    vm::PowdrExtensionExecutor,
};
use strum::IntoEnumIterator;
use womir_circuit::{WomirConfig, WomirConfigExecutor, WomirCpuBuilder, WomirProverExt};
use womir_translation::LinkedProgram;

#[derive(Clone, Default)]
pub struct WomirISA;

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

#[derive(Clone, Default)]
pub struct WomirDummyBuilder;

impl VmBuilder<BabyBearPoseidon2Engine> for WomirDummyBuilder {
    type VmConfig = WomirConfig;
    type SystemChipInventory = SystemChipInventory<powdr_openvm_common::BabyBearSC>;
    type RecordArena = MatrixRecordArena<Val<powdr_openvm_common::BabyBearSC>>;

    fn create_chip_complex(
        &self,
        config: &WomirConfig,
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
            config,
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
    type Program<'a> = LinkedProgram<'a, BabyBear>;
    type RegisterAddress = ();
    type DummyExecutor = WomirConfigExecutor<BabyBear>;
    type DummyConfig = WomirConfig;
    type DummyBuilder = WomirDummyBuilder;
    type Executor = SpecializedExecutor;
    type OriginalConfig = WomirConfig;

    fn lower(original: Self::OriginalConfig) -> Self::DummyConfig {
        original
    }

    fn create_original_chip_complex(
        config: &Self::OriginalConfig,
        airs: AirInventory<powdr_openvm_common::BabyBearSC>,
    ) -> Result<OriginalCpuChipComplex, ChipInventoryError> {
        <WomirCpuBuilder as VmBuilder<BabyBearPoseidon2Engine>>::create_chip_complex(
            &WomirCpuBuilder,
            config,
            airs,
        )
    }

    fn create_dummy_inventory(
        config: &Self::OriginalConfig,
        context: SharedPeripheryChipsCpu<Self>,
    ) -> OriginalCpuChipInventory {
        let dummy_config = Self::lower(config.clone());
        let mut airs = dummy_config
            .system
            .create_airs()
            .expect("failed to create system AIR inventory for dummy config");
        airs.start_new_extension();
        VmCircuitExtension::extend_circuit(&context, &mut airs)
            .expect("failed to extend dummy AIRs with shared periphery");
        VmCircuitExtension::extend_circuit(&dummy_config.base, &mut airs)
            .expect("failed to extend dummy AIRs with womir extension");

        let mut chip_complex = VmBuilder::<BabyBearPoseidon2Engine>::create_chip_complex(
            &SystemCpuBuilder,
            &dummy_config.system,
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
            &dummy_config.base,
            inventory,
        )
        .expect("failed to extend dummy inventory with womir chips");

        chip_complex.inventory
    }

    fn is_branching(opcode: VmOpcode) -> bool {
        branch_opcodes().contains(&opcode)
    }

    fn instruction_allowlist() -> HashSet<VmOpcode> {
        vm_opcode_set()
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

    fn get_labels(program: &OriginalCompiledProgram<Self>) -> BTreeMap<u64, Vec<String>> {
        // TODO: is this correct?
        program.elf.labels()
    }
}
