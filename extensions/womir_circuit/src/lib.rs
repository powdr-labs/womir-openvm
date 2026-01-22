#![cfg_attr(feature = "tco", allow(incomplete_features))]
#![cfg_attr(feature = "tco", feature(explicit_tail_calls))]
#![cfg_attr(feature = "tco", allow(internal_features))]
#![cfg_attr(feature = "tco", feature(core_intrinsics))]
use openvm_circuit::{
    arch::{
        AirInventory, ChipInventoryError, EmptyAdapterCoreLayout, ExecutionError,
        InitFileGenerator, InterpreterExecutor, MatrixRecordArena,
        PreflightExecutor, RecordArena, SystemConfig, VmBuilder, VmChipComplex,
        VmProverExtension, VmStateMut,
    },
    system::{
        SystemChipInventory, SystemCpuBuilder, SystemExecutor,
        memory::online::TracingMemory,
    },
};
use openvm_circuit_derive::{Executor, MeteredExecutor, VmConfig};
use openvm_instructions::{instruction::Instruction, LocalOpcode};
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    engine::StarkEngine,
    p3_field::PrimeField32,
    prover::cpu::{CpuBackend, CpuDevice},
};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

// ============ FP Adapter Traits ============
// Traits for adapters that support frame pointer (fp) tracking

/// Adapter executor trait that supports frame pointer operations.
/// Similar to AdapterTraceExecutor but includes fp parameter.
pub trait FpAdapterTraceExecutor<F> {
    type ReadData;
    type WriteData;
    type RecordMut<'a>;

    /// Start instruction execution with frame pointer
    fn start_with_fp(pc: u32, fp: u32, memory: &TracingMemory, record: &mut Self::RecordMut<'_>);

    /// Read data for instruction
    fn read(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        record: &mut Self::RecordMut<'_>,
    ) -> Self::ReadData;

    /// Write data after instruction execution
    fn write(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        data: Self::WriteData,
        record: &mut Self::RecordMut<'_>,
    );
}

/// Core executor trait that can execute ALU operations
pub trait FpCoreExecutor<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    /// Execute the core ALU operation and return result
    fn execute_core(
        opcode: usize,
        offset: usize,
        input_b: &[u8; NUM_LIMBS],
        input_c: &[u8; NUM_LIMBS],
    ) -> [u8; NUM_LIMBS];

    /// Get the opcode name for debugging
    fn get_opcode_name(opcode: usize, offset: usize) -> String;
}

/// Wrapper around BaseAluExecutor that adds frame pointer tracking
#[derive(Clone)]
pub struct PreflightExecutorWrapperFp<Inner, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub inner: Inner,
    /// Frame pointer - tracks the current frame offset for register access
    /// This is separate from PC and persists across instructions
    pub fp: std::cell::Cell<u32>,
    _phantom: PhantomData<([u8; NUM_LIMBS], [u8; LIMB_BITS])>,
}

impl<Inner, const NUM_LIMBS: usize, const LIMB_BITS: usize>
    PreflightExecutorWrapperFp<Inner, NUM_LIMBS, LIMB_BITS>
{
    pub fn new(inner: Inner) -> Self {
        Self {
            inner,
            fp: std::cell::Cell::new(0), // Initialize fp to 0
            _phantom: PhantomData,
        }
    }

    pub fn get_fp(&self) -> u32 {
        self.fp.get()
    }

    pub fn set_fp(&self, new_fp: u32) {
        self.fp.set(new_fp);
    }
}

// Implement PreflightExecutor for the wrapper
impl<F, A, RA, const NUM_LIMBS: usize, const LIMB_BITS: usize> PreflightExecutor<F, RA>
    for PreflightExecutorWrapperFp<base_alu::BaseAluExecutor<A, NUM_LIMBS, LIMB_BITS>, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    A: 'static
        + FpAdapterTraceExecutor<
            F,
            ReadData: Into<[[u8; NUM_LIMBS]; 2]>,
            WriteData: From<[[u8; NUM_LIMBS]; 1]>
        >,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (A::RecordMut<'buf>, &'buf mut base_alu::BaseAluCoreRecord<NUM_LIMBS>),
    >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        use openvm_rv32im_transpiler::BaseAluOpcode;
        format!("{:?}", BaseAluOpcode::from_usize(opcode - self.inner.offset))
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        use openvm_instructions::program::DEFAULT_PC_STEP;
        use openvm_rv32im_transpiler::BaseAluOpcode;

        let Instruction { opcode, .. } = instruction;

        let local_opcode = BaseAluOpcode::from_usize(opcode.local_opcode_idx(self.inner.offset));
        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

        // Get current fp from the wrapper's state
        let fp = self.get_fp();

        // Call FP-aware start
        A::start_with_fp(*state.pc, fp, state.memory, &mut adapter_record);

        // Adapter read/write will use fp internally
        [core_record.b, core_record.c] = self
            .inner.adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();

        let rd = base_alu::run_alu::<NUM_LIMBS, LIMB_BITS>(local_opcode, &core_record.b, &core_record.c);

        core_record.local_opcode = local_opcode as u8;

        self.inner.adapter
            .write(state.memory, instruction, [rd].into(), &mut adapter_record);

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        // FP may be updated by some instructions (e.g., frame allocation)
        // For basic ALU, fp stays the same

        Ok(())
    }
}

// Implement InterpreterExecutor for the wrapper by delegating to inner
impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> InterpreterExecutor<F>
    for PreflightExecutorWrapperFp<base_alu::BaseAluExecutor<A, NUM_LIMBS, LIMB_BITS>, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    base_alu::BaseAluExecutor<A, NUM_LIMBS, LIMB_BITS>: InterpreterExecutor<F>,
{
    fn pre_compute_size(&self) -> usize {
        self.inner.pre_compute_size()
    }

    #[cfg(not(feature = "tco"))]
    fn pre_compute<Ctx>(
        &self,
        pc: u32,
        inst: &openvm_instructions::instruction::Instruction<F>,
        data: &mut [u8],
    ) -> Result<openvm_circuit::arch::ExecuteFunc<F, Ctx>, openvm_circuit::arch::StaticProgramError>
    where
        Ctx: openvm_circuit::arch::ExecutionCtxTrait,
    {
        self.inner.pre_compute(pc, inst, data)
    }

    #[cfg(feature = "tco")]
    fn handler<Ctx>(
        &self,
        pc: u32,
        inst: &openvm_instructions::instruction::Instruction<F>,
        data: &mut [u8],
    ) -> Result<openvm_circuit::arch::Handler<F, Ctx>, openvm_circuit::arch::StaticProgramError>
    where
        Ctx: openvm_circuit::arch::ExecutionCtxTrait,
    {
        self.inner.handler(pc, inst, data)
    }
}

pub mod adapters;
// mod auipc;
mod base_alu;
// mod branch_eq;
// mod branch_lt;
pub mod common;
// mod divrem;
// mod hintstore;
// mod jal_lui;
// mod jalr;
// mod less_than;
mod load_sign_extend;
mod loadstore;
// mod mul;
// mod mulh;
// mod shift;

// pub use auipc::*;
pub use base_alu::*;
// pub use branch_eq::*;
// pub use branch_lt::*;
// pub use divrem::*;
// pub use hintstore::*;
// pub use jal_lui::*;
// pub use jalr::*;
// pub use less_than::*;
pub use load_sign_extend::*;
pub use loadstore::*;
// pub use mul::*;
// pub use mulh::*;
// pub use shift::*;

mod extension;
pub use extension::*;

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
        let system = SystemConfig::default();
        Self {
            system,
            base: Default::default(),
        }
    }
}

impl WomirConfig {
    pub fn with_public_values(public_values: usize) -> Self {
        let system = SystemConfig::default().with_public_values(public_values);
        Self {
            system,
            base: Default::default(),
        }
    }

    pub fn with_public_values_and_segment_len(public_values: usize, segment_len: usize) -> Self {
        let system = SystemConfig::default()
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

#[cfg(feature = "cuda")]
#[derive(Clone)]
pub struct WomirGpuBuilder;

#[cfg(feature = "cuda")]
impl VmBuilder<GpuBabyBearPoseidon2Engine> for WomirGpuBuilder {
    type VmConfig = WomirConfig;
    type SystemChipInventory = SystemChipInventoryGPU;
    type RecordArena = DenseRecordArena;

    fn create_chip_complex(
        &self,
        config: &WomirConfig,
        circuit: AirInventory<BabyBearPoseidon2Config>,
    ) -> Result<
        VmChipComplex<
            BabyBearPoseidon2Config,
            Self::RecordArena,
            GpuBackend,
            Self::SystemChipInventory,
        >,
        ChipInventoryError,
    > {
        let mut chip_complex = VmBuilder::<GpuBabyBearPoseidon2Engine>::create_chip_complex(
            &SystemGpuBuilder,
            &config.system,
            circuit,
        )?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<GpuBabyBearPoseidon2Engine, _, _>::extend_prover(
            &WomirGpuProverExt,
            &config.base,
            inventory,
        )?;
        Ok(chip_complex)
    }
}
