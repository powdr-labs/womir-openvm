use std::sync::{Arc, Mutex};

use derive_more::derive::From;
use openvm_circuit::{
    arch::{
        AdapterRuntimeContext, ExecutionState, InitFileGenerator,
        InstructionExecutor as InstructionExecutorTrait, SystemConfig, SystemPort,
        VmAdapterInterface, VmExtension, VmInventory, VmInventoryBuilder, VmInventoryError,
    },
    system::{
        memory::{MemoryController, OfflineMemory},
        phantom::PhantomChip,
    },
};
use openvm_circuit_derive::{AnyEnum, InstructionExecutor, VmConfig};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    range_tuple::{RangeTupleCheckerBus, SharedRangeTupleCheckerChip},
    AlignedBorrow,
};
use openvm_circuit_primitives_derive::{Chip, ChipUsageGetter};
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode, PhantomDiscriminant, VmOpcode,
};
use openvm_rv32im_wom_transpiler::{
    BaseAluOpcode, BranchEqualOpcode, BranchLessThanOpcode, DivRemOpcode, LessThanOpcode,
    MulHOpcode, MulOpcode, Rv32AuipcOpcode, Rv32HintStoreOpcode, Rv32JalLuiOpcode, Rv32JalrOpcode,
    Rv32LoadStoreOpcode, Rv32Phantom, ShiftOpcode,
};
use openvm_stark_backend::{
    interaction::{BusIndex, InteractionBuilder, PermutationCheckBus},
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    p3_matrix::dense::RowMajorMatrix,
    rap::{BaseAirWithPublicValues, ColumnsAir},
};

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use struct_reflection::{StructReflection, StructReflectionHelper};
use strum::IntoEnumIterator;
use thiserror::Error;

use crate::{adapters::*, *};

use openvm_circuit::arch::Result as ResultVm;

/// Config for a VM with base extension and IO extension
#[derive(Clone, Debug, VmConfig, derive_new::new, Serialize, Deserialize)]
pub struct Rv32IConfig {
    #[system]
    pub system: SystemConfig,
    #[extension]
    pub base: Rv32I,
    #[extension]
    pub io: Rv32Io,
}

// Default implementation uses no init file
impl InitFileGenerator for Rv32IConfig {}

/// Config for a VM with base extension, IO extension, and multiplication extension
#[derive(Clone, Debug, Default, VmConfig, derive_new::new, Serialize, Deserialize)]
pub struct Rv32ImConfig {
    #[config]
    pub rv32i: Rv32IConfig,
    #[extension]
    pub mul: Rv32M,
}

// Default implementation uses no init file
impl InitFileGenerator for Rv32ImConfig {}

impl Default for Rv32IConfig {
    fn default() -> Self {
        let system = SystemConfig::default().with_continuations();
        Self {
            system,
            base: Default::default(),
            io: Default::default(),
        }
    }
}

impl Rv32IConfig {
    pub fn with_public_values(public_values: usize) -> Self {
        let system = SystemConfig::default()
            .with_continuations()
            .with_public_values(public_values);
        Self {
            system,
            base: Default::default(),
            io: Default::default(),
        }
    }

    pub fn with_public_values_and_segment_len(public_values: usize, segment_len: usize) -> Self {
        let system = SystemConfig::default()
            .with_continuations()
            .with_public_values(public_values)
            .with_max_segment_len(segment_len);
        Self {
            system,
            base: Default::default(),
            io: Default::default(),
        }
    }
}

impl Rv32ImConfig {
    pub fn with_public_values(public_values: usize) -> Self {
        Self {
            rv32i: Rv32IConfig::with_public_values(public_values),
            mul: Default::default(),
        }
    }

    pub fn with_public_values_and_segment_len(public_values: usize, segment_len: usize) -> Self {
        Self {
            rv32i: Rv32IConfig::with_public_values_and_segment_len(public_values, segment_len),
            mul: Default::default(),
        }
    }
}

// ============ Extension Implementations ============

/// RISC-V 32-bit Base (RV32I) Extension
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Rv32I;

/// RISC-V Extension for handling IO (not to be confused with I base extension)
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Rv32Io;

/// RISC-V 32-bit Multiplication Extension (RV32M) Extension
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Rv32M {
    #[serde(default = "default_range_tuple_checker_sizes")]
    pub range_tuple_checker_sizes: [u32; 2],
}

impl Default for Rv32M {
    fn default() -> Self {
        Self {
            range_tuple_checker_sizes: default_range_tuple_checker_sizes(),
        }
    }
}

fn default_range_tuple_checker_sizes() -> [u32; 2] {
    [1 << 8, 8 * (1 << 8)]
}

// ============ Executor and Periphery Enums for Extension ============

/// RISC-V 32-bit Base (RV32I) Instruction Executors
#[derive(ChipUsageGetter, Chip, InstructionExecutor, From, AnyEnum)]
pub enum Rv32IExecutor<F: PrimeField32> {
    // Rv32 (for standard 32-bit integers):
    BaseAlu(Rv32WomBaseAluChip<F>),
    // LessThan(Rv32LessThanChip<F>),
    // Shift(Rv32ShiftChip<F>),
    // LoadStore(Rv32LoadStoreChip<F>),
    // LoadSignExtend(Rv32LoadSignExtendChip<F>),
    // BranchEqual(Rv32BranchEqualChip<F>),
    // BranchLessThan(Rv32BranchLessThanChip<F>),
    // JalLui(Rv32JalLuiChip<F>),
    // Jalr(Rv32JalrChip<F>),
    // Auipc(Rv32AuipcChip<F>),
}

/// RISC-V 32-bit Multiplication Extension (RV32M) Instruction Executors
#[derive(ChipUsageGetter, Chip, InstructionExecutor, From, AnyEnum)]
pub enum Rv32MExecutor<F: PrimeField32> {
    Multiplication(Rv32MultiplicationChip<F>),
    MultiplicationHigh(Rv32MulHChip<F>),
    DivRem(Rv32DivRemChip<F>),
}

/// RISC-V 32-bit Io Instruction Executors
#[derive(ChipUsageGetter, Chip, InstructionExecutor, From, AnyEnum)]
pub enum Rv32IoExecutor<F: PrimeField32> {
    HintStore(Rv32HintStoreChip<F>),
}

#[derive(From, ChipUsageGetter, Chip, AnyEnum)]
pub enum Rv32IPeriphery<F: PrimeField32> {
    BitwiseOperationLookup(SharedBitwiseOperationLookupChip<8>),
    // We put this only to get the <F> generic to work
    Phantom(PhantomChip<F>),
}

#[derive(From, ChipUsageGetter, Chip, AnyEnum)]
pub enum Rv32MPeriphery<F: PrimeField32> {
    BitwiseOperationLookup(SharedBitwiseOperationLookupChip<8>),
    /// Only needed for multiplication extension
    RangeTupleChecker(SharedRangeTupleCheckerChip<2>),
    // We put this only to get the <F> generic to work
    Phantom(PhantomChip<F>),
}

#[derive(From, ChipUsageGetter, Chip, AnyEnum)]
pub enum Rv32IoPeriphery<F: PrimeField32> {
    BitwiseOperationLookup(SharedBitwiseOperationLookupChip<8>),
    // We put this only to get the <F> generic to work
    Phantom(PhantomChip<F>),
}

// ============ VmExtension Implementations ============

impl<F: PrimeField32> VmExtension<F> for Rv32I {
    type Executor = Rv32IExecutor<F>;
    type Periphery = Rv32IPeriphery<F>;

    fn build(
        &self,
        builder: &mut VmInventoryBuilder<F>,
    ) -> Result<VmInventory<Rv32IExecutor<F>, Rv32IPeriphery<F>>, VmInventoryError> {
        let mut inventory = VmInventory::new();
        let SystemPort {
            execution_bus,
            program_bus,
            memory_bridge,
        } = builder.system_port();

        let frame_bus = FrameBus::new(builder.new_bus_idx());

        let range_checker = builder.system_base().range_checker_chip.clone();
        let offline_memory = builder.system_base().offline_memory();
        let pointer_max_bits = builder.system_config().memory_config.pointer_max_bits;

        let bitwise_lu_chip = if let Some(&chip) = builder
            .find_chip::<SharedBitwiseOperationLookupChip<8>>()
            .first()
        {
            chip.clone()
        } else {
            let bitwise_lu_bus = BitwiseOperationLookupBus::new(builder.new_bus_idx());
            let chip = SharedBitwiseOperationLookupChip::new(bitwise_lu_bus);
            inventory.add_periphery_chip(chip.clone());
            chip
        };

        let base_alu_chip = Rv32WomBaseAluChip::new(
            Rv32WomBaseAluAdapterChip::new(
                execution_bus,
                program_bus,
                frame_bus,
                memory_bridge,
                bitwise_lu_chip.clone(),
            ),
            BaseAluCoreChipWom::new(bitwise_lu_chip.clone(), BaseAluOpcode::CLASS_OFFSET),
            offline_memory.clone(),
        );
        inventory.add_executor(
            base_alu_chip,
            BaseAluOpcode::iter().map(|x| x.global_opcode()),
        )?;

        // let lt_chip = Rv32LessThanChip::new(
        //     Rv32WomBaseAluAdapterChip::new(
        //         execution_bus,
        //         program_bus,
        //         memory_bridge,
        //         bitwise_lu_chip.clone(),
        //     ),
        //     LessThanCoreChip::new(bitwise_lu_chip.clone(), LessThanOpcode::CLASS_OFFSET),
        //     offline_memory.clone(),
        // );
        // inventory.add_executor(lt_chip, LessThanOpcode::iter().map(|x| x.global_opcode()))?;
        //
        // let shift_chip = Rv32ShiftChip::new(
        //     Rv32WomBaseAluAdapterChip::new(
        //         execution_bus,
        //         program_bus,
        //         memory_bridge,
        //         bitwise_lu_chip.clone(),
        //     ),
        //     ShiftCoreChip::new(
        //         bitwise_lu_chip.clone(),
        //         range_checker.clone(),
        //         ShiftOpcode::CLASS_OFFSET,
        //     ),
        //     offline_memory.clone(),
        // );
        // inventory.add_executor(shift_chip, ShiftOpcode::iter().map(|x| x.global_opcode()))?;
        //
        // let load_store_chip = Rv32LoadStoreChip::new(
        //     Rv32LoadStoreAdapterChip::new(
        //         execution_bus,
        //         program_bus,
        //         memory_bridge,
        //         pointer_max_bits,
        //         range_checker.clone(),
        //     ),
        //     LoadStoreCoreChip::new(Rv32LoadStoreOpcode::CLASS_OFFSET),
        //     offline_memory.clone(),
        // );
        // inventory.add_executor(
        //     load_store_chip,
        //     Rv32LoadStoreOpcode::iter()
        //         .take(Rv32LoadStoreOpcode::STOREB as usize + 1)
        //         .map(|x| x.global_opcode()),
        // )?;
        //
        // let load_sign_extend_chip = Rv32LoadSignExtendChip::new(
        //     Rv32LoadStoreAdapterChip::new(
        //         execution_bus,
        //         program_bus,
        //         memory_bridge,
        //         pointer_max_bits,
        //         range_checker.clone(),
        //     ),
        //     LoadSignExtendCoreChip::new(range_checker.clone()),
        //     offline_memory.clone(),
        // );
        // inventory.add_executor(
        //     load_sign_extend_chip,
        //     [Rv32LoadStoreOpcode::LOADB, Rv32LoadStoreOpcode::LOADH].map(|x| x.global_opcode()),
        // )?;
        //
        // let beq_chip = Rv32BranchEqualChip::new(
        //     Rv32BranchAdapterChip::new(execution_bus, program_bus, memory_bridge),
        //     BranchEqualCoreChip::new(BranchEqualOpcode::CLASS_OFFSET, DEFAULT_PC_STEP),
        //     offline_memory.clone(),
        // );
        // inventory.add_executor(
        //     beq_chip,
        //     BranchEqualOpcode::iter().map(|x| x.global_opcode()),
        // )?;
        //
        // let blt_chip = Rv32BranchLessThanChip::new(
        //     Rv32BranchAdapterChip::new(execution_bus, program_bus, memory_bridge),
        //     BranchLessThanCoreChip::new(
        //         bitwise_lu_chip.clone(),
        //         BranchLessThanOpcode::CLASS_OFFSET,
        //     ),
        //     offline_memory.clone(),
        // );
        // inventory.add_executor(
        //     blt_chip,
        //     BranchLessThanOpcode::iter().map(|x| x.global_opcode()),
        // )?;
        //
        // let jal_lui_chip = Rv32JalLuiChip::new(
        //     Rv32CondRdWriteAdapterChip::new(execution_bus, program_bus, memory_bridge),
        //     Rv32JalLuiCoreChip::new(bitwise_lu_chip.clone()),
        //     offline_memory.clone(),
        // );
        // inventory.add_executor(
        //     jal_lui_chip,
        //     Rv32JalLuiOpcode::iter().map(|x| x.global_opcode()),
        // )?;
        //
        // let jalr_chip = Rv32JalrChip::new(
        //     Rv32JalrAdapterChip::new(execution_bus, program_bus, memory_bridge),
        //     Rv32JalrCoreChip::new(bitwise_lu_chip.clone(), range_checker.clone()),
        //     offline_memory.clone(),
        // );
        // inventory.add_executor(jalr_chip, Rv32JalrOpcode::iter().map(|x| x.global_opcode()))?;
        //
        // let auipc_chip = Rv32AuipcChip::new(
        //     Rv32RdWriteAdapterChip::new(execution_bus, program_bus, memory_bridge),
        //     Rv32AuipcCoreChip::new(bitwise_lu_chip.clone()),
        //     offline_memory.clone(),
        // );
        // inventory.add_executor(
        //     auipc_chip,
        //     Rv32AuipcOpcode::iter().map(|x| x.global_opcode()),
        // )?;

        // There is no downside to adding phantom sub-executors, so we do it in the base extension.
        builder.add_phantom_sub_executor(
            phantom::Rv32HintInputSubEx,
            PhantomDiscriminant(Rv32Phantom::HintInput as u16),
        )?;
        builder.add_phantom_sub_executor(
            phantom::Rv32HintRandomSubEx::new(),
            PhantomDiscriminant(Rv32Phantom::HintRandom as u16),
        )?;
        builder.add_phantom_sub_executor(
            phantom::Rv32PrintStrSubEx,
            PhantomDiscriminant(Rv32Phantom::PrintStr as u16),
        )?;
        builder.add_phantom_sub_executor(
            phantom::Rv32HintLoadByKeySubEx,
            PhantomDiscriminant(Rv32Phantom::HintLoadByKey as u16),
        )?;

        Ok(inventory)
    }
}

impl<F: PrimeField32> VmExtension<F> for Rv32M {
    type Executor = Rv32MExecutor<F>;
    type Periphery = Rv32MPeriphery<F>;

    fn build(
        &self,
        builder: &mut VmInventoryBuilder<F>,
    ) -> Result<VmInventory<Rv32MExecutor<F>, Rv32MPeriphery<F>>, VmInventoryError> {
        let mut inventory = VmInventory::new();
        let SystemPort {
            execution_bus,
            program_bus,
            memory_bridge,
        } = builder.system_port();
        let offline_memory = builder.system_base().offline_memory();

        let bitwise_lu_chip = if let Some(&chip) = builder
            .find_chip::<SharedBitwiseOperationLookupChip<8>>()
            .first()
        {
            chip.clone()
        } else {
            let bitwise_lu_bus = BitwiseOperationLookupBus::new(builder.new_bus_idx());
            let chip = SharedBitwiseOperationLookupChip::new(bitwise_lu_bus);
            inventory.add_periphery_chip(chip.clone());
            chip
        };

        let range_tuple_checker = if let Some(chip) = builder
            .find_chip::<SharedRangeTupleCheckerChip<2>>()
            .into_iter()
            .find(|c| {
                c.bus().sizes[0] >= self.range_tuple_checker_sizes[0]
                    && c.bus().sizes[1] >= self.range_tuple_checker_sizes[1]
            }) {
            chip.clone()
        } else {
            let range_tuple_bus =
                RangeTupleCheckerBus::new(builder.new_bus_idx(), self.range_tuple_checker_sizes);
            let chip = SharedRangeTupleCheckerChip::new(range_tuple_bus);
            inventory.add_periphery_chip(chip.clone());
            chip
        };

        let mul_chip = Rv32MultiplicationChip::new(
            Rv32MultAdapterChip::new(execution_bus, program_bus, memory_bridge),
            MultiplicationCoreChip::new(range_tuple_checker.clone(), MulOpcode::CLASS_OFFSET),
            offline_memory.clone(),
        );
        inventory.add_executor(mul_chip, MulOpcode::iter().map(|x| x.global_opcode()))?;

        let mul_h_chip = Rv32MulHChip::new(
            Rv32MultAdapterChip::new(execution_bus, program_bus, memory_bridge),
            MulHCoreChip::new(bitwise_lu_chip.clone(), range_tuple_checker.clone()),
            offline_memory.clone(),
        );
        inventory.add_executor(mul_h_chip, MulHOpcode::iter().map(|x| x.global_opcode()))?;

        let div_rem_chip = Rv32DivRemChip::new(
            Rv32MultAdapterChip::new(execution_bus, program_bus, memory_bridge),
            DivRemCoreChip::new(
                bitwise_lu_chip.clone(),
                range_tuple_checker.clone(),
                DivRemOpcode::CLASS_OFFSET,
            ),
            offline_memory.clone(),
        );
        inventory.add_executor(
            div_rem_chip,
            DivRemOpcode::iter().map(|x| x.global_opcode()),
        )?;

        Ok(inventory)
    }
}

impl<F: PrimeField32> VmExtension<F> for Rv32Io {
    type Executor = Rv32IoExecutor<F>;
    type Periphery = Rv32IoPeriphery<F>;

    fn build(
        &self,
        builder: &mut VmInventoryBuilder<F>,
    ) -> Result<VmInventory<Self::Executor, Self::Periphery>, VmInventoryError> {
        let mut inventory = VmInventory::new();
        let SystemPort {
            execution_bus,
            program_bus,
            memory_bridge,
        } = builder.system_port();
        let offline_memory = builder.system_base().offline_memory();

        let bitwise_lu_chip = if let Some(&chip) = builder
            .find_chip::<SharedBitwiseOperationLookupChip<8>>()
            .first()
        {
            chip.clone()
        } else {
            let bitwise_lu_bus = BitwiseOperationLookupBus::new(builder.new_bus_idx());
            let chip = SharedBitwiseOperationLookupChip::new(bitwise_lu_bus);
            inventory.add_periphery_chip(chip.clone());
            chip
        };

        let mut hintstore_chip = Rv32HintStoreChip::new(
            execution_bus,
            program_bus,
            bitwise_lu_chip.clone(),
            memory_bridge,
            offline_memory.clone(),
            builder.system_config().memory_config.pointer_max_bits,
            Rv32HintStoreOpcode::CLASS_OFFSET,
        );
        hintstore_chip.set_streams(builder.streams().clone());

        inventory.add_executor(
            hintstore_chip,
            Rv32HintStoreOpcode::iter().map(|x| x.global_opcode()),
        )?;

        Ok(inventory)
    }
}

/// Phantom sub-executors
mod phantom {
    use eyre::bail;
    use openvm_circuit::{
        arch::{PhantomSubExecutor, Streams},
        system::memory::MemoryController,
    };
    use openvm_instructions::PhantomDiscriminant;
    use openvm_stark_backend::p3_field::{Field, PrimeField32};
    use rand::{rngs::OsRng, Rng};

    use crate::adapters::unsafe_read_rv32_register;

    pub struct Rv32HintInputSubEx;
    pub struct Rv32HintRandomSubEx {
        rng: OsRng,
    }
    impl Rv32HintRandomSubEx {
        pub fn new() -> Self {
            Self { rng: OsRng }
        }
    }
    pub struct Rv32PrintStrSubEx;
    pub struct Rv32HintLoadByKeySubEx;

    impl<F: Field> PhantomSubExecutor<F> for Rv32HintInputSubEx {
        fn phantom_execute(
            &mut self,
            _: &MemoryController<F>,
            streams: &mut Streams<F>,
            _: PhantomDiscriminant,
            _: F,
            _: F,
            _: u16,
        ) -> eyre::Result<()> {
            let mut hint = match streams.input_stream.pop_front() {
                Some(hint) => hint,
                None => {
                    bail!("EndOfInputStream");
                }
            };
            streams.hint_stream.clear();
            streams.hint_stream.extend(
                (hint.len() as u32)
                    .to_le_bytes()
                    .iter()
                    .map(|b| F::from_canonical_u8(*b)),
            );
            // Extend by 0 for 4 byte alignment
            let capacity = hint.len().div_ceil(4) * 4;
            hint.resize(capacity, F::ZERO);
            streams.hint_stream.extend(hint);
            Ok(())
        }
    }

    impl<F: PrimeField32> PhantomSubExecutor<F> for Rv32HintRandomSubEx {
        fn phantom_execute(
            &mut self,
            memory: &MemoryController<F>,
            streams: &mut Streams<F>,
            _: PhantomDiscriminant,
            a: F,
            _: F,
            _: u16,
        ) -> eyre::Result<()> {
            let len = unsafe_read_rv32_register(memory, a) as usize;
            streams.hint_stream.clear();
            streams.hint_stream.extend(
                std::iter::repeat_with(|| F::from_canonical_u8(self.rng.gen::<u8>())).take(len * 4),
            );
            Ok(())
        }
    }

    impl<F: PrimeField32> PhantomSubExecutor<F> for Rv32PrintStrSubEx {
        fn phantom_execute(
            &mut self,
            memory: &MemoryController<F>,
            _: &mut Streams<F>,
            _: PhantomDiscriminant,
            a: F,
            b: F,
            _: u16,
        ) -> eyre::Result<()> {
            let rd = unsafe_read_rv32_register(memory, a);
            let rs1 = unsafe_read_rv32_register(memory, b);
            let bytes = (0..rs1)
                .map(|i| -> eyre::Result<u8> {
                    let val = memory.unsafe_read_cell(F::TWO, F::from_canonical_u32(rd + i));
                    let byte: u8 = val.as_canonical_u32().try_into()?;
                    Ok(byte)
                })
                .collect::<eyre::Result<Vec<u8>>>()?;
            let peeked_str = String::from_utf8(bytes)?;
            print!("{peeked_str}");
            Ok(())
        }
    }

    impl<F: PrimeField32> PhantomSubExecutor<F> for Rv32HintLoadByKeySubEx {
        fn phantom_execute(
            &mut self,
            memory: &MemoryController<F>,
            streams: &mut Streams<F>,
            _: PhantomDiscriminant,
            a: F,
            b: F,
            _: u16,
        ) -> eyre::Result<()> {
            let ptr = unsafe_read_rv32_register(memory, a);
            let len = unsafe_read_rv32_register(memory, b);
            let key: Vec<u8> = (0..len)
                .map(|i| {
                    memory
                        .unsafe_read_cell(F::TWO, F::from_canonical_u32(ptr + i))
                        .as_canonical_u32() as u8
                })
                .collect();
            if let Some(val) = streams.kv_store.get(&key) {
                let to_push = hint_load_by_key_decode::<F>(val);
                for input in to_push.into_iter().rev() {
                    streams.input_stream.push_front(input);
                }
            } else {
                bail!("Rv32HintLoadByKey: key not found");
            }
            Ok(())
        }
    }

    pub fn hint_load_by_key_decode<F: PrimeField32>(value: &[u8]) -> Vec<Vec<F>> {
        let mut offset = 0;
        let len = extract_u32(value, offset) as usize;
        offset += 4;
        let mut ret = Vec::with_capacity(len);
        for _ in 0..len {
            let v_len = extract_u32(value, offset) as usize;
            offset += 4;
            let v = (0..v_len)
                .map(|_| {
                    let ret = F::from_canonical_u32(extract_u32(value, offset));
                    offset += 4;
                    ret
                })
                .collect();
            ret.push(v);
        }
        ret
    }

    fn extract_u32(value: &[u8], offset: usize) -> u32 {
        u32::from_le_bytes(value[offset..offset + 4].try_into().unwrap())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FrameBus {
    pub inner: PermutationCheckBus,
}

impl FrameBus {
    pub const fn new(index: BusIndex) -> Self {
        Self {
            inner: PermutationCheckBus::new(index),
        }
    }

    #[inline(always)]
    pub fn index(&self) -> BusIndex {
        self.inner.index
    }
}

#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Default, AlignedBorrow, Serialize, Deserialize, StructReflection,
)]
pub struct FrameState<T> {
    pub pc: T,
    pub fp: T,
}

#[derive(Copy, Clone, Debug)]
pub struct FrameBridge {
    frame_bus: FrameBus,
}

impl FrameBridge {
    pub fn new(frame_bus: FrameBus) -> Self {
        Self { frame_bus }
    }
}

pub struct FrameBridgeInteractor<AB: InteractionBuilder> {
    frame_bus: FrameBus,
    from_state: FrameState<AB::Expr>,
    to_state: FrameState<AB::Expr>,
}

impl<T> FrameState<T> {
    pub fn new(pc: impl Into<T>, fp: impl Into<T>) -> Self {
        Self {
            pc: pc.into(),
            fp: fp.into(),
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_iter<I: Iterator<Item = T>>(iter: &mut I) -> Self {
        let mut next = || iter.next().unwrap();
        Self {
            pc: next(),
            fp: next(),
        }
    }

    pub fn flatten(self) -> [T; 2] {
        [self.pc, self.fp]
    }

    pub fn get_width() -> usize {
        2
    }

    pub fn map<U: Clone, F: Fn(T) -> U>(self, function: F) -> FrameState<U> {
        FrameState::from_iter(&mut self.flatten().map(function).into_iter())
    }
}

impl FrameBus {
    /// Caller must constrain that `enabled` is boolean.
    pub fn execute<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        enabled: impl Into<AB::Expr>,
        prev_state: FrameState<impl Into<AB::Expr>>,
        next_state: FrameState<impl Into<AB::Expr>>,
    ) {
        let enabled = enabled.into();
        self.inner.receive(
            builder,
            [prev_state.pc.into(), prev_state.fp.into()],
            enabled.clone(),
        );
        self.inner.send(
            builder,
            [next_state.pc.into(), next_state.fp.into()],
            enabled,
        );
    }
}

pub struct VmChipWrapperWom<F, A: VmAdapterChipWom<F>, C: VmCoreChipWom<F, A::Interface>> {
    pub adapter: A,
    pub core: C,
    pub records: Vec<(A::ReadRecord, A::WriteRecord, C::Record)>,
    offline_memory: Arc<Mutex<OfflineMemory<F>>>,
}

const DEFAULT_RECORDS_CAPACITY: usize = 1 << 5;

impl<F, A, C> VmChipWrapperWom<F, A, C>
where
    A: VmAdapterChipWom<F>,
    C: VmCoreChipWom<F, A::Interface>,
{
    pub fn new(adapter: A, core: C, offline_memory: Arc<Mutex<OfflineMemory<F>>>) -> Self {
        Self {
            adapter,
            core,
            records: Vec::with_capacity(DEFAULT_RECORDS_CAPACITY),
            offline_memory,
        }
    }
}

pub trait VmAdapterChipWom<F> {
    /// Records generated by adapter before main instruction execution
    type ReadRecord: Send + Serialize + DeserializeOwned;
    /// Records generated by adapter after main instruction execution
    type WriteRecord: Send + Serialize + DeserializeOwned;
    /// AdapterAir should not have public values
    type Air: BaseAir<F> + Clone;

    type Interface: VmAdapterInterface<F>;

    /// Given instruction, perform memory reads and return only the read data that the integrator
    /// needs to use. This is called at the start of instruction execution.
    ///
    /// The implementer may choose to store data in the `Self::ReadRecord` struct, for example in
    /// an [Option], which will later be sent to the `postprocess` method.
    #[allow(clippy::type_complexity)]
    fn preprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        fp: usize,
        instruction: &Instruction<F>,
    ) -> ResultVm<(
        <Self::Interface as VmAdapterInterface<F>>::Reads,
        Self::ReadRecord,
    )>;

    /// Given instruction and the data to write, perform memory writes and return the `(record,
    /// next_timestamp)` of the full adapter record for this instruction. This is guaranteed to
    /// be called after `preprocess`.
    fn postprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
        from_frame: FrameState<u32>,
        output: AdapterRuntimeContextWom<F, Self::Interface>,
        read_record: &Self::ReadRecord,
    ) -> ResultVm<(ExecutionState<u32>, Self::WriteRecord)>;

    /// Populates `row_slice` with values corresponding to `record`.
    /// The provided `row_slice` will have length equal to `self.air().width()`.
    /// This function will be called for each row in the trace which is being used, and all other
    /// rows in the trace will be filled with zeroes.
    fn generate_trace_row(
        &self,
        row_slice: &mut [F],
        read_record: Self::ReadRecord,
        write_record: Self::WriteRecord,
        memory: &OfflineMemory<F>,
    );

    fn air(&self) -> &Self::Air;
}

pub struct AdapterRuntimeContextWom<T, I: VmAdapterInterface<T>> {
    /// Leave as `None` to allow the adapter to decide the `to_pc` automatically.
    pub to_pc: Option<u32>,
    pub to_fp: Option<u32>,
    pub writes: I::Writes,
}

impl<T, I: VmAdapterInterface<T>> AdapterRuntimeContextWom<T, I> {
    /// Leave `to_pc` as `None` to allow the adapter to decide the `to_pc` automatically.
    pub fn without_pc_fp(writes: impl Into<I::Writes>) -> Self {
        Self {
            to_pc: None,
            to_fp: None,
            writes: writes.into(),
        }
    }
}

pub trait VmCoreChipWom<F, I: VmAdapterInterface<F>> {
    /// Minimum data that must be recorded to be able to generate trace for one row of
    /// `PrimitiveAir`.
    type Record: Send + Serialize + DeserializeOwned;
    /// The primitive AIR with main constraints that do not depend on memory and other
    /// architecture-specifics.
    type Air: BaseAirWithPublicValues<F> + Clone;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        from_pc: u32,
        from_fp: u32,
        reads: I::Reads,
    ) -> ResultVm<(AdapterRuntimeContextWom<F, I>, Self::Record)>;

    fn get_opcode_name(&self, opcode: usize) -> String;

    /// Populates `row_slice` with values corresponding to `record`.
    /// The provided `row_slice` will have length equal to `self.air().width()`.
    /// This function will be called for each row in the trace which is being used, and all other
    /// rows in the trace will be filled with zeroes.
    fn generate_trace_row(&self, row_slice: &mut [F], record: Self::Record);

    /// Returns a list of public values to publish.
    fn generate_public_values(&self) -> Vec<F> {
        vec![]
    }

    fn air(&self) -> &Self::Air;

    /// Finalize the trace, especially the padded rows if the all-zero rows don't satisfy the
    /// constraints. This is done **after** records are consumed and the trace matrix is
    /// generated. Most implementations should just leave the default implementation if padding
    /// with rows of all 0s satisfies the constraints.
    fn finalize(&self, _trace: &mut RowMajorMatrix<F>, _num_records: usize) {
        // do nothing by default
    }
}

impl<F, A, M> InstructionExecutorTrait<F> for VmChipWrapperWom<F, A, M>
where
    F: PrimeField32,
    A: VmAdapterChipWom<F> + Send + Sync,
    M: VmCoreChipWom<F, A::Interface> + Send + Sync,
{
    fn execute(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
    ) -> ResultVm<ExecutionState<u32>> {
        let fp = 0;
        let (reads, read_record) = self.adapter.preprocess(memory, fp, instruction)?;
        let (output, core_record) =
            self.core
                .execute_instruction(instruction, from_state.pc, fp as u32, reads)?;
        let (to_state, write_record) = self.adapter.postprocess(
            memory,
            instruction,
            from_state,
            FrameState::default(),
            output,
            &read_record,
        )?;
        self.records.push((read_record, write_record, core_record));
        Ok(to_state)
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        self.core.get_opcode_name(opcode)
    }
}
