use std::{
    cell::RefCell,
    sync::{Arc, Mutex},
};

use crate::PhantomChip;
use derive_more::derive::From;
use openvm_circuit::{
    arch::{
        InitFileGenerator, SystemConfig, SystemPort, VmExtension, VmInventory, VmInventoryBuilder,
        VmInventoryError,
    },
    // system::phantom::PhantomChip,
};
use openvm_circuit_derive::{AnyEnum, InstructionExecutor, VmConfig};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    range_tuple::{RangeTupleCheckerBus, SharedRangeTupleCheckerChip},
};
use openvm_circuit_primitives_derive::{Chip, ChipUsageGetter};
use openvm_instructions::{LocalOpcode, PhantomDiscriminant};
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_womir_transpiler::{
    AllocateFrameOpcode, BaseAluOpcode, ConstOpcodes, CopyIntoFrameOpcode, DivRemOpcode,
    HintStoreOpcode, JaafOpcode, JumpOpcode, LessThanOpcode, MulHOpcode, MulOpcode, Phantom,
    WomSystemOpcodes,
};

use serde::{Deserialize, Serialize};
use strum::IntoEnumIterator;

use crate::allocate_frame::AllocateFrameCoreChipWom;
use crate::consts::ConstsCoreChipWom;
use crate::copy_into_frame::CopyIntoFrameCoreChipWom;
use crate::{adapters::*, wom_traits::*, *};

const DEFAULT_INIT_FP: u32 = 0;

/// Config for a VM with base extension and IO extension
#[derive(Clone, Debug, VmConfig, derive_new::new, Serialize, Deserialize)]
pub struct WomirIConfig {
    #[system]
    pub system: SystemConfig,
    #[extension]
    pub base: WomirI,
}

// Default implementation uses no init file
impl InitFileGenerator for WomirIConfig {}

/// Config for a VM with base extension, IO extension, and multiplication extension
#[derive(Clone, Debug, Default, VmConfig, derive_new::new, Serialize, Deserialize)]
pub struct WomirImConfig {
    #[config]
    pub womir_i: WomirIConfig,
    #[extension]
    pub mul: WomirM,
}

// Default implementation uses no init file
impl InitFileGenerator for WomirImConfig {}

impl Default for WomirIConfig {
    fn default() -> Self {
        let system = SystemConfig::default().with_continuations();
        Self {
            system,
            base: Default::default(),
        }
    }
}

impl WomirIConfig {
    pub fn with_public_values(public_values: usize) -> Self {
        let system = SystemConfig::default()
            .with_continuations()
            .with_public_values(public_values);
        Self {
            system,
            base: Default::default(),
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
        }
    }
}

impl WomirImConfig {
    pub fn with_public_values(public_values: usize) -> Self {
        Self {
            womir_i: WomirIConfig::with_public_values(public_values),
            mul: Default::default(),
        }
    }

    pub fn with_public_values_and_segment_len(public_values: usize, segment_len: usize) -> Self {
        Self {
            womir_i: WomirIConfig::with_public_values_and_segment_len(public_values, segment_len),
            mul: Default::default(),
        }
    }
}

// ============ Extension Implementations ============

/// Extension similar to RISC-V 32-bit Base (RV32I)
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct WomirI;

/// Extension similar to RISC-V 32-bit Multiplication Extension (RV32M)
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct WomirM {
    #[serde(default = "default_range_tuple_checker_sizes")]
    pub range_tuple_checker_sizes: [u32; 2],
}

impl Default for WomirM {
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

#[derive(ChipUsageGetter, Chip, InstructionExecutor, From, AnyEnum)]
pub enum WomirIExecutor<F: PrimeField32> {
    BaseAlu(WomBaseAluChip<F>),
    Jaaf(JaafChipWom<F>),
    Jump(JumpChipWom<F>),
    AllocateFrame(AllocateFrameChipWom<F>),
    CopyIntoFrame(CopyIntoFrameChipWom<F>),
    Const32(ConstsChipWom<F>),
    LessThan(LessThanChipWom<F>),
    HintStore(HintStoreChip<F>),
    // Phantom(PhantomChip<F>),
    // Shift(Rv32ShiftChip<F>),
    // LoadStore(Rv32LoadStoreChip<F>),
    // LoadSignExtend(Rv32LoadSignExtendChip<F>),
    // BranchEqual(Rv32BranchEqualChip<F>),
    // BranchLessThan(Rv32BranchLessThanChip<F>),
    // JalLui(Rv32JalLuiChip<F>),
    // Jalr(Rv32JalrChip<F>),
    // Auipc(Rv32AuipcChip<F>),
}

#[derive(ChipUsageGetter, Chip, InstructionExecutor, From, AnyEnum)]
pub enum WomirMExecutor<F: PrimeField32> {
    Multiplication(Rv32MultiplicationChip<F>),
    MultiplicationHigh(Rv32MulHChip<F>),
    DivRem(Rv32DivRemChip<F>),
}

#[derive(From, ChipUsageGetter, Chip, AnyEnum)]
pub enum WomirIPeriphery<F: PrimeField32> {
    BitwiseOperationLookup(SharedBitwiseOperationLookupChip<8>),
    // We put this only to get the <F> generic to work
    Phantom(PhantomChip<F>),
}

#[derive(From, ChipUsageGetter, Chip, AnyEnum)]
pub enum WomirMPeriphery<F: PrimeField32> {
    BitwiseOperationLookup(SharedBitwiseOperationLookupChip<8>),
    /// Only needed for multiplication extension
    RangeTupleChecker(SharedRangeTupleCheckerChip<2>),
    // We put this only to get the <F> generic to work
    Phantom(PhantomChip<F>),
}

// ============ VmExtension Implementations ============

impl<F: PrimeField32> VmExtension<F> for WomirI {
    type Executor = WomirIExecutor<F>;
    type Periphery = WomirIPeriphery<F>;

    fn build(
        &self,
        builder: &mut VmInventoryBuilder<F>,
    ) -> Result<VmInventory<WomirIExecutor<F>, WomirIPeriphery<F>>, VmInventoryError> {
        let mut inventory = VmInventory::new();
        let SystemPort {
            execution_bus,
            program_bus,
            memory_bridge,
        } = builder.system_port();

        let frame_bus = FrameBus::new(builder.new_bus_idx());

        let range_checker = builder.system_base().range_checker_chip.clone();
        let offline_memory = builder.system_base().offline_memory();
        let _pointer_max_bits = builder.system_config().memory_config.pointer_max_bits;

        let shared_fp = Arc::new(Mutex::new(DEFAULT_INIT_FP));

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

        let base_alu_chip = WomBaseAluChip::new(
            WomBaseAluAdapterChip::new(
                execution_bus,
                program_bus,
                frame_bus,
                memory_bridge,
                bitwise_lu_chip.clone(),
            ),
            BaseAluCoreChipWom::new(bitwise_lu_chip.clone(), BaseAluOpcode::CLASS_OFFSET),
            offline_memory.clone(),
            shared_fp.clone(),
        );
        inventory.add_executor(
            base_alu_chip,
            BaseAluOpcode::iter().map(|x| x.global_opcode()),
        )?;

        let jaaf_chip = JaafChipWom::new(
            JaafAdapterChipWom::new(execution_bus, program_bus, frame_bus, memory_bridge),
            JaafCoreChipWom::new(bitwise_lu_chip.clone(), range_checker.clone()),
            offline_memory.clone(),
            shared_fp.clone(),
        );
        inventory.add_executor(jaaf_chip, JaafOpcode::iter().map(|x| x.global_opcode()))?;

        let jump_chip = JumpChipWom::new(
            JumpAdapterChipWom::new(execution_bus, program_bus, memory_bridge),
            JumpCoreChipWom::new(bitwise_lu_chip.clone(), range_checker.clone()),
            offline_memory.clone(),
            shared_fp.clone(),
        );
        inventory.add_executor(jump_chip, JumpOpcode::iter().map(|x| x.global_opcode()))?;

        let allocate_frame_chip = AllocateFrameChipWom::new(
            AllocateFrameAdapterChipWom::new(execution_bus, program_bus, frame_bus, memory_bridge),
            AllocateFrameCoreChipWom::new(bitwise_lu_chip.clone(), range_checker.clone()),
            offline_memory.clone(),
            shared_fp.clone(),
        );
        inventory.add_executor(
            allocate_frame_chip,
            AllocateFrameOpcode::iter().map(|x| x.global_opcode()),
        )?;

        let copy_into_frame_chip = CopyIntoFrameChipWom::new(
            CopyIntoFrameAdapterChipWom::new(execution_bus, program_bus, frame_bus, memory_bridge),
            CopyIntoFrameCoreChipWom::new(),
            offline_memory.clone(),
            shared_fp.clone(),
        );
        inventory.add_executor(
            copy_into_frame_chip,
            CopyIntoFrameOpcode::iter().map(|x| x.global_opcode()),
        )?;

        let consts_chip = ConstsChipWom::new(
            ConstsAdapterChipWom::new(execution_bus, program_bus, frame_bus, memory_bridge),
            ConstsCoreChipWom::new(),
            offline_memory.clone(),
            shared_fp.clone(),
        );
        inventory.add_executor(consts_chip, ConstOpcodes::iter().map(|x| x.global_opcode()))?;

        let lt_chip = LessThanChipWom::new(
            WomBaseAluAdapterChip::new(
                execution_bus,
                program_bus,
                frame_bus,
                memory_bridge,
                bitwise_lu_chip.clone(),
            ),
            LessThanCoreChip::new(bitwise_lu_chip.clone(), LessThanOpcode::CLASS_OFFSET),
            offline_memory.clone(),
            shared_fp.clone(),
        );
        inventory.add_executor(lt_chip, LessThanOpcode::iter().map(|x| x.global_opcode()))?;

        let mut hintstore_chip = HintStoreChip::new(
            execution_bus,
            frame_bus,
            program_bus,
            bitwise_lu_chip.clone(),
            memory_bridge,
            offline_memory.clone(),
            shared_fp.clone(),
            builder.system_config().memory_config.pointer_max_bits,
            HintStoreOpcode::CLASS_OFFSET,
        );
        hintstore_chip.set_streams(builder.streams().clone());

        inventory.add_executor(
            hintstore_chip,
            HintStoreOpcode::iter().map(|x| x.global_opcode()),
        )?;

        /*
                let streams = builder.streams().clone();
                let phantom_opcode = WomSystemOpcodes::PHANTOM.global_opcode();
                let mut phantom_chip =
                    PhantomChip::new(execution_bus, program_bus, WomSystemOpcodes::CLASS_OFFSET);
                println!(
                    "phantom opcode: {phantom_opcode:?}, offset: {}",
                    WomSystemOpcodes::CLASS_OFFSET
                );
                phantom_chip.set_streams(streams.clone());
                let _ = phantom_chip.add_sub_executor(
                    phantom::HintInputSubEx,
                    PhantomDiscriminant(Phantom::HintInput as u16),
                );
                inventory
                    .add_executor(phantom_chip, [phantom_opcode])
                    .unwrap();
        */

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
            phantom::HintInputSubEx,
            PhantomDiscriminant(Phantom::HintInput as u16),
        )?;
        builder.add_phantom_sub_executor(
            phantom::HintRandomSubEx::new(),
            PhantomDiscriminant(Phantom::HintRandom as u16),
        )?;
        builder.add_phantom_sub_executor(
            phantom::PrintStrSubEx,
            PhantomDiscriminant(Phantom::PrintStr as u16),
        )?;
        builder.add_phantom_sub_executor(
            phantom::HintLoadByKeySubEx,
            PhantomDiscriminant(Phantom::HintLoadByKey as u16),
        )?;

        Ok(inventory)
    }
}

impl<F: PrimeField32> VmExtension<F> for WomirM {
    type Executor = WomirMExecutor<F>;
    type Periphery = WomirMPeriphery<F>;

    fn build(
        &self,
        builder: &mut VmInventoryBuilder<F>,
    ) -> Result<VmInventory<WomirMExecutor<F>, WomirMPeriphery<F>>, VmInventoryError> {
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

    pub struct HintInputSubEx;
    pub struct HintRandomSubEx {
        rng: OsRng,
    }
    impl HintRandomSubEx {
        pub fn new() -> Self {
            Self { rng: OsRng }
        }
    }
    pub struct PrintStrSubEx;
    pub struct HintLoadByKeySubEx;

    impl<F: Field> PhantomSubExecutor<F> for HintInputSubEx {
        fn phantom_execute(
            &mut self,
            _: &MemoryController<F>,
            streams: &mut Streams<F>,
            _: PhantomDiscriminant,
            _: F,
            _: F,
            _: u16,
        ) -> eyre::Result<()> {
            println!("Inside PhantomSubExecutor::HintInputSubEx");
            let mut hint = match streams.input_stream.pop_front() {
                Some(hint) => hint,
                None => {
                    bail!("EndOfInputStream");
                }
            };
            println!("hint = {hint:?}");
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

    impl<F: PrimeField32> PhantomSubExecutor<F> for HintRandomSubEx {
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

    impl<F: PrimeField32> PhantomSubExecutor<F> for PrintStrSubEx {
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

    impl<F: PrimeField32> PhantomSubExecutor<F> for HintLoadByKeySubEx {
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
                bail!("HintLoadByKey: key not found");
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
