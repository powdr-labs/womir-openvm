use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use derive_more::derive::From;
use openvm_circuit::{
    arch::{
        ExecutorInventoryBuilder, ExecutorInventoryError, PhantomSubExecutor, VmCircuitExtension,
        VmExecutionExtension, AirInventory, AirInventoryError,
    },
    system::{memory::online::GuestMemory, phantom::PhantomChip},
};
use openvm_circuit_derive::AnyEnum;
use openvm_circuit_primitives::{
    bitwise_op_lookup::SharedBitwiseOperationLookupChip,
    range_tuple::SharedRangeTupleCheckerChip,
};
use openvm_circuit_primitives_derive::{Chip, ChipUsageGetter};
use openvm_instructions::PhantomDiscriminant;
use openvm_rv32im_circuit::*;
use rand::rngs::StdRng;
use openvm_stark_backend::{p3_field::PrimeField32, config::{StarkGenericConfig, Val}};
use openvm_womir_transpiler::{
    AllocateFrameOpcode, BaseAlu64Opcode, BaseAluOpcode, ConstOpcodes, CopyIntoFrameOpcode,
    DivRem64Opcode, DivRemOpcode, Eq64Opcode, EqOpcode, HintStoreOpcode, JaafOpcode, JumpOpcode,
    LessThan64Opcode, LessThanOpcode, LoadStoreOpcode, Mul64Opcode, MulOpcode, Phantom,
    Shift64Opcode, ShiftOpcode,
};

use serde::{Deserialize, Serialize};
use strum::IntoEnumIterator;

use crate::consts::ConstsCoreChipWom;
use crate::copy_into_frame::CopyIntoFrameCoreChipWom;
use crate::loadstore::LoadStoreChip;
use crate::{adapters::frame_allocator::FrameAllocator, allocate_frame::AllocateFrameCoreChipWom};
use crate::{adapters::*, wom_traits::*, *};

const DEFAULT_INIT_FP: u32 = 0;

// ============ Extension Implementations ============

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WomirI<F> {
    #[serde(default = "default_range_tuple_checker_sizes")]
    pub range_tuple_checker_sizes: [u32; 2],
    // TODO: shouldn't this be AtomicU32 instead of Mutex<u32>?
    // technically mutex is more robust, but I don't think it is
    // possible for two instructions to be executed in parallel,
    // so atomic might be enough...
    fp: Arc<Mutex<u32>>,
    frame_allocator: Arc<Mutex<FrameAllocator>>,
    frame_stack: Arc<Mutex<Vec<u32>>>,
    wom_controller: Arc<Mutex<WomController<F>>>,
}

impl<F: PrimeField32> Default for WomirI<F> {
    fn default() -> Self {
        Self {
            range_tuple_checker_sizes: default_range_tuple_checker_sizes(),
            // The entry frame starts at fp=0:
            fp: Arc::new(Mutex::new(DEFAULT_INIT_FP)),
            frame_stack: Arc::new(Mutex::new(vec![0])),
            frame_allocator: Arc::new(Mutex::new(Self::new_frame_allocator(
                // Reserve range 0..8 that is used by the startup code.
                [(0, 8)].into(),
            ))),
            wom_controller: Arc::new(Mutex::new(WomController::new())),
        }
    }
}

impl<F: PrimeField32> WomirI<F> {
    fn new_frame_allocator(existing_allocations: BTreeMap<u32, u32>) -> FrameAllocator {
        FrameAllocator::new(
            // The frame pointer is a field element, so the field limits its size.
            F::ORDER_U32 - 1,
            existing_allocations,
        )
    }

    /// Carry over the state that persists across segments.
    fn prepare_new_segment(&self) {
        // Reset the frame allocator.
        let mut frame_allocator = self.frame_allocator.lock().unwrap();
        let old_ranges = frame_allocator.get_allocated_ranges();

        // Keep all ranges that are in the frame stack.
        let remaining_ranges: BTreeMap<u32, u32> = self
            .frame_stack
            .lock()
            .unwrap()
            .iter()
            .map(|fp| (*fp, *old_ranges.get(fp).unwrap()))
            .collect();

        // Clear all the memory ranges outside of the remaining frames.
        self.wom_controller
            .lock()
            .unwrap()
            .clear_unused(remaining_ranges.iter().map(|(s, sz)| (*s, *sz)));

        // Create the new frame allocator.
        *frame_allocator = Self::new_frame_allocator(remaining_ranges);
    }
}

fn default_range_tuple_checker_sizes() -> [u32; 2] {
    // doubled from original openvm sizes to allow 64bit
    [1 << 8, 16 * (1 << 8)]
}

// ============ Executor and Periphery Enums for Extension ============

#[derive(ChipUsageGetter, Chip, InstructionExecutor, From, AnyEnum)]
pub enum WomirIExecutor<F: PrimeField32> {
    BaseAlu(WomBaseAluChip<F>),
    BaseAlu64(WomBaseAlu64Chip<F>),
    Jaaf(JaafChipWom<F>),
    Jump(JumpChipWom<F>),
    AllocateFrame(AllocateFrameChipWom<F>),
    CopyIntoFrame(CopyIntoFrameChipWom<F>),
    Const32(ConstsChipWom<F>),
    LessThan(LessThanChipWom<F>),
    LessThan64(LessThan64ChipWom<F>),
    HintStore(HintStoreChip<F>),
    Multiplication(WomMultiplicationChip<F>),
    Multiplication64(WomMultiplication64Chip<F>),
    DivRem(WomDivRemChip<F>),
    DivRem64(WomDivRem64Chip<F>),
    Shift(WomShiftChip<F>),
    Shift64(WomShift64Chip<F>),
    LoadStore(LoadStoreChip<F>),
    Eq(EqChipWom<F>),
    Eq64(Eq64ChipWom<F>),
    LoadSignExtend(LoadSignExtendChip<F>),
}

#[derive(From, ChipUsageGetter, Chip, AnyEnum)]
pub enum WomirIPeriphery<F: PrimeField32> {
    BitwiseOperationLookup(SharedBitwiseOperationLookupChip<8>),
    /// Only needed for multiplication extension
    RangeTupleChecker(SharedRangeTupleCheckerChip<2>),
    // We put this only to get the <F> generic to work
    Phantom(PhantomChip<F>),
}

// ============ VmExtension Implementations ============

impl<F: PrimeField32> VmExecutionExtension<F> for WomirI<F> {
    type Executor = WomirIExecutor<F>;

    fn extend_execution(
        &self,
        builder: &mut ExecutorInventoryBuilder<F, Self::Executor>,
    ) -> Result<(), ExecutorInventoryError> {
        // TODO: Implement executor registration
        // This needs to be rewritten to use the new Executor pattern
        // For now, we'll register phantom sub-executors which are still supported
        
        self.prepare_new_segment();
        
        // Register phantom sub-executors
        builder.add_phantom_sub_executor(
            phantom::HintInputSubEx,
            PhantomDiscriminant(Phantom::HintInput as u16),
        )?;
        builder.add_phantom_sub_executor(
            phantom::HintRandomSubEx::new(self.wom_controller.clone()),
            PhantomDiscriminant(Phantom::HintRandom as u16),
        )?;
        builder.add_phantom_sub_executor(
            phantom::PrintStrSubEx {
                wom: self.wom_controller.clone(),
                fp: self.fp.clone(),
            },
            PhantomDiscriminant(Phantom::PrintStr as u16),
        )?;
        builder.add_phantom_sub_executor(
            phantom::HintLoadByKeySubEx {
                wom: self.wom_controller.clone(),
            },
            PhantomDiscriminant(Phantom::HintLoadByKey as u16),
        )?;
        
        // TODO: Register all the WOM executors here following the pattern from rv32im
        // For example:
        // let base_alu = WomBaseAluExecutor::new(...);
        // builder.add_executor(base_alu, BaseAluOpcode::iter().map(|x| x.global_opcode()))?;
        
        Ok(())
    }
}

impl<SC: StarkGenericConfig> VmCircuitExtension<SC> for WomirI<Val<SC>>
where
    Val<SC>: PrimeField32,
{
    fn extend_circuit(&self, _inventory: &mut AirInventory<SC>) -> Result<(), AirInventoryError> {
        // TODO: Implement AIR registration
        // This needs to register all the AIRs for the WOM extension
        // For now, return Ok to allow compilation
        Ok(())
    }
}

// The old build method is commented out for reference
/*
impl<F: PrimeField32> VmExtension<F> for WomirI<F> {
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
        let pointer_max_bits = builder.system_config().memory_config.pointer_max_bits;

        let shared_fp = self.fp.clone();
        let wom_controller = self.wom_controller.clone();
        let wom_bridge = WomBridge::new(builder.new_bus_idx());

        self.prepare_new_segment();

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
                wom_bridge,
                bitwise_lu_chip.clone(),
            ),
            BaseAluCoreChip::new(bitwise_lu_chip.clone(), BaseAluOpcode::CLASS_OFFSET),
            offline_memory.clone(),
            shared_fp.clone(),
            wom_controller.clone(),
        );
        inventory.add_executor(
            base_alu_chip,
            BaseAluOpcode::iter().map(|x| x.global_opcode()),
        )?;

        let base_alu_64_chip = WomBaseAlu64Chip::new(
            WomBaseAluAdapterChip::new(
                execution_bus,
                program_bus,
                frame_bus,
                memory_bridge,
                wom_bridge,
                bitwise_lu_chip.clone(),
            ),
            BaseAluCoreChip::new(bitwise_lu_chip.clone(), BaseAlu64Opcode::CLASS_OFFSET),
            offline_memory.clone(),
            shared_fp.clone(),
            wom_controller.clone(),
        );
        inventory.add_executor(
            base_alu_64_chip,
            BaseAlu64Opcode::iter().map(|x| x.global_opcode()),
        )?;

        let jaaf_chip = JaafChipWom::new(
            JaafAdapterChipWom::new(
                execution_bus,
                program_bus,
                frame_bus,
                self.frame_stack.clone(),
                memory_bridge,
                wom_bridge,
            ),
            JaafCoreChipWom::default(),
            offline_memory.clone(),
            shared_fp.clone(),
            wom_controller.clone(),
        );
        inventory.add_executor(jaaf_chip, JaafOpcode::iter().map(|x| x.global_opcode()))?;

        let jump_chip = JumpChipWom::new(
            JumpAdapterChipWom::new(
                execution_bus,
                program_bus,
                frame_bus,
                memory_bridge,
                wom_bridge,
            ),
            JumpCoreChipWom::default(),
            offline_memory.clone(),
            shared_fp.clone(),
            wom_controller.clone(),
        );
        inventory.add_executor(jump_chip, JumpOpcode::iter().map(|x| x.global_opcode()))?;

        let allocate_frame_chip = AllocateFrameChipWom::new(
            AllocateFrameAdapterChipWom::new(
                execution_bus,
                program_bus,
                frame_bus,
                memory_bridge,
                wom_bridge,
                self.frame_allocator.clone(),
                self.frame_stack.clone(),
            ),
            AllocateFrameCoreChipWom::default(),
            offline_memory.clone(),
            shared_fp.clone(),
            wom_controller.clone(),
        );
        inventory.add_executor(
            allocate_frame_chip,
            AllocateFrameOpcode::iter().map(|x| x.global_opcode()),
        )?;

        let copy_into_frame_chip = CopyIntoFrameChipWom::new(
            CopyIntoFrameAdapterChipWom::new(
                execution_bus,
                program_bus,
                frame_bus,
                memory_bridge,
                wom_bridge,
            ),
            CopyIntoFrameCoreChipWom::new(),
            offline_memory.clone(),
            shared_fp.clone(),
            wom_controller.clone(),
        );
        inventory.add_executor(
            copy_into_frame_chip,
            CopyIntoFrameOpcode::iter().map(|x| x.global_opcode()),
        )?;

        let consts_chip = ConstsChipWom::new(
            ConstsAdapterChipWom::new(
                execution_bus,
                program_bus,
                frame_bus,
                memory_bridge,
                wom_bridge,
            ),
            ConstsCoreChipWom::new(),
            offline_memory.clone(),
            shared_fp.clone(),
            wom_controller.clone(),
        );
        inventory.add_executor(consts_chip, ConstOpcodes::iter().map(|x| x.global_opcode()))?;

        let lt_chip = LessThanChipWom::new(
            WomBaseAluAdapterChip::new(
                execution_bus,
                program_bus,
                frame_bus,
                memory_bridge,
                wom_bridge,
                bitwise_lu_chip.clone(),
            ),
            LessThanCoreChip::new(bitwise_lu_chip.clone(), LessThanOpcode::CLASS_OFFSET),
            offline_memory.clone(),
            shared_fp.clone(),
            wom_controller.clone(),
        );
        inventory.add_executor(lt_chip, LessThanOpcode::iter().map(|x| x.global_opcode()))?;

        let lt_chip_64 = LessThan64ChipWom::new(
            WomBaseAluAdapterChip::new(
                execution_bus,
                program_bus,
                frame_bus,
                memory_bridge,
                wom_bridge,
                bitwise_lu_chip.clone(),
            ),
            LessThanCoreChip::new(bitwise_lu_chip.clone(), LessThan64Opcode::CLASS_OFFSET),
            offline_memory.clone(),
            shared_fp.clone(),
            wom_controller.clone(),
        );
        inventory.add_executor(
            lt_chip_64,
            LessThan64Opcode::iter().map(|x| x.global_opcode()),
        )?;

        let eq_chip = EqChipWom::new(
            WomBaseAluAdapterChip::new(
                execution_bus,
                program_bus,
                frame_bus,
                memory_bridge,
                wom_bridge,
                bitwise_lu_chip.clone(),
            ),
            EqCoreChip::new(EqOpcode::CLASS_OFFSET),
            offline_memory.clone(),
            shared_fp.clone(),
            wom_controller.clone(),
        );
        inventory.add_executor(eq_chip, EqOpcode::iter().map(|x| x.global_opcode()))?;

        let eq_chip_64 = Eq64ChipWom::new(
            WomBaseAluAdapterChip::new(
                execution_bus,
                program_bus,
                frame_bus,
                memory_bridge,
                wom_bridge,
                bitwise_lu_chip.clone(),
            ),
            EqCoreChip::new(Eq64Opcode::CLASS_OFFSET),
            offline_memory.clone(),
            shared_fp.clone(),
            wom_controller.clone(),
        );
        inventory.add_executor(eq_chip_64, Eq64Opcode::iter().map(|x| x.global_opcode()))?;

        let mut hintstore_chip = HintStoreChip::new(
            execution_bus,
            frame_bus,
            program_bus,
            bitwise_lu_chip.clone(),
            memory_bridge,
            offline_memory.clone(),
            shared_fp.clone(),
            wom_controller.clone(),
            builder.system_config().memory_config.pointer_max_bits,
            HintStoreOpcode::CLASS_OFFSET,
        );
        hintstore_chip.set_streams(builder.streams().clone());

        inventory.add_executor(
            hintstore_chip,
            HintStoreOpcode::iter().map(|x| x.global_opcode()),
        )?;

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

        let mul_chip = WomMultiplicationChip::new(
            WomBaseAluAdapterChip::new(
                execution_bus,
                program_bus,
                frame_bus,
                memory_bridge,
                wom_bridge,
                bitwise_lu_chip.clone(),
            ),
            MultiplicationCoreChip::new(range_tuple_checker.clone(), MulOpcode::CLASS_OFFSET),
            offline_memory.clone(),
            shared_fp.clone(),
            wom_controller.clone(),
        );
        inventory.add_executor(mul_chip, MulOpcode::iter().map(|x| x.global_opcode()))?;

        let mul_chip_64 = WomMultiplication64Chip::new(
            WomBaseAluAdapterChip::new(
                execution_bus,
                program_bus,
                frame_bus,
                memory_bridge,
                wom_bridge,
                bitwise_lu_chip.clone(),
            ),
            MultiplicationCoreChip::new(range_tuple_checker.clone(), Mul64Opcode::CLASS_OFFSET),
            offline_memory.clone(),
            shared_fp.clone(),
            wom_controller.clone(),
        );
        inventory.add_executor(mul_chip_64, Mul64Opcode::iter().map(|x| x.global_opcode()))?;

        let div_rem_chip = WomDivRemChip::new(
            WomBaseAluAdapterChip::new(
                execution_bus,
                program_bus,
                frame_bus,
                memory_bridge,
                wom_bridge,
                bitwise_lu_chip.clone(),
            ),
            DivRemCoreChip::new(
                bitwise_lu_chip.clone(),
                range_tuple_checker.clone(),
                DivRemOpcode::CLASS_OFFSET,
            ),
            offline_memory.clone(),
            shared_fp.clone(),
            wom_controller.clone(),
        );
        inventory.add_executor(
            div_rem_chip,
            DivRemOpcode::iter().map(|x| x.global_opcode()),
        )?;

        let div_rem_chip_64 = WomDivRem64Chip::new(
            WomBaseAluAdapterChip::new(
                execution_bus,
                program_bus,
                frame_bus,
                memory_bridge,
                wom_bridge,
                bitwise_lu_chip.clone(),
            ),
            DivRemCoreChip::new(
                bitwise_lu_chip.clone(),
                range_tuple_checker.clone(),
                DivRem64Opcode::CLASS_OFFSET,
            ),
            offline_memory.clone(),
            shared_fp.clone(),
            wom_controller.clone(),
        );
        inventory.add_executor(
            div_rem_chip_64,
            DivRem64Opcode::iter().map(|x| x.global_opcode()),
        )?;

        let shift_chip = WomShiftChip::new(
            WomBaseAluAdapterChip::new(
                execution_bus,
                program_bus,
                frame_bus,
                memory_bridge,
                wom_bridge,
                bitwise_lu_chip.clone(),
            ),
            ShiftCoreChip::new(
                bitwise_lu_chip.clone(),
                range_checker.clone(),
                ShiftOpcode::CLASS_OFFSET,
            ),
            offline_memory.clone(),
            shared_fp.clone(),
            wom_controller.clone(),
        );
        inventory.add_executor(shift_chip, ShiftOpcode::iter().map(|x| x.global_opcode()))?;

        let shift_64_chip = WomShift64Chip::new(
            WomBaseAluAdapterChip::new(
                execution_bus,
                program_bus,
                frame_bus,
                memory_bridge,
                wom_bridge,
                bitwise_lu_chip.clone(),
            ),
            ShiftCoreChip::new(
                bitwise_lu_chip.clone(),
                range_checker.clone(),
                Shift64Opcode::CLASS_OFFSET,
            ),
            offline_memory.clone(),
            shared_fp.clone(),
            wom_controller.clone(),
        );
        inventory.add_executor(
            shift_64_chip,
            Shift64Opcode::iter().map(|x| x.global_opcode()),
        )?;

        let load_store_chip = LoadStoreChip::new(
            Rv32LoadStoreAdapterChip::new(
                execution_bus,
                program_bus,
                frame_bus,
                memory_bridge,
                wom_bridge,
                pointer_max_bits,
                range_checker.clone(),
            ),
            LoadStoreCoreChip::new(LoadStoreOpcode::CLASS_OFFSET),
            offline_memory.clone(),
            shared_fp.clone(),
            wom_controller.clone(),
        );
        inventory.add_executor(
            load_store_chip,
            LoadStoreOpcode::iter()
                .take(LoadStoreOpcode::STOREB as usize + 1)
                .map(|x| x.global_opcode()),
        )?;

        let load_sign_extend_chip = LoadSignExtendChip::new(
            Rv32LoadStoreAdapterChip::new(
                execution_bus,
                program_bus,
                frame_bus,
                memory_bridge,
                wom_bridge,
                pointer_max_bits,
                range_checker.clone(),
            ),
            LoadSignExtendCoreChip::new(range_checker.clone()),
            offline_memory.clone(),
            shared_fp.clone(),
            wom_controller.clone(),
        );
        inventory.add_executor(
            load_sign_extend_chip,
            [LoadStoreOpcode::LOADB, LoadStoreOpcode::LOADH].map(|x| x.global_opcode()),
        )?;

        // There is no downside to adding phantom sub-executors, so we do it in the base extension.
        builder.add_phantom_sub_executor(
            phantom::HintInputSubEx,
            PhantomDiscriminant(Phantom::HintInput as u16),
        )?;
        builder.add_phantom_sub_executor(
            phantom::HintRandomSubEx::new(wom_controller.clone()),
            PhantomDiscriminant(Phantom::HintRandom as u16),
        )?;
        builder.add_phantom_sub_executor(
            phantom::PrintStrSubEx {
                wom: wom_controller.clone(),
                fp: shared_fp.clone(),
            },
            PhantomDiscriminant(Phantom::PrintStr as u16),
        )?;
        builder.add_phantom_sub_executor(
            phantom::HintLoadByKeySubEx {
                wom: wom_controller,
            },
            PhantomDiscriminant(Phantom::HintLoadByKey as u16),
        )?;

        Ok(inventory)
    }
}
*/

/// Phantom sub-executors
mod phantom {
    use std::sync::{Arc, Mutex};

    use eyre::bail;
    use openvm_circuit::{
        arch::{PhantomSubExecutor, Streams},
        system::memory::MemoryController,
    };
    use openvm_instructions::PhantomDiscriminant;
    use openvm_stark_backend::p3_field::{Field, PrimeField32};
    use rand::{Rng, rngs::OsRng};

    use crate::{WomController, adapters::unsafe_read_wom_register};

    pub struct HintInputSubEx;

    pub struct HintRandomSubEx<F> {
        wom: Arc<Mutex<WomController<F>>>,
        rng: OsRng,
    }
    impl<F> HintRandomSubEx<F> {
        pub fn new(wom: Arc<Mutex<WomController<F>>>) -> Self {
            Self { wom, rng: OsRng }
        }
    }

    pub struct PrintStrSubEx<F> {
        pub wom: Arc<Mutex<WomController<F>>>,
        pub fp: Arc<Mutex<u32>>,
    }

    pub struct HintLoadByKeySubEx<F> {
        pub wom: Arc<Mutex<WomController<F>>>,
    }

    impl<F: Field> PhantomSubExecutor<F> for HintInputSubEx {
        fn phantom_execute(
            &self,
            _: &GuestMemory,
            streams: &mut Streams<F>,
            _: &mut StdRng,
            _: PhantomDiscriminant,
            _: u32,
            _: u32,
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

    impl<F: PrimeField32> PhantomSubExecutor<F> for HintRandomSubEx<F> {
        fn phantom_execute(
            &self,
            _memory: &GuestMemory,
            streams: &mut Streams<F>,
            _: &mut StdRng,
            _: PhantomDiscriminant,
            a: u32,
            _: u32,
            _: u16,
        ) -> eyre::Result<()> {
            let len = unsafe_read_wom_register(&self.wom.lock().unwrap(), F::from_canonical_u32(a)) as usize;
            streams.hint_stream.clear();
            streams.hint_stream.extend(
                std::iter::repeat_with(|| F::from_canonical_u8(self.rng.r#gen::<u8>()))
                    .take(len * 4),
            );
            Ok(())
        }
    }

    impl<F: PrimeField32> PhantomSubExecutor<F> for PrintStrSubEx<F> {
        fn phantom_execute(
            &self,
            memory: &GuestMemory,
            _: &mut Streams<F>,
            _: &mut StdRng,
            _: PhantomDiscriminant,
            a: u32,
            b: u32,
            mem_start_imm: u16,
        ) -> eyre::Result<()> {
            // TODO mem_start_imm may be larger than u16 in some cases.
            let mem_start_imm = mem_start_imm as u32;
            let fp = self.fp.lock().unwrap();
            let fp_f = F::from_canonical_u32(*fp);
            let a_f = F::from_canonical_u32(a);
            let b_f = F::from_canonical_u32(b);
            let rd = unsafe_read_wom_register(&self.wom.lock().unwrap(), a_f + fp_f);
            let rs1 = unsafe_read_wom_register(&self.wom.lock().unwrap(), b_f + fp_f);
            let bytes = (0..rs1)
                .map(|i| -> eyre::Result<u8> {
                    let val = memory
                        .unsafe_read_cell(F::TWO, F::from_canonical_u32(mem_start_imm + rd + i));
                    let byte: u8 = val.as_canonical_u32().try_into()?;
                    Ok(byte)
                })
                .collect::<eyre::Result<Vec<u8>>>()?;
            let peeked_str = String::from_utf8(bytes)?;
            print!("{peeked_str}");
            Ok(())
        }
    }

    impl<F: PrimeField32> PhantomSubExecutor<F> for HintLoadByKeySubEx<F> {
        fn phantom_execute(
            &self,
            memory: &GuestMemory,
            streams: &mut Streams<F>,
            _: &mut StdRng,
            _: PhantomDiscriminant,
            a: u32,
            b: u32,
            _: u16,
        ) -> eyre::Result<()> {
            let ptr = unsafe_read_wom_register(&self.wom.lock().unwrap(), F::from_canonical_u32(a));
            let len = unsafe_read_wom_register(&self.wom.lock().unwrap(), F::from_canonical_u32(b));
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
