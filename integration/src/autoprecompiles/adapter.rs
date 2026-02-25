use std::fmt::Display;
use std::hash::Hash;
use std::iter::once;
use std::sync::Arc;

use derive_more::From;
use openvm_instructions::instruction::Instruction as OpenVmInstruction;
use openvm_instructions::program::{DEFAULT_PC_STEP, Program as OpenVmProgram};
use openvm_stark_backend::p3_field::{FieldAlgebra, PrimeField32};
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use powdr_autoprecompiles::InstructionHandler;
use powdr_autoprecompiles::adapter::{Adapter, AdapterApc};
use powdr_autoprecompiles::blocks::{Instruction, PcStep, Program};
use powdr_autoprecompiles::execution::ExecutionState;
use powdr_number::{BabyBearField, FieldElement, LargeInt};
use powdr_openvm::bus_map::OpenVmBusType;
use powdr_openvm::extraction_utils::{AirWidthsDiff, get_air_metrics};
use powdr_openvm_bus_interaction_handler::OpenVmBusInteractionHandler;
use powdr_openvm_bus_interaction_handler::memory_bus_interaction::OpenVmMemoryBusInteraction;
use serde::{Deserialize, Serialize};

use crate::autoprecompiles::air::WomirAir;

use super::instruction_handler::WomirOriginalAirs;

// ---- Instruction wrapper ----

/// A newtype wrapper around `OpenVmInstruction` to implement the `Instruction` trait.
/// Same as the one in powdr-openvm, since WOMIR uses the same instruction format.
#[derive(Clone, Serialize, Deserialize)]
pub struct Instr<F>(pub OpenVmInstruction<F>);

impl<F: PrimeField32> Display for Instr<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // TODO: Improve formatting
        write!(
            f,
            "Instr({}, {}, {}, {}, {}, {}, {}, {})",
            self.0.opcode.as_usize(),
            self.0.a.as_canonical_u32(),
            self.0.b.as_canonical_u32(),
            self.0.c.as_canonical_u32(),
            self.0.d.as_canonical_u32(),
            self.0.e.as_canonical_u32(),
            self.0.f.as_canonical_u32(),
            self.0.g.as_canonical_u32(),
        )
    }
}

impl<F> PcStep for Instr<F> {
    fn pc_step() -> u32 {
        DEFAULT_PC_STEP
    }
}

impl<F: PrimeField32> Instruction<F> for Instr<F> {
    fn pc_lookup_row(&self, pc: u64) -> Vec<F> {
        let args = [
            self.0.opcode.to_field(),
            self.0.a,
            self.0.b,
            self.0.c,
            self.0.d,
            self.0.e,
            self.0.f,
            self.0.g,
        ];
        // The PC lookup row has the format: [pc, opcode, a, b, c, d, e, f, g]
        let pc = F::from_canonical_u32(pc.try_into().unwrap());
        once(pc).chain(args).collect()
    }
}

// ---- Program wrapper ----

/// A newtype wrapper around `OpenVmProgram` to implement the `Program` trait.
/// Same as the one in powdr-openvm, since WOMIR uses the same program format.
pub struct Prog<'a, F>(pub &'a OpenVmProgram<F>);

impl<'a, F> From<&'a OpenVmProgram<F>> for Prog<'a, F> {
    fn from(program: &'a OpenVmProgram<F>) -> Self {
        Prog(program)
    }
}

impl<F: PrimeField32> Program<Instr<F>> for Prog<'_, F> {
    fn base_pc(&self) -> u64 {
        self.0.pc_base as u64
    }

    fn instructions(&self) -> Box<dyn Iterator<Item = Instr<F>> + '_> {
        Box::new(
            self.0
                .instructions_and_debug_infos
                .iter()
                .filter_map(|x| x.as_ref().map(|i| Instr(i.0.clone()))),
        )
    }

    fn length(&self) -> u32 {
        self.0.instructions_and_debug_infos.len() as u32
    }
}

// ---- Execution state ----

/// Dummy implementation of `ExecutionState`. This is needed in the context
/// of optimistic precompiles, which are not supported yet, so this code should
/// never be called.
#[derive(From)]
pub struct WomirExecutionState {}

impl ExecutionState for WomirExecutionState {
    type RegisterAddress = ();
    type Value = ();

    fn pc(&self) -> Self::Value {
        unimplemented!()
    }

    fn reg(&self, _addr: &Self::RegisterAddress) -> Self::Value {
        unimplemented!()
    }

    fn value_limb(_value: Self::Value, _limb_index: usize) -> Self::Value {
        unimplemented!()
    }

    fn global_clk(&self) -> usize {
        unimplemented!()
    }
}

// ---- APC stats ----

/// Statistics about an autoprecompile.
// TODO: Identical to OvmApcStats in powdr-openvm, but reimplemented here because OvmApcStats is private.
#[derive(Clone, Serialize, Deserialize)]
pub struct ApcStats {
    pub widths: AirWidthsDiff,
}

impl ApcStats {
    fn new(widths: AirWidthsDiff) -> Self {
        Self { widths }
    }
}

// ---- Adapter ----

/// The autoprecompiles adapter for WOMIR-OpenVM.
pub struct WomirApcAdapter<'a> {
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a> Adapter for WomirApcAdapter<'a> {
    type PowdrField = BabyBearField;
    type Field = BabyBear;
    type InstructionHandler = WomirOriginalAirs<Self::Field>;
    type BusInteractionHandler = OpenVmBusInteractionHandler<Self::PowdrField>;
    type Program = Prog<'a, Self::Field>;
    type Instruction = Instr<Self::Field>;
    type MemoryBusInteraction<V: Ord + Clone + Eq + Display + Hash> =
        OpenVmMemoryBusInteraction<Self::PowdrField, V>;
    type CustomBusTypes = OpenVmBusType;
    type ApcStats = ApcStats;
    type AirId = String;
    type ExecutionState = WomirExecutionState;

    fn into_field(e: Self::PowdrField) -> Self::Field {
        BabyBear::from_canonical_u32(e.to_integer().try_into_u32().unwrap())
    }

    fn from_field(e: Self::Field) -> Self::PowdrField {
        BabyBearField::from(e.as_canonical_u32())
    }

    fn apc_stats(
        apc: Arc<AdapterApc<Self>>,
        instruction_handler: &Self::InstructionHandler,
    ) -> Self::ApcStats {
        // Get the metrics for the apc using the same degree bound as the one used for the instruction chips
        let apc_metrics = get_air_metrics(
            Arc::new(WomirAir::new(apc.clone())),
            instruction_handler.degree_bound().identities,
        );
        let width_after = apc_metrics.widths;

        // Sum up the metrics for each instruction
        let width_before = apc
            .instructions()
            .map(|instr| {
                instruction_handler
                    .get_instruction_metrics(instr.0.opcode)
                    .unwrap()
                    .widths
            })
            .sum();

        ApcStats::new(AirWidthsDiff::new(width_before, width_after))
    }
}
