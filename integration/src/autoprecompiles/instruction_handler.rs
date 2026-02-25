use std::collections::{BTreeMap, HashMap};

use openvm_instructions::VmOpcode;
use powdr_autoprecompiles::evaluation::AirStats;
use powdr_autoprecompiles::symbolic_machine::SymbolicMachine;
use powdr_autoprecompiles::{DegreeBound, InstructionHandler};
use powdr_openvm::AirMetrics;
use serde::{Deserialize, Serialize};

use super::adapter::Instr;
use super::opcodes::branch_opcodes_set;

/// Maps WOMIR opcodes to their symbolic AIR machines.
/// Analogous to `OriginalAirs<F>` in powdr-openvm.
#[derive(Clone, Serialize, Deserialize)]
pub struct WomirOriginalAirs<F> {
    /// The degree bound used when building the airs
    degree_bound: DegreeBound,
    /// Maps a VM opcode to the name of the (unique) AIR that implements it.
    opcode_to_air: HashMap<VmOpcode, String>,
    /// Maps an AIR name to its symbolic machine and metrics.
    /// Note that this map only contains AIRs that implement instructions.
    air_name_to_machine: BTreeMap<String, (SymbolicMachine<F>, AirMetrics)>,
}

impl<F> InstructionHandler for WomirOriginalAirs<F> {
    type Field = F;
    type Instruction = Instr<F>;
    type AirId = String;

    fn get_instruction_air_and_id(
        &self,
        instruction: &Self::Instruction,
    ) -> (Self::AirId, &SymbolicMachine<Self::Field>) {
        let id = self
            .opcode_to_air
            .get(&instruction.0.opcode)
            .unwrap()
            .clone();
        let air = &self.air_name_to_machine.get(&id).unwrap().0;
        (id, air)
    }

    fn is_allowed(&self, instruction: &Self::Instruction) -> bool {
        self.opcode_to_air.contains_key(&instruction.0.opcode)
    }

    fn is_branching(&self, instruction: &Self::Instruction) -> bool {
        branch_opcodes_set().contains(&instruction.0.opcode)
    }

    fn get_instruction_air_stats(&self, instruction: &Self::Instruction) -> AirStats {
        self.get_instruction_metrics(instruction.0.opcode)
            .map(|metrics| metrics.clone().into())
            .unwrap()
    }

    fn degree_bound(&self) -> DegreeBound {
        self.degree_bound
    }
}

impl<F> WomirOriginalAirs<F> {
    /// Build the instruction handler from a WomirConfig.
    ///
    /// This involves building the WOMIR chip complex, extracting symbolic AIRs
    /// for each instruction chip, and mapping opcodes to those AIRs.
    ///
    /// Follows the pattern from `OriginalVmConfig::airs()` in
    /// `powdr-openvm/src/extraction_utils.rs`.
    pub fn from_womir_config(_degree_bound: DegreeBound) -> Self {
        // TODO: Build the WOMIR chip complex (WomirConfig), extract AirInventory,
        // iterate over allowed opcodes, and build opcode_to_air + air_name_to_machine maps.
        //
        // High-level steps:
        // 1. Create WomirConfig::default() and build its AirInventory
        // 2. Create executor inventory to get opcode -> executor mappings
        // 3. For each opcode in instruction_allowlist():
        //    a. Find the AIR that handles it
        //    b. Extract symbolic constraints and bus interactions
        //    c. Convert to powdr SymbolicMachine format
        // 4. Populate opcode_to_air and air_name_to_machine
        todo!("Build WomirOriginalAirs from WomirConfig")
    }

    pub fn get_instruction_metrics(&self, opcode: VmOpcode) -> Option<&AirMetrics> {
        self.opcode_to_air.get(&opcode).and_then(|air_name| {
            self.air_name_to_machine
                .get(air_name)
                .map(|(_, metrics)| metrics)
        })
    }
}
