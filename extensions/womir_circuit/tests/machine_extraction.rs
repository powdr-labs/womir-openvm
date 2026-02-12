//! Snapshot test for WOMIR machine constraints.
//!
//! This test extracts the symbolic constraints from all AIRs in the WOMIR VM
//! and compares them against a snapshot file. This helps catch unintended
//! changes to the constraint system.

use std::{fs, io, path::Path, sync::Arc};

use itertools::Itertools;
use openvm_circuit::arch::VmCircuitConfig;
use openvm_stark_backend::rap::AnyRap;
use powdr_autoprecompiles::bus_map::BusMap;
use powdr_autoprecompiles::expression::try_convert;
use powdr_autoprecompiles::symbolic_machine::SymbolicMachine;
use powdr_openvm::customize_exe::openvm_bus_interaction_to_powdr;
use powdr_openvm::{
    BabyBearSC,
    bus_map::OpenVmBusType,
    extraction_utils::{get_columns, get_constraints},
    utils::symbolic_to_algebraic,
};
use pretty_assertions::assert_eq;
use womir_circuit::WomirConfig;

/// Convert an OpenVM AIR to a powdr SymbolicMachine.
fn air_to_symbolic_machine(
    air: Arc<dyn AnyRap<BabyBearSC>>,
) -> Result<
    SymbolicMachine<powdr_number::BabyBearField>,
    powdr_openvm::utils::UnsupportedOpenVmReferenceError,
> {
    let columns = get_columns(air.clone());
    let constraints = get_constraints(air);

    let powdr_exprs = constraints
        .constraints
        .iter()
        .map(|expr| try_convert(symbolic_to_algebraic(expr, &columns)))
        .collect::<Result<Vec<_>, _>>()?;

    let powdr_bus_interactions = constraints
        .interactions
        .iter()
        .map(|interaction| openvm_bus_interaction_to_powdr(interaction, &columns))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(SymbolicMachine::new(powdr_exprs, powdr_bus_interactions))
}

/// Create a bus map for WOMIR.
/// This maps bus indices to human-readable bus type names.
fn womir_bus_map() -> BusMap<OpenVmBusType> {
    use powdr_autoprecompiles::bus_map::BusType;

    // These bus indices are based on the OpenVM system configuration.
    // The exact values depend on how the system is configured.
    BusMap::from_id_type_pairs(
        [
            (0, BusType::ExecutionBridge),
            (1, BusType::Memory),
            (2, BusType::PcLookup),
            (3, BusType::Other(OpenVmBusType::VariableRangeChecker)),
            (6, BusType::Other(OpenVmBusType::BitwiseLookup)),
        ]
        .into_iter(),
    )
}

#[test]
fn extract_machine() {
    let config = WomirConfig::default();
    let air_inventory = config
        .create_airs()
        .expect("Failed to create AIR inventory");

    let bus_map = womir_bus_map();

    // Get all extension AIRs (these are the WOMIR-specific ones)
    let ext_airs = air_inventory.ext_airs();

    let rendered = ext_airs
        .iter()
        .filter_map(|air| {
            let name = air.name();
            match air_to_symbolic_machine(air.clone()) {
                Ok(machine) => Some(format!("# {name}\n{}", machine.render(&bus_map))),
                Err(e) => {
                    // Skip AIRs that can't be converted (e.g., peripherals without instructions)
                    eprintln!("Skipping {name}: {e}");
                    None
                }
            }
        })
        .join("\n\n\n");

    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("womir_constraints.txt");

    match fs::read_to_string(&path) {
        // Snapshot exists, compare it with the extracted constraints
        Ok(expected) => {
            assert_eq!(rendered, expected)
        }

        // Snapshot does not exist, create it
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).unwrap();
            }
            fs::write(&path, &rendered).unwrap();
            panic!("Created new snapshot at {path:?}. Inspect it, then rerun the tests.");
        }

        Err(err) => panic!("Failed to read snapshot file: {err}"),
    }
}
