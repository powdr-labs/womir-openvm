//! Snapshot test for WOMIR machine constraints.
//!
//! This test extracts the symbolic constraints from all AIRs in the WOMIR VM
//! and compares them against a snapshot file. This helps catch unintended
//! changes to the constraint system.

use std::{fs, io, path::Path, sync::Arc};

use itertools::Itertools;
use openvm_circuit::arch::{VmBuilder, VmCircuitConfig};
use openvm_circuit::system::memory::interface::MemoryInterfaceAirs;
use openvm_circuit_primitives::bitwise_op_lookup::SharedBitwiseOperationLookupChip;
use openvm_stark_backend::rap::AnyRap;
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Engine;
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use powdr_autoprecompiles::bus_map::{BusMap, BusType};
use powdr_autoprecompiles::expression::try_convert;
use powdr_autoprecompiles::symbolic_machine::SymbolicMachine;
use powdr_openvm::{
    bus_map::OpenVmBusType,
    extraction_utils::{get_columns, get_constraints},
    utils::{
        UnsupportedOpenVmReferenceError, openvm_bus_interaction_to_powdr, symbolic_to_algebraic,
    },
};
use pretty_assertions::assert_eq;
use womir_circuit::{WomirConfig, WomirCpuBuilder};

type BabyBearSC = <BabyBearPoseidon2Engine as openvm_stark_backend::engine::StarkEngine>::SC;

/// Convert an OpenVM AIR to a powdr SymbolicMachine.
fn air_to_symbolic_machine(
    air: Arc<dyn AnyRap<BabyBearSC>>,
) -> Result<SymbolicMachine<BabyBear>, UnsupportedOpenVmReferenceError> {
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

    Ok(SymbolicMachine {
        constraints: powdr_exprs.into_iter().map(Into::into).collect(),
        bus_interactions: powdr_bus_interactions,
        derived_columns: vec![],
    })
}

#[test]
fn extract_machine() {
    let config = WomirConfig::default();
    let air_inventory = VmCircuitConfig::<BabyBearSC>::create_airs(&config)
        .expect("Failed to create AIR inventory");

    // Create chip complex to extract bus map dynamically
    let chip_complex =
        <WomirCpuBuilder as VmBuilder<BabyBearPoseidon2Engine>>::create_chip_complex(
            &WomirCpuBuilder,
            &config,
            air_inventory,
        )
        .expect("Failed to create chip complex");
    let inventory = &chip_complex.inventory;

    // Extract bus map dynamically from chip complex (like OriginalVmConfig::bus_map())
    let shared_bitwise_lookup = inventory
        .find_chip::<SharedBitwiseOperationLookupChip<8>>()
        .next();

    let system_air_inventory = inventory.airs().system();
    let connector_air = system_air_inventory.connector;
    let memory_air = &system_air_inventory.memory;

    let bus_map: BusMap<OpenVmBusType> = BusMap::from_id_type_pairs(
        [
            (
                connector_air.execution_bus.index(),
                BusType::ExecutionBridge,
            ),
            (
                match &memory_air.interface {
                    MemoryInterfaceAirs::Volatile { boundary } => boundary.memory_bus.inner.index,
                    MemoryInterfaceAirs::Persistent { boundary, .. } => {
                        boundary.memory_bus.inner.index
                    }
                },
                BusType::Memory,
            ),
            (connector_air.program_bus.index(), BusType::PcLookup),
            (
                connector_air.range_bus.index(),
                BusType::Other(OpenVmBusType::VariableRangeChecker),
            ),
        ]
        .into_iter()
        .chain(shared_bitwise_lookup.into_iter().map(|chip| {
            (
                chip.bus().inner.index,
                BusType::Other(OpenVmBusType::BitwiseLookup),
            )
        }))
        .map(|(id, bus_type)| (id as u64, bus_type)),
    );

    // Get all extension AIRs from the chip complex
    let ext_airs = inventory.airs().ext_airs();

    // Debug: print AIR info
    for air in ext_airs.iter() {
        let name = air.name();
        let has_preprocessed = air.preprocessed_trace().is_some();
        let width = air.width();
        eprintln!("AIR: {name}, width: {width}, has_preprocessed: {has_preprocessed}");
    }

    // Only process instruction AIRs (VmAirWrapper), skip peripherals like Poseidon2
    let instruction_airs: Vec<_> = ext_airs
        .iter()
        .filter(|air| {
            let name = air.name();
            // Skip peripheral AIRs that don't implement instructions
            air.preprocessed_trace().is_none() && name.starts_with("VmAirWrapper")
        })
        .collect();

    eprintln!("Processing {} instruction AIRs", instruction_airs.len());

    let rendered = instruction_airs
        .iter()
        .filter_map(|air| {
            let name = air.name();
            eprintln!("Converting: {name}");
            match air_to_symbolic_machine((*air).clone()) {
                Ok(machine) => {
                    eprintln!("  Success!");
                    Some(format!("# {name}\n{}", machine.render(&bus_map)))
                }
                Err(e) => {
                    eprintln!("  Skipping: {e:?}");
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
