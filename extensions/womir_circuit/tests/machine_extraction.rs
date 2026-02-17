//! Snapshot test for WOMIR machine constraints.
//!
//! This test extracts the symbolic constraints from all AIRs in the WOMIR VM
//! and compares them against a snapshot file. This helps catch unintended
//! changes to the constraint system.

use std::{fs, io, path::Path, sync::Arc};

use itertools::Itertools;
use openvm_circuit::arch::{MatrixRecordArena, VmBuilder, VmChipComplex, VmCircuitConfig};
use openvm_circuit::system::SystemChipInventory;
use openvm_circuit::system::memory::interface::MemoryInterfaceAirs;
use openvm_circuit_primitives::bitwise_op_lookup::SharedBitwiseOperationLookupChip;
use openvm_circuit_primitives::range_tuple::SharedRangeTupleCheckerChip;
use openvm_stark_backend::config::Val;
use openvm_stark_backend::prover::cpu::CpuBackend;
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
use womir_circuit::{WomirConfig, WomirCpuBuilder};

type BabyBearSC = <BabyBearPoseidon2Engine as openvm_stark_backend::engine::StarkEngine>::SC;
type WomirChipComplex = VmChipComplex<
    BabyBearSC,
    MatrixRecordArena<Val<BabyBearSC>>,
    CpuBackend<BabyBearSC>,
    SystemChipInventory<BabyBearSC>,
>;

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

/// Create a chip complex for the WOMIR VM.
fn create_chip_complex(config: &WomirConfig) -> WomirChipComplex {
    let air_inventory =
        VmCircuitConfig::<BabyBearSC>::create_airs(config).expect("Failed to create AIR inventory");

    <WomirCpuBuilder as VmBuilder<BabyBearPoseidon2Engine>>::create_chip_complex(
        &WomirCpuBuilder,
        config,
        air_inventory,
    )
    .expect("Failed to create chip complex")
}

/// Extract bus map dynamically from chip complex (like OriginalVmConfig::bus_map()).
fn extract_bus_map(chip_complex: &WomirChipComplex) -> BusMap<OpenVmBusType> {
    let inventory = &chip_complex.inventory;

    let shared_bitwise_lookup = inventory
        .find_chip::<SharedBitwiseOperationLookupChip<8>>()
        .next();
    let shared_range_tuple_checker = inventory
        .find_chip::<SharedRangeTupleCheckerChip<2>>()
        .next();

    let system_air_inventory = inventory.airs().system();
    let connector_air = system_air_inventory.connector;
    let memory_air = &system_air_inventory.memory;

    BusMap::from_id_type_pairs(
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
        .chain(shared_range_tuple_checker.into_iter().map(|chip| {
            (
                chip.bus().inner.index,
                BusType::Other(OpenVmBusType::TupleRangeChecker),
            )
        }))
        .map(|(id, bus_type)| (id as u64, bus_type)),
    )
}

/// Compare rendered output against snapshot file, creating it if it doesn't exist.
fn assert_snapshot(rendered: &str, snapshot_name: &str) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join(snapshot_name);

    match fs::read_to_string(&path) {
        Ok(expected) => {
            assert!(
                rendered == expected,
                "Snapshots differ. If you want to updated the snapshot, delete the file at {path:?} and rerun the tests."
            );
        }
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).unwrap();
            }
            fs::write(&path, rendered).unwrap();
            panic!("Created new snapshot at {path:?}. Inspect it, then rerun the tests.");
        }
        Err(err) => panic!("Failed to read snapshot file: {err}"),
    }
}

#[test]
fn extract_machine() {
    let config = WomirConfig::default();
    let chip_complex = create_chip_complex(&config);
    let bus_map = extract_bus_map(&chip_complex);

    let ext_airs = chip_complex.inventory.airs().ext_airs();
    let instruction_airs: Vec<_> = ext_airs
        .iter()
        // Skip large AIRs (e.g. large precompiles) and lookup tables
        .filter(|air| air.preprocessed_trace().is_none() && air.width() < 200)
        .collect();

    let rendered = instruction_airs
        .iter()
        .filter_map(|air| {
            let name = air.name();
            air_to_symbolic_machine((*air).clone())
                .ok()
                .map(|machine| format!("# {name}\n{}", machine.render(&bus_map)))
        })
        .join("\n\n\n");

    assert_snapshot(&rendered, "womir_constraints.txt");
}
