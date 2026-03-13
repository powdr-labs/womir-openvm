//! Snapshot test for CRUSH machine constraints.
//!
//! This test extracts the symbolic constraints from all AIRs in the CRUSH VM
//! and compares them against a snapshot file. This helps catch unintended
//! changes to the constraint system.

use std::{fs, io, path::Path};

use autoprecompiles::CrushISA;
use crush_circuit::CrushConfig;
use itertools::Itertools;
use powdr_openvm::{DEFAULT_DEGREE_BOUND, extraction_utils::OriginalVmConfig};

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
    let original_config = OriginalVmConfig::<CrushISA>::new(CrushConfig::default());
    let airs = original_config
        .airs(DEFAULT_DEGREE_BOUND)
        .expect("failed to extract instruction AIRs");
    let bus_map = original_config.bus_map();

    let rendered = airs
        .airs_by_name()
        .map(|(name, machine)| format!("# {name}\n{}", machine.render(&bus_map)))
        .join("\n\n\n");

    assert_snapshot(&rendered, "crush_constraints.txt");
}
