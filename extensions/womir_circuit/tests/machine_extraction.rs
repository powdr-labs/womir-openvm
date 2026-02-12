//! Snapshot test for WOMIR machine constraints.
//!
//! This test extracts key metrics from all AIRs in the WOMIR VM
//! and compares them against a snapshot file. This helps catch unintended
//! changes to the constraint system.

use std::{fs, io, path::Path, sync::Arc};

use openvm_stark_backend::{
    air_builders::symbolic::get_symbolic_builder,
    config::{Com, StarkGenericConfig},
    interaction::RapPhaseSeqKind,
    keygen::types::{ProverOnlySinglePreprocessedData, TraceWidth, VerifierSinglePreprocessedData},
    p3_commit::Pcs,
    p3_matrix::Matrix,
    rap::AnyRap,
};
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::{BabyBearPoseidon2Config, config_from_perm, default_perm},
    config::fri_params::SecurityParameters,
};
use womir_circuit::WomirConfig;

type SC = BabyBearPoseidon2Config;

/// Preprocessed keygen data for an AIR.
struct PrepKeygenData<SC: StarkGenericConfig> {
    _verifier_data: Option<VerifierSinglePreprocessedData<Com<SC>>>,
    prover_data: Option<ProverOnlySinglePreprocessedData<SC>>,
}

impl<SC: StarkGenericConfig> PrepKeygenData<SC> {
    fn width(&self) -> Option<usize> {
        self.prover_data.as_ref().map(|d| d.trace.width())
    }
}

fn compute_prep_data_for_air<SC: StarkGenericConfig>(
    pcs: &SC::Pcs,
    air: &dyn AnyRap<SC>,
) -> PrepKeygenData<SC> {
    let preprocessed_trace = air.preprocessed_trace();
    let vpdata_opt = preprocessed_trace.map(|trace| {
        let domain = pcs.natural_domain_for_degree(trace.height());
        let (commit, data) = pcs.commit(vec![(domain, trace.clone())]);
        let vdata = VerifierSinglePreprocessedData { commit };
        let pdata = ProverOnlySinglePreprocessedData {
            trace: Arc::new(trace),
            data: Arc::new(data),
        };
        (vdata, pdata)
    });
    if let Some((vdata, pdata)) = vpdata_opt {
        PrepKeygenData {
            prover_data: Some(pdata),
            _verifier_data: Some(vdata),
        }
    } else {
        PrepKeygenData {
            prover_data: None,
            _verifier_data: None,
        }
    }
}

/// Extract constraint and interaction counts from an AIR.
fn get_constraint_counts(
    pcs: &<SC as StarkGenericConfig>::Pcs,
    air: Arc<dyn AnyRap<SC>>,
) -> (usize, usize) {
    let prep_keygen_data = compute_prep_data_for_air(pcs, air.as_ref());
    let width = TraceWidth {
        preprocessed: prep_keygen_data.width(),
        cached_mains: air.cached_main_widths(),
        common_main: air.common_main_width(),
        after_challenge: vec![],
    };
    let builder = get_symbolic_builder(
        air.as_ref(),
        &width,
        &[],
        &[],
        RapPhaseSeqKind::None,
        0, // max_constraint_degree: 0 means no limit
    );
    let constraints = builder.constraints();
    (
        constraints.constraints.len(),
        constraints.interactions.len(),
    )
}

#[test]
fn extract_machine() {
    use openvm_circuit::arch::VmCircuitConfig;

    let config = WomirConfig::default();
    let air_inventory = config
        .create_airs()
        .expect("Failed to create AIR inventory");

    // Set up the PCS for constraint extraction
    let perm = default_perm();
    let security_params = SecurityParameters::standard_fast();
    let stark_config = config_from_perm(&perm, security_params);
    let pcs = stark_config.pcs();

    // Get all extension AIRs (these are the WOMIR-specific ones)
    let ext_airs = air_inventory.ext_airs();

    let mut rendered_parts = Vec::new();
    for air in ext_airs.iter() {
        let name = air.name();
        let width = air.width();
        let prep_data = compute_prep_data_for_air(pcs, air.as_ref());
        let preprocessed_width = prep_data.width().unwrap_or(0);
        let (num_constraints, num_interactions) = get_constraint_counts(pcs, air.clone());

        rendered_parts.push(format!(
            "# {name}\n\
             Width: {width}\n\
             Preprocessed Width: {preprocessed_width}\n\
             Constraints: {num_constraints}\n\
             Interactions: {num_interactions}"
        ));
    }

    let rendered = rendered_parts.join("\n\n");

    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("womir_constraints.txt");

    match fs::read_to_string(&path) {
        // Snapshot exists, compare it with the extracted constraints
        Ok(expected) => {
            assert_eq!(rendered.trim(), expected.trim())
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
