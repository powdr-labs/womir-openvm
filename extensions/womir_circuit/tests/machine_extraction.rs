//! Snapshot test for WOMIR machine constraints.
//!
//! This test extracts the symbolic constraints from all AIRs in the WOMIR VM
//! and compares them against a snapshot file. This helps catch unintended
//! changes to the constraint system.

use std::{collections::BTreeMap, fs, io, path::Path, sync::Arc};

use itertools::Itertools;
use openvm_stark_backend::{
    air_builders::symbolic::{
        get_symbolic_builder, symbolic_expression::SymbolicExpression, SymbolicConstraints,
    },
    config::{Com, StarkGenericConfig},
    interaction::RapPhaseSeqKind,
    keygen::types::{ProverOnlySinglePreprocessedData, TraceWidth, VerifierSinglePreprocessedData},
    p3_commit::Pcs,
    p3_field::PrimeField32,
    p3_matrix::Matrix,
    rap::AnyRap,
};
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::{BabyBearPoseidon2Config, config_from_perm, default_perm},
    config::fri_params::SecurityParameters,
    p3_baby_bear::BabyBear,
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

/// Extract symbolic constraints from an AIR.
fn get_constraints(
    pcs: &<SC as StarkGenericConfig>::Pcs,
    air: Arc<dyn AnyRap<SC>>,
) -> SymbolicConstraints<BabyBear> {
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
    builder.constraints()
}

/// Format a field element in a human-readable way.
fn format_field_element(v: BabyBear) -> String {
    let v = v.as_canonical_u32();
    if v < BabyBear::ORDER_U32 / 2 {
        format!("{v}")
    } else {
        format!("-{}", BabyBear::ORDER_U32 - v)
    }
}

/// Format a symbolic expression with column names.
fn format_expr(expr: &SymbolicExpression<BabyBear>, columns: &[String]) -> String {
    match expr {
        SymbolicExpression::Constant(c) => format_field_element(*c),
        SymbolicExpression::Variable(var) => {
            let col_idx = var.index;
            let col_name = columns.get(col_idx).map(|s| s.as_str()).unwrap_or("?");
            let offset = var.entry.offset().unwrap_or(0);
            if offset == 0 {
                col_name.to_string()
            } else {
                format!("{col_name}'")
            }
        }
        SymbolicExpression::Add { x, y, .. } => {
            format!("({} + {})", format_expr(x, columns), format_expr(y, columns))
        }
        SymbolicExpression::Sub { x, y, .. } => {
            format!("({} - {})", format_expr(x, columns), format_expr(y, columns))
        }
        SymbolicExpression::Neg { x, .. } => {
            format!("(-{})", format_expr(x, columns))
        }
        SymbolicExpression::Mul { x, y, .. } => {
            format!("{} * {}", format_expr(x, columns), format_expr(y, columns))
        }
        SymbolicExpression::IsFirstRow => "is_first_row".to_string(),
        SymbolicExpression::IsLastRow => "is_last_row".to_string(),
        SymbolicExpression::IsTransition => "is_transition".to_string(),
    }
}

/// Render constraints and interactions in a human-readable format.
fn render_constraints(
    constraints: &SymbolicConstraints<BabyBear>,
    columns: &[String],
) -> String {
    let mut output = String::new();

    // List columns
    output.push_str(&format!(
        "Symbolic machine using {} unique main columns:\n",
        columns.len()
    ));
    for col in columns {
        output.push_str(&format!("  {col}\n"));
    }
    output.push('\n');

    // Group interactions by bus index
    let mut interactions_by_bus: BTreeMap<u16, Vec<_>> = BTreeMap::new();
    for interaction in &constraints.interactions {
        interactions_by_bus
            .entry(interaction.bus_index)
            .or_default()
            .push(interaction);
    }

    for (bus_idx, interactions) in interactions_by_bus {
        output.push_str(&format!("// Bus {bus_idx}:\n"));
        for interaction in interactions {
            let count_str = format_expr(&interaction.count, columns);
            let args_str = interaction
                .message
                .iter()
                .map(|e| format_expr(e, columns))
                .join(", ");
            output.push_str(&format!("mult={count_str}, args=[{args_str}]\n"));
        }
        output.push('\n');
    }

    // Algebraic constraints
    if !constraints.constraints.is_empty() {
        output.push_str("// Algebraic constraints:\n");
        for constraint in &constraints.constraints {
            let constraint_str = format_expr(constraint, columns);
            output.push_str(&format!("{constraint_str} = 0\n"));
        }
    }

    output
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

    let rendered = ext_airs
        .iter()
        .map(|air| {
            let name = air.name();
            let width = air.width();

            // Get column names, falling back to generated names
            let columns: Vec<String> = air
                .columns()
                .unwrap_or_else(|| (0..width).map(|i| format!("col_{i}")).collect());

            let constraints = get_constraints(pcs, air.clone());
            let constraints_rendered = render_constraints(&constraints, &columns);

            format!("# {name}\n{constraints_rendered}")
        })
        .join("\n\n");

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
