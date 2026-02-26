use std::collections::BTreeMap;
use std::sync::Arc;

use itertools::Itertools;
use openvm_stark_backend::p3_matrix::Matrix;
use openvm_stark_backend::rap::PartitionedBaseAir;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, BaseAir},
    p3_field::PrimeField32,
    rap::{BaseAirWithPublicValues, ColumnsAir},
};
use powdr_autoprecompiles::expression::AlgebraicEvaluator;
use powdr_autoprecompiles::expression::WitnessEvaluator;
use powdr_autoprecompiles::{Apc, expression::AlgebraicReference};

use crate::autoprecompiles::adapter::Instr;

// TODO: This type and implementations are identical to PowdrAir in powdr-openvm, except for different type parameters in
//       the APC.
pub struct WomirAir<F> {
    /// The columns in arbitrary order
    columns: Vec<AlgebraicReference>,
    apc: Arc<Apc<F, Instr<F>, (), ()>>,
}

impl<F: PrimeField32> ColumnsAir<F> for WomirAir<F> {
    fn columns(&self) -> Option<Vec<String>> {
        Some(self.columns.iter().map(|c| (*c.name).clone()).collect())
    }
}

impl<F: PrimeField32> WomirAir<F> {
    pub fn new(apc: Arc<Apc<F, Instr<F>, (), ()>>) -> Self {
        Self {
            columns: apc.machine().main_columns().collect(),
            apc,
        }
    }
}

impl<F: PrimeField32> BaseAir<F> for WomirAir<F> {
    fn width(&self) -> usize {
        let res = self.columns.len();
        assert!(res > 0);
        res
    }
}

// No public values, but the trait is implemented
impl<F: PrimeField32> BaseAirWithPublicValues<F> for WomirAir<F> {}

impl<AB: InteractionBuilder> Air<AB> for WomirAir<AB::F>
where
    AB::F: PrimeField32,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let witnesses = main.row_slice(0);
        // TODO: cache?
        let witness_values: BTreeMap<u64, AB::Var> = self
            .columns
            .iter()
            .map(|c| c.id)
            .zip_eq(witnesses.iter().cloned())
            .collect();

        let witness_evaluator = WitnessEvaluator::new(&witness_values);

        for constraint in &self.apc.machine().constraints {
            let constraint = witness_evaluator.eval_constraint(constraint);
            builder.assert_zero(constraint.expr);
        }

        for interaction in &self.apc.machine().bus_interactions {
            let interaction = witness_evaluator.eval_bus_interaction(interaction);
            // TODO: is this correct?
            let count_weight = 1;

            builder.push_interaction(
                interaction.id as u16,
                interaction.args,
                interaction.mult,
                count_weight,
            );
        }
    }
}

impl<F: PrimeField32> PartitionedBaseAir<F> for WomirAir<F> {}
