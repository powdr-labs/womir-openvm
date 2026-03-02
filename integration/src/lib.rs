mod builtin_functions;
mod const_collapse;
mod instruction_builder;
pub mod proving;
#[cfg(test)]
mod tests;
pub mod womir_translation;

use std::sync::atomic::AtomicU32;
use std::sync::mpsc::channel;
use std::sync::{Mutex, RwLock};

use itertools::Itertools;
use womir::loader::rwm::RWMStages;
use womir::loader::{
    CommonStages, FunctionAsm, FunctionProcessingStage, Module, PartiallyParsedProgram, Statistics,
};

pub use openvm_sdk::StdIn;
pub use womir_circuit::WomirConfig;
pub use womir_translation::{Directive, LinkedProgram, OpenVMSettings};

pub type F = openvm_stark_sdk::p3_baby_bear::BabyBear;

use crate::builtin_functions::BuiltinFunction;

/// Load and compile a WASM binary. Returns the module and compiled functions.
pub fn load_wasm(wasm_bytes: &[u8]) -> (Module<'_>, Vec<FunctionAsm<Directive<F>>>) {
    load_wasm_with_settings(wasm_bytes, OpenVMSettings::<F>::new())
}

/// Load and compile a WASM binary with custom settings. Returns the module and compiled functions.
pub fn load_wasm_with_settings(
    wasm_bytes: &[u8],
    settings: OpenVMSettings<F>,
) -> (Module<'_>, Vec<FunctionAsm<Directive<F>>>) {
    let PartiallyParsedProgram { s: _, m, functions } =
        womir::loader::load_wasm(settings, wasm_bytes).unwrap();

    let num_functions = functions.len() as u32;
    let tracker = RwLock::new(Some(builtin_functions::Tracker::new(num_functions)));
    let global_stats = Mutex::new(Statistics::default());
    let module = RwLock::new(m);
    let label_gen = AtomicU32::new(0);
    let (jobs_s, jobs_r) = channel();
    let jobs_r = Mutex::new(jobs_r);

    let functions = std::thread::scope(|scope| {
        type FunctionInProcessing<'a> = RWMStages<'a, OpenVMSettings<F>>;
        enum Job<'a> {
            PatchFunc(u32, FunctionInProcessing<'a>),
            FinishFunc(u32, FunctionInProcessing<'a>),
            LoadBuiltin(BuiltinFunction),
        }

        let (patched_s, patched_r) = channel();
        let (final_s, final_r) = channel();

        let num_threads = num_cpus::get().max(1);
        for _ in 0..num_threads {
            let tracker = &tracker;
            let global_stats = &global_stats;
            let label_gen = &label_gen;
            let module = &module;
            let jobs_r = &jobs_r;
            let patched_s = patched_s.clone();
            let final_s = final_s.clone();
            scope.spawn(move || {
                let mut stats = Statistics::default();
                while let Ok(job) = {
                    // This extra scope is needed to release the lock before processing the function.
                    jobs_r.lock().unwrap().recv()
                } {
                    match job {
                        Job::PatchFunc(func_idx, mut func) => {
                            // Advance to BlocklessDag stage
                            let module = module.read().unwrap();
                            let dag = loop {
                                if let RWMStages::CommonStages(CommonStages::BlocklessDag(dag)) =
                                    &mut func
                                {
                                    break dag;
                                }
                                func = func
                                    .advance_stage(
                                        &settings,
                                        &module,
                                        func_idx,
                                        label_gen,
                                        Some(&mut stats),
                                    )
                                    .unwrap();
                            };

                            // In the BlocklessDag stage, we need to find instructions we don't
                            // implement and replace them with function calls to built-in functions.
                            tracker
                                .read()
                                .unwrap()
                                .as_ref()
                                .unwrap()
                                .replace_with_builtins(dag);

                            patched_s.send((func_idx, func)).unwrap();
                        }
                        Job::FinishFunc(func_idx, func) => {
                            let module = module.read().unwrap();
                            let func = func
                                .advance_all_stages(
                                    &settings,
                                    &module,
                                    func_idx,
                                    label_gen,
                                    Some(&mut stats),
                                )
                                .unwrap();
                            final_s.send((func_idx, func)).unwrap();
                        }
                        Job::LoadBuiltin(builtin) => {
                            let func_idx = builtin.function_index;
                            let func = builtin.load(&module.read().unwrap(), label_gen);
                            final_s.send((func_idx, func)).unwrap();
                        }
                    }
                }
                *global_stats.lock().unwrap() += stats;
            });
        }
        // Drop the channel endpoints this thread will no longer use.
        drop(final_s);
        drop(patched_s);

        // Send the functions for processing.
        for (idx, func) in functions.into_iter().enumerate() {
            let rwm_func: FunctionInProcessing<'_> = func.into();
            jobs_s.send(Job::PatchFunc(idx as u32, rwm_func)).unwrap();
        }

        // Received the patched functions, update the module, and resend for processing.
        let patched_functions = patched_r
            .into_iter()
            .take(num_functions as usize)
            .collect_vec();
        let tracker = tracker.write().unwrap().take().unwrap();
        let used_builtins = tracker.into_used_builtins().collect_vec();
        {
            let mut module = module.write().unwrap();
            for bf in &used_builtins {
                let given_idx = module.append_function(bf.definition.params, bf.definition.results);
                assert_eq!(given_idx, bf.function_index);
            }
        }

        // Now that module was updated, send the functions back to resume processing.
        for (idx, func) in patched_functions {
            jobs_s.send(Job::FinishFunc(idx, func)).unwrap();
        }

        // And also send the built-in functions to be loaded
        for bf in used_builtins {
            jobs_s.send(Job::LoadBuiltin(bf)).unwrap();
        }

        // Signal threads there are no more jobs.
        drop(jobs_s);

        // Receive the fully processed functions, which included the loaded builtins.
        final_r
            .into_iter()
            .sorted_unstable_by_key(|f| f.0)
            .map(|f| f.1)
            .collect_vec()
    });

    println!(
        "WOMIR loading statistics: {}",
        global_stats.into_inner().unwrap()
    );

    (module.into_inner().unwrap(), functions)
}

pub fn setup_tracing_with_log_level(level: tracing::Level) {
    use metrics_tracing_context::MetricsLayer;
    use tracing_forest::ForestLayer;
    use tracing_subscriber::util::SubscriberInitExt;
    use tracing_subscriber::{EnvFilter, Registry, layer::SubscriberExt};

    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(format!("{level},p3_=warn")));
    let _ = Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .with(MetricsLayer::new())
        .try_init();
}

/// Convenience: load wasm, execute the named function, return output bytes.
pub fn run_wasm(wasm_bytes: &[u8], function_name: &str, stdin: StdIn<F>) -> eyre::Result<Vec<u8>> {
    let (module, functions) = load_wasm(wasm_bytes);
    let mut program = LinkedProgram::new(module, functions);
    Ok(program.execute(WomirConfig::default(), function_name, stdin)?)
}

#[cfg(test)]
mod lib_tests {
    use super::*;
    use std::path::PathBuf;

    fn sample_program(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("sample-programs")
            .join(name)
    }

    #[test]
    fn test_run_wasm_fib() {
        let wasm_bytes = std::fs::read(sample_program("fib_loop.wasm")).unwrap();
        let mut stdin = StdIn::default();
        stdin.write(&10u32);
        let output = run_wasm(&wasm_bytes, "fib", stdin).unwrap();
        let result = u32::from_le_bytes(output[..4].try_into().unwrap());
        assert_eq!(result, 55);
    }

    #[test]
    fn test_run_wasm_n_first_sum() {
        let wasm_bytes = std::fs::read(sample_program("n_first_sum.wasm")).unwrap();
        let mut stdin = StdIn::default();
        stdin.write(&42u32);
        stdin.write(&0u32);
        let output = run_wasm(&wasm_bytes, "n_first_sum", stdin).unwrap();
        let results: Vec<u32> = output[..8]
            .chunks(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(results, vec![903, 0]);
    }

    #[test]
    fn test_load_wasm_produces_functions() {
        let wasm_bytes = std::fs::read(sample_program("fib_loop.wasm")).unwrap();
        let (_module, functions) = load_wasm(&wasm_bytes);
        assert!(!functions.is_empty());
    }
}
