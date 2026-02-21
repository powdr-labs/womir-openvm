mod builtin_functions;
mod const_collapse;
mod instruction_builder;
#[cfg(test)]
mod isolated_tests;
#[cfg(test)]
mod riscv_tests;
#[cfg(test)]
mod test_stages;
#[cfg(test)]
mod vm_tests;
#[cfg(test)]
mod wasm_tests;
mod womir_translation;

use std::fs;

use clap::{Parser, Subcommand};
use derive_more::From;
use eyre::Result;
use itertools::Itertools;
use metrics_tracing_context::{MetricsLayer, TracingContextLayer};
use metrics_util::{debugging::DebuggingRecorder, layers::Layer};
use openvm_sdk::StdIn;
use openvm_stark_sdk::bench::serialize_metric_snapshot;
use std::path::PathBuf;
use std::sync::atomic::AtomicU32;
use std::sync::mpsc::channel;
use std::sync::{Mutex, RwLock};
use tracing_forest::ForestLayer;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry, layer::SubscriberExt};
use womir::loader::rwm::RWMStages;
use womir::loader::{
    CommonStages, FunctionAsm, FunctionProcessingStage, Module, PartiallyParsedProgram, Statistics,
};

use tracing::Level;
type F = openvm_stark_sdk::p3_baby_bear::BabyBear;

// use openvm_womir_circuit::{self, WomirI, WomirIExecutor, WomirIPeriphery};

use crate::builtin_functions::BuiltinFunction;
use crate::womir_translation::{Directive, LinkedProgram, OpenVMSettings};

use womir_circuit::WomirConfig;

// #[derive(Serialize, Deserialize, Clone)]
// pub struct SpecializedConfig {
//     pub sdk_config: SdkVmConfig,
//     wom: WomirI<F>,
// }
//
// impl SpecializedConfig {
//     fn new(sdk_config: SdkVmConfig) -> Self {
//         Self {
//             sdk_config,
//             wom: WomirI::default(),
//         }
//     }
// }
//
// impl InitFileGenerator for SpecializedConfig {
//     fn generate_init_file_contents(&self) -> Option<String> {
//         self.sdk_config.generate_init_file_contents()
//     }
//
//     fn write_to_init_file(
//         &self,
//         manifest_dir: &Path,
//         init_file_name: Option<&str>,
//     ) -> eyre::Result<()> {
//         self.sdk_config
//             .write_to_init_file(manifest_dir, init_file_name)
//     }
// }
//
// #[allow(clippy::large_enum_variant)]
// #[derive(ChipUsageGetter, InstructionExecutor, Chip, From, AnyEnum)]
// pub enum SpecializedExecutor<F: PrimeField32> {
//     #[any_enum]
//     SdkExecutor(SdkVmConfigExecutor<F>),
//     #[any_enum]
//     WomExecutor(WomirIExecutor<F>),
// }
//
// #[derive(From, ChipUsageGetter, Chip, AnyEnum)]
// pub enum SpecializedPeriphery<F: PrimeField32> {
//     #[any_enum]
//     SdkPeriphery(SdkVmConfigPeriphery<F>),
//     #[any_enum]
//     WomPeriphery(WomirIPeriphery<F>),
// }
//
// impl VmConfig<F> for SpecializedConfig {
//     type Executor = SpecializedExecutor<F>;
//     type Periphery = SpecializedPeriphery<F>;
//
//     fn system(&self) -> &SystemConfig {
//         VmConfig::system(&self.sdk_config)
//     }
//
//     fn system_mut(&mut self) -> &mut SystemConfig {
//         VmConfig::system_mut(&mut self.sdk_config)
//     }
//
//     fn create_chip_complex(
//         &self,
//     ) -> Result<VmChipComplex<F, Self::Executor, Self::Periphery>, VmInventoryError> {
//         let chip = self.sdk_config.create_chip_complex()?;
//         let chip = chip.extend(&self.wom)?;
//
//         Ok(chip)
//     }
// }

#[derive(Parser)]
struct CliArgs {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Clone)]
enum Commands {
    /// Just prints the program WOM listing
    PrintWom {
        /// Path to the WASM program
        program: String,
    },
    /// Runs a function from the program with arguments
    Run {
        /// Path to the WASM program
        program: String,
        /// Function name
        function: String,
        /// Arguments to pass to the function
        #[arg(long)]
        args: Vec<String>,
        /// Files to be read as bytes.
        #[arg(long)]
        binary_input_files: Vec<String>,
    },
    /// Proves execution of a function from the WASM program with the given arguments
    Prove {
        /// Path to the WASM program
        program: String,
        /// Function name
        function: String,
        /// Arguments to pass to the function
        args: Vec<String>,
        /// Path to output metrics JSON file
        #[arg(long)]
        metrics: Option<PathBuf>,
    },
    /// Proves execution of a function from the RISC-V program with the given arguments.
    /// Even though not the main goal of this crate, this is useful for benchmarking against
    /// womir-openvm.
    ProveRiscv {
        /// Path to the Rust crate
        program: String,
        /// Arguments to pass to OpenVM RISC-V StdIn
        args: Vec<String>,
        /// Path to output metrics JSON file
        #[arg(long)]
        metrics: Option<PathBuf>,
    },
}

impl Commands {
    fn get_program_path(&self) -> &str {
        match self {
            Commands::PrintWom { program } => program,
            Commands::Run { program, .. } => program,
            Commands::Prove { program, .. } => program,
            Commands::ProveRiscv { program, .. } => program,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    setup_tracing_with_log_level(Level::INFO);

    // Parse command line arguments
    let cli_args = CliArgs::parse();
    let cmd = cli_args.command.clone();
    let program_path = cmd.get_program_path();

    match cli_args.command {
        Commands::PrintWom { .. } => {
            // Load the module
            let wasm_bytes = std::fs::read(program_path).expect("Failed to read WASM file");
            let (_module, functions) = load_wasm(&wasm_bytes);

            for func in &functions {
                println!("Function {}:", func.func_idx);
                for directive in &func.directives {
                    println!("  {directive:?}");
                }
            }
        }
        Commands::Run {
            function,
            args,
            binary_input_files,
            ..
        } => {
            // Load the module
            let wasm_bytes = std::fs::read(program_path).expect("Failed to read WASM file");
            let (module, functions) = load_wasm(&wasm_bytes);

            // Create and execute program
            let mut linked_program = LinkedProgram::new(module, functions);

            let mut stdin = StdIn::default();
            for arg in args {
                let val = arg.parse::<u32>().unwrap();
                stdin.write(&val);
            }

            for binary_input_file in &binary_input_files {
                stdin.write_bytes(&fs::read(binary_input_file).unwrap());
            }

            let output = linked_program.execute(WomirConfig::default(), &function, stdin)?;

            println!("output: {output:?}");
        }
        Commands::Prove { function, .. } => {
            // Load the module
            let wasm_bytes = std::fs::read(program_path).expect("Failed to read WASM file");
            let (module, functions) = load_wasm(&wasm_bytes);

            // Create program
            let linked_program = LinkedProgram::new(module, functions);
            let _exe = linked_program.program_with_entry_point(&function);

            // let prove = || -> Result<()> {
            //     // Create VM configuration
            //     let vm_config = SdkVmConfig::builder()
            //         .system(Default::default())
            //         .io(Default::default())
            //         .build();
            //     let vm_config = SpecializedConfig::new(vm_config);

            // let sdk = Sdk::new();
            //
            // // Set app configuration
            // let app_fri_params = FriParameters::standard_with_100_bits_conjectured_security(
            //     DEFAULT_APP_LOG_BLOWUP,
            // );
            // let app_config = AppConfig::new(app_fri_params, vm_config.clone());
            //
            // // Commit the exe
            // let app_committed_exe = sdk.commit_app_exe(app_fri_params, exe.clone())?;
            //
            // // Generate an AppProvingKey
            // let app_pk = Arc::new(sdk.app_keygen(app_config)?);
            //
            // // Setup input
            // let mut stdin = StdIn::default();
            // for arg in args {
            //     let val = arg.parse::<u32>().unwrap();
            //     stdin.write(&val);
            // }
            //
            // // Generate a proof
            // tracing::info!("Generating app proof...");
            // let start = std::time::Instant::now();
            // let app_proof = sdk.generate_app_proof(
            //     app_pk.clone(),
            //     app_committed_exe.clone(),
            //     stdin.clone(),
            // )?;
            // tracing::info!("App proof took {:?}", start.elapsed());
            //
            // tracing::info!(
            //     "Public values: {:?}",
            //     app_proof.user_public_values.public_values
            // );

            //     Ok(())
            // };
            // if let Some(metrics_path) = metrics {
            //     run_with_metric_collection_to_file(
            //         std::fs::File::create(metrics_path).expect("Failed to create metrics file"),
            //         prove,
            //     )?;
            // } else {
            //     prove()?
            // }
        }
        Commands::ProveRiscv { metrics, .. } => {
            let prove = || -> Result<()> {
                // let compiled_program = powdr_openvm::compile_guest(
                //     &program,
                //     Default::default(),
                //     powdr_autoprecompiles::PowdrConfig::new(
                //         0,
                //         0,
                //         powdr_openvm::DegreeBound {
                //             identities: 3,
                //             bus_interactions: 2,
                //         },
                //     ),
                //     Default::default(),
                //     Default::default(),
                // )
                // .unwrap();

                // let mut stdin = StdIn::default();
                // for arg in args {
                //     let val = arg.parse::<u32>().unwrap();
                //     stdin.write(&val);
                // }

                // powdr_openvm::prove(&compiled_program, false, false, stdin, None).unwrap();

                Ok(())
            };
            if let Some(metrics_path) = metrics {
                run_with_metric_collection_to_file(
                    std::fs::File::create(metrics_path).expect("Failed to create metrics file"),
                    prove,
                )?;
            } else {
                prove()?
            }
        }
    }

    Ok(())
}

fn load_wasm(wasm_bytes: &[u8]) -> (Module<'_>, Vec<FunctionAsm<Directive<F>>>) {
    let PartiallyParsedProgram { s: _, m, functions } =
        womir::loader::load_wasm(OpenVMSettings::<F>::new(), wasm_bytes).unwrap();

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
                                        &OpenVMSettings::new(),
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
                                    &OpenVMSettings::new(),
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

pub fn setup_tracing_with_log_level(level: Level) {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(format!("{level},p3_=warn")));
    let _ = Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .with(MetricsLayer::new())
        .try_init();
}

/// export stark-backend metrics to the given file
fn run_with_metric_collection_to_file<R>(file: std::fs::File, f: impl FnOnce() -> R) -> R {
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let recorder = TracingContextLayer::all().layer(recorder);
    metrics::set_global_recorder(recorder).unwrap();
    let res = f();

    serde_json::to_writer_pretty(&file, &serialize_metric_snapshot(snapshotter.snapshot()))
        .unwrap();
    res
}
