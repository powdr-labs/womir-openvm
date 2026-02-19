mod builtin_functions;
mod const_collapse;
mod instruction_builder;
#[cfg(test)]
mod isolated_tests;
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
                                        &OpenVMSettings::with_module(&module),
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
                                    &OpenVMSettings::with_module(&module),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::womir_translation::ERROR_CODE_OFFSET;
    use instruction_builder as wom;
    use openvm_circuit::{
        arch::{ExecutionError, VmExecutor},
        system::memory::merkle::public_values::extract_public_values,
    };
    use openvm_instructions::{exe::VmExe, instruction::Instruction, program::Program};
    use openvm_sdk::{
        StdIn,
        config::{AppConfig, DEFAULT_APP_LOG_BLOWUP},
        keygen::AppProvingKey,
        prover::AppProver,
    };
    use openvm_stark_backend::p3_field::PrimeField32;
    use openvm_stark_sdk::config::{FriParameters, baby_bear_poseidon2::BabyBearPoseidon2Engine};
    use tracing::Level;
    use womir_circuit::WomirCpuBuilder;

    /// Helper function to run a VM test with given instructions and return the error or
    /// verify the output on success.
    fn run_vm_test_with_result(
        test_name: &str,
        instructions: Vec<Instruction<F>>,
        expected_output: u32,
        stdin: Option<StdIn>,
    ) -> Result<(), ExecutionError> {
        setup_tracing_with_log_level(Level::WARN);

        // Create and execute program
        let program = Program::from_instructions(&instructions);
        let exe = VmExe::new(program);
        let stdin = stdin.unwrap_or_default();

        let vm_config = WomirConfig::default();
        let vm = VmExecutor::new(vm_config.clone()).unwrap();
        let instance = vm.instance(&exe).unwrap();
        let final_state = instance.execute(stdin, None)?;
        let output = extract_public_values(
            vm_config.system.num_public_values,
            &final_state.memory.memory,
        );

        println!("{test_name} output: {output:?}");

        // Verify output
        let output_0 = u32::from_le_bytes(output[0..4].try_into().unwrap());
        assert_eq!(
            output_0, expected_output,
            "{test_name} failed: expected {expected_output}, got {output_0}"
        );

        Ok(())
    }

    /// Helper function to run a VM test with given instructions and verify the output
    fn run_vm_test(
        test_name: &str,
        instructions: Vec<Instruction<F>>,
        expected_output: u32,
        stdin: Option<StdIn>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        run_vm_test_with_result(test_name, instructions, expected_output, stdin)?;
        Ok(())
    }

    fn run_vm_test_proof_with_result(
        test_name: &str,
        instructions: Vec<Instruction<F>>,
        expected_output: u32,
        stdin: Option<StdIn>,
    ) -> Result<(), ExecutionError> {
        setup_tracing_with_log_level(Level::WARN);

        // Create and execute program
        let program = Program::from_instructions(&instructions);
        let exe = VmExe::new(program);
        let stdin = stdin.unwrap_or_default();

        let vm_config = WomirConfig::default();

        // Set app configuration
        let app_fri_params =
            FriParameters::standard_with_100_bits_conjectured_security(DEFAULT_APP_LOG_BLOWUP);
        let app_config = AppConfig::new(app_fri_params, vm_config.clone());
        let app_pk = AppProvingKey::keygen(app_config.clone()).expect("app_keygen failed");

        let mut app_prover = AppProver::<BabyBearPoseidon2Engine, WomirCpuBuilder>::new(
            WomirCpuBuilder,
            &app_pk.app_vm_pk,
            exe.clone().into(),
            app_pk.leaf_verifier_program_commit(),
        )
        .expect("app_prover failed");

        tracing::info!("Generating app proof...");
        let start = std::time::Instant::now();
        let app_proof = app_prover.prove(stdin.clone()).expect("App proof failed");
        tracing::info!("App proof took {:?}", start.elapsed());

        tracing::info!("Public values: {:?}", app_proof.user_public_values);

        let output = app_proof.user_public_values.public_values;

        println!("{test_name} output: {output:?}");

        // Verify output - convert field elements to bytes
        let output_bytes: Vec<u8> = output.iter().map(|f| f.as_canonical_u32() as u8).collect();
        let output_0 = u32::from_le_bytes(output_bytes[0..4].try_into().unwrap());
        // TODO bring this back once LoadStore is supported properly for proofs.
        assert_eq!(
            output_0, expected_output,
            "{test_name} failed: expected {expected_output}, got {output_0}"
        );

        Ok(())
    }

    fn run_vm_test_proof(
        test_name: &str,
        instructions: Vec<Instruction<F>>,
        expected_output: u32,
        stdin: Option<StdIn>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        run_vm_test_proof_with_result(test_name, instructions, expected_output, stdin)?;
        Ok(())
    }

    #[test]
    fn test_basic_add() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            // TODO uncomment when const32 is implemented
            wom::const_32_imm(0, 0, 0),
            wom::add_imm(8, 0, 666_i16),
            wom::add_imm(9, 0, 1_i16),
            wom::add(10, 8, 9),
            wom::reveal(10, 0),
            wom::halt(),
        ];

        run_vm_test("Basic WOM operations", instructions, 667, None)
    }

    #[test]
    fn test_basic_add_proof() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            // TODO uncomment when const32 is implemented
            //wom::const_32_imm(0, 0, 0),
            wom::add_imm(8, 0, 666_i16),
            wom::add_imm(9, 0, 1_i16),
            wom::add(10, 8, 9),
            wom::reveal(10, 0),
            wom::halt(),
        ];

        run_vm_test_proof("Basic WOM operations", instructions, 667, None)
    }

    #[test]
    fn test_basic_wom_operations() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![
            // TODO uncomment when const32 is implemented
            // wom::const_32_imm(0, 0, 0),
            wom::add_imm(8, 0, 666_i16),
            wom::add_imm(9, 0, 1_i16),
            wom::add(10, 8, 9),
            wom::reveal(10, 0),
            wom::halt(),
        ];

        run_vm_test("Basic WOM operations", instructions, 667, None)
    }

    #[test]
    fn test_trap() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = vec![wom::trap(42), wom::trap(8), wom::halt()];

        let err = run_vm_test_with_result("Trap instruction", instructions, 0, None).unwrap_err();
        if let ExecutionError::FailedWithExitCode(code) = err {
            assert_eq!(code, ERROR_CODE_OFFSET + 42);
        } else {
            panic!("Unexpected error: {err:?}");
        }
        Ok(())
    }

    #[test]
    fn test_basic_addi_64() {
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::const_32_imm(1, 0, 0),
            wom::add_imm_64(8, 0, 666_i16),
            wom::add_imm_64(10, 8, 1_i16),
            wom::reveal(10, 0),
            wom::halt(),
        ];

        run_vm_test("Basic addi_64", instructions, 667, None).unwrap()
    }

    #[test]
    fn test_basic_mul() {
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::add_imm(8, 0, 666_i16),
            wom::add_imm(9, 0, 1_i16),
            wom::mul(10, 8, 9),
            wom::reveal(10, 0),
            wom::halt(),
        ];

        run_vm_test("Basic multiplication", instructions, 666, None).unwrap()
    }

    #[test]
    fn test_mul_zero() {
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::add_imm(8, 0, 12345_i16),
            wom::add_imm(9, 0, 0_i16),
            wom::mul(10, 8, 9), // 12345 * 0 = 0
            wom::reveal(10, 0),
            wom::halt(),
        ];
        run_vm_test("Multiplication by zero", instructions, 0, None).unwrap()
    }

    #[test]
    fn test_mul_one() {
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::add_imm(8, 0, 999_i16),
            wom::add_imm(9, 0, 1_i16),
            wom::mul(10, 8, 9), // 999 * 1 = 999
            wom::reveal(10, 0),
            wom::halt(),
        ];
        run_vm_test("Multiplication by one", instructions, 999, None).unwrap()
    }

    #[test]
    #[should_panic]
    fn test_skip() {
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            // Sets to skip 5 instructions.
            wom::const_32_imm(8, 5, 0),
            wom::skip(8),
            //// SKIPPED BLOCK ////
            wom::halt(),
            wom::const_32_imm(10, 666, 0),
            wom::reveal(10, 0),
            wom::halt(),
            wom::halt(),
            ///////////////////////
            wom::const_32_imm(10, 42, 0),
            wom::reveal(10, 0),
            wom::halt(),
        ];
        run_vm_test("Skipping 5 instructions", instructions, 42, None).unwrap()
    }

    #[test]
    fn test_mul_powers_of_two() {
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::add_imm(8, 0, 7_i16),
            wom::add_imm(9, 0, 8_i16), // 2^3
            wom::mul(10, 8, 9),        // 7 * 8 = 56
            wom::reveal(10, 0),
            wom::halt(),
        ];
        run_vm_test("Multiplication by power of 2", instructions, 56, None).unwrap()
    }

    #[test]
    fn test_mul_large_numbers() {
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            // Load large numbers
            wom::const_32_imm(8, 1, 1), // 65537 = 0x10001 (1 << 16 | 1)
            wom::const_32_imm(9, 65521, 0), // 65521 = 0xFFF1
            wom::mul(10, 8, 9),         // 65537 * 65521 = 4,294,836,577
            wom::reveal(10, 0),
            wom::halt(),
        ];
        run_vm_test(
            "Multiplication of large numbers",
            instructions,
            4294049777u32,
            None,
        )
        .unwrap()
    }

    #[test]
    fn test_mul_overflow() {
        let instructions = vec![
            // Test multiplication that would overflow 32-bit
            wom::const_32_imm(0, 0, 0),
            wom::const_32_imm(8, 0, 1), // 2^16 = 65536 (upper=1, lower=0)
            wom::const_32_imm(9, 1, 1), // 65537 (upper=1, lower=1)
            wom::mul(10, 8, 9), // 65536 * 65537 = 4,295,032,832 (overflows to 65536 in 32-bit)
            wom::reveal(10, 0),
            wom::halt(),
        ];
        // In 32-bit arithmetic: 4,295,032,832 & 0xFFFFFFFF = 65536
        run_vm_test("Multiplication with overflow", instructions, 65536, None).unwrap()
    }

    #[test]
    fn test_mul_commutative() {
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::add_imm(8, 0, 13_i16),
            wom::add_imm(9, 0, 17_i16),
            wom::mul(10, 8, 9),   // 13 * 17 = 221
            wom::mul(11, 9, 8),   // 17 * 13 = 221 (should be same)
            wom::sub(12, 10, 11), // Should be 0 if commutative
            wom::reveal(12, 0),
            wom::halt(),
        ];
        run_vm_test("Multiplication commutativity", instructions, 0, None).unwrap()
    }

    #[test]
    fn test_mul_chain() {
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::add_imm(8, 0, 2_i16),
            wom::add_imm(9, 0, 3_i16),
            wom::add_imm(10, 0, 5_i16),
            wom::mul(11, 8, 9),   // 2 * 3 = 6
            wom::mul(12, 11, 10), // 6 * 5 = 30
            wom::reveal(12, 0),
            wom::halt(),
        ];
        run_vm_test("Chained multiplication", instructions, 30, None).unwrap()
    }

    #[test]
    fn test_mul_max_value() {
        let instructions = vec![
            // Test with maximum 32-bit value
            wom::const_32_imm(0, 0, 0),
            wom::const_32_imm(8, 0xFFFF, 0xFFFF), // 2^32 - 1
            wom::add_imm(9, 0, 1_i16),
            wom::mul(10, 8, 9), // (2^32 - 1) * 1 = 2^32 - 1
            wom::reveal(10, 0),
            wom::halt(),
        ];
        run_vm_test(
            "Multiplication with max value",
            instructions,
            0xFFFFFFFF,
            None,
        )
        .unwrap()
    }

    #[test]
    fn test_mul_negative_positive() {
        // Test multiplication of negative and positive numbers
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::const_32_imm(8, 0xFFFB, 0xFFFF), // -5 in two's complement
            wom::add_imm(9, 0, 3_i16),
            wom::mul(10, 8, 9), // -5 * 3 = -15
            wom::reveal(10, 0),
            wom::halt(),
        ];
        // -15 in 32-bit two's complement is 0xFFFFFFF1
        run_vm_test(
            "Multiplication negative * positive",
            instructions,
            0xFFFFFFF1,
            None,
        )
        .unwrap()
    }

    #[test]
    fn test_mul_positive_negative() {
        // Test multiplication of positive and negative numbers
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::add_imm(8, 0, 4_i16),
            wom::const_32_imm(9, 0xFFFA, 0xFFFF), // -6 in two's complement
            wom::mul(10, 8, 9),                   // 4 * -6 = -24
            wom::reveal(10, 0),
            wom::halt(),
        ];
        // -24 in 32-bit two's complement is 0xFFFFFFE8
        run_vm_test(
            "Multiplication positive * negative",
            instructions,
            0xFFFFFFE8,
            None,
        )
        .unwrap()
    }

    #[test]
    fn test_mul_both_negative() {
        // Test multiplication of two negative numbers
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::const_32_imm(8, 0xFFF9, 0xFFFF), // -7 in two's complement
            wom::const_32_imm(9, 0xFFFD, 0xFFFF), // -3 in two's complement
            wom::mul(10, 8, 9),                   // -7 * -3 = 21
            wom::reveal(10, 0),
            wom::halt(),
        ];
        run_vm_test("Multiplication both negative", instructions, 21, None).unwrap()
    }

    #[test]
    fn test_mul_negative_one() {
        // Test multiplication by -1
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::add_imm(8, 0, 42_i16),
            wom::const_32_imm(9, 0xFFFF, 0xFFFF), // -1 in two's complement
            wom::mul(10, 8, 9),                   // 42 * -1 = -42
            wom::reveal(10, 0),
            wom::halt(),
        ];
        // -42 in 32-bit two's complement is 0xFFFFFFD6
        run_vm_test(
            "Multiplication by negative one",
            instructions,
            0xFFFFFFD6,
            None,
        )
        .unwrap()
    }

    #[test]
    fn test_mul_negative_overflow() {
        // Test multiplication that would overflow with signed numbers
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::const_32_imm(8, 0x0000, 0x8000), // -2147483648 (INT32_MIN)
            wom::const_32_imm(9, 0xFFFF, 0xFFFF), // -1
            wom::mul(10, 8, 9),                   // INT32_MIN * -1 = INT32_MIN (overflow)
            wom::reveal(10, 0),
            wom::halt(),
        ];
        // INT32_MIN * -1 overflows back to INT32_MIN (0x80000000)
        run_vm_test(
            "Multiplication negative overflow",
            instructions,
            0x80000000,
            None,
        )
        .unwrap()
    }

    #[test]
    fn test_basic_div() {
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::add_imm(8, 0, 100_i16),
            wom::add_imm(9, 0, 10_i16),
            wom::div(10, 8, 9), // 100 / 10 = 10
            wom::reveal(10, 0),
            wom::halt(),
        ];
        run_vm_test("Basic division", instructions, 10, None).unwrap()
    }

    #[test]
    fn test_div_by_one() {
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::add_imm(8, 0, 999_i16),
            wom::add_imm(9, 0, 1_i16),
            wom::div(10, 8, 9), // 999 / 1 = 999
            wom::reveal(10, 0),
            wom::halt(),
        ];
        run_vm_test("Division by one", instructions, 999, None).unwrap()
    }

    #[test]
    fn test_div_equal_numbers() {
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::add_imm(8, 0, 42_i16),
            wom::add_imm(9, 0, 42_i16),
            wom::div(10, 8, 9), // 42 / 42 = 1
            wom::reveal(10, 0),
            wom::halt(),
        ];
        run_vm_test("Division of equal numbers", instructions, 1, None).unwrap()
    }

    #[test]
    fn test_div_with_remainder() {
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::add_imm(8, 0, 17_i16),
            wom::add_imm(9, 0, 5_i16),
            wom::div(10, 8, 9), // 17 / 5 = 3 (integer division)
            wom::reveal(10, 0),
            wom::halt(),
        ];
        run_vm_test("Division with remainder", instructions, 3, None).unwrap()
    }

    #[test]
    fn test_div_zero_dividend() {
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::add_imm(8, 0, 0_i16),
            wom::add_imm(9, 0, 100_i16),
            wom::div(10, 8, 9), // 0 / 100 = 0
            wom::reveal(10, 0),
            wom::halt(),
        ];
        run_vm_test("Division of zero", instructions, 0, None).unwrap()
    }

    #[test]
    fn test_div_large_numbers() {
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::const_32_imm(8, 0, 1000), // 65536000
            wom::const_32_imm(9, 256, 0),  // 256
            wom::div(10, 8, 9),            // 65536000 / 256 = 256000
            wom::reveal(10, 0),
            wom::halt(),
        ];
        run_vm_test("Division of large numbers", instructions, 256000, None).unwrap()
    }

    #[test]
    fn test_div_powers_of_two() {
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::add_imm(8, 0, 128_i16),
            wom::add_imm(9, 0, 8_i16), // 2^3
            wom::div(10, 8, 9),        // 128 / 8 = 16
            wom::reveal(10, 0),
            wom::halt(),
        ];
        run_vm_test("Division by power of 2", instructions, 16, None).unwrap()
    }

    #[test]
    fn test_div_chain() {
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::add_imm(8, 0, 120_i16),
            wom::add_imm(9, 0, 2_i16),
            wom::add_imm(10, 0, 3_i16),
            wom::div(11, 8, 9),   // 120 / 2 = 60
            wom::div(12, 11, 10), // 60 / 3 = 20
            wom::reveal(12, 0),
            wom::halt(),
        ];
        run_vm_test("Chained division", instructions, 20, None).unwrap()
    }

    #[test]
    fn test_div_negative_signed() {
        // Testing signed division with negative numbers
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::const_32_imm(8, 0xFFF6, 0xFFFF), // -10 in two's complement
            wom::add_imm(9, 0, 2_i16),
            wom::div(10, 8, 9), // -10 / 2 = -5
            wom::reveal(10, 0),
            wom::halt(),
        ];
        // -5 in 32-bit two's complement is 0xFFFFFFFB
        run_vm_test(
            "Signed division with negative dividend",
            instructions,
            0xFFFFFFFB,
            None,
        )
        .unwrap()
    }

    #[test]
    fn test_div_both_negative() {
        // Testing signed division with both numbers negative
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::const_32_imm(8, 0xFFEC, 0xFFFF), // -20 in two's complement
            wom::const_32_imm(9, 0xFFFB, 0xFFFF), // -5 in two's complement
            wom::div(10, 8, 9),                   // -20 / -5 = 4
            wom::reveal(10, 0),
            wom::halt(),
        ];
        run_vm_test("Signed division with both negative", instructions, 4, None).unwrap()
    }

    #[test]
    fn test_div_and_mul_inverse() {
        // Test that (a / b) * b ≈ a (with integer truncation)
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::add_imm(8, 0, 100_i16),
            wom::add_imm(9, 0, 7_i16),
            wom::div(10, 8, 9),  // 100 / 7 = 14
            wom::mul(11, 10, 9), // 14 * 7 = 98 (not 100 due to truncation)
            wom::reveal(11, 0),
            wom::halt(),
        ];
        run_vm_test(
            "Division and multiplication relationship",
            instructions,
            98,
            None,
        )
        .unwrap()
    }

    #[test]
    fn test_ret_instruction() {
        // Test RET: return to saved PC and FP
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),  // PC=0
            wom::add_imm(10, 0, 24_i16), // PC=4: x10 = 24 (return PC)
            wom::add_imm(11, 0, 0_i16),  // PC=8: x11 = 0 (saved FP)
            wom::add_imm(8, 0, 88_i16),  // PC=12: x8 = 88
            wom::ret(10, 11),            // PC=16: Return to PC=x10, FP=x11
            wom::halt(),                 // PC=20: This should be skipped
            // PC = 24 (where x10 points)
            wom::reveal(8, 0), // PC=24: reveal x8 (should be 88)
            wom::halt(),       // PC=28
        ];

        run_vm_test("RET instruction", instructions, 88, None).unwrap()
    }

    #[test]
    fn test_call_instruction() {
        // Test CALL: save PC and FP, sets a new frame then jump
        // CALL saves return PC (pc+4=8) into x10 in the new frame, then jumps to PC=16
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),      // PC=0
            wom::call::<F>(10, 11, 16, 100), // PC=4: CALL to PC=16, FP offset=100
            wom::add_imm(8, 0, 123_i16),     // PC=8: should NOT execute
            wom::halt(),                     // PC=12: padding
            // New frame starts here (PC=16), FP = old_fp + 100
            wom::reveal(10, 0), // PC=16: reveal x10 (should be 8, the return address)
            wom::halt(),        // PC=20: End
        ];

        run_vm_test("CALL instruction", instructions, 8, None).unwrap()
    }

    #[test]
    fn test_call_indirect_instruction() {
        // Test CALL_INDIRECT: save PC and FP, jump to register value
        // x12 holds the target PC, CALL_INDIRECT saves old FP into x11 in the new frame
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),               // PC=0
            wom::add_imm(12, 0, 16_i16),              // PC=4: x12 = 16 (target PC)
            wom::call_indirect::<F>(10, 11, 12, 100), // PC=8: CALL_INDIRECT to PC=x12, FP offset=100
            wom::halt(),                              // PC=12: padding
            // New frame starts here (PC=16), FP = old_fp + 100
            wom::reveal(11, 0), // PC=16: reveal x11 (should be 0, the saved old FP)
            wom::halt(),        // PC=20: End
        ];

        run_vm_test("CALL_INDIRECT instruction", instructions, 0, None).unwrap()
    }

    #[test]
    fn test_call_and_return() {
        // Test a complete call and return sequence
        // Note: When FP changes, register addressing changes too
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::add_imm(8, 0, 50_i16),      // x8 = 50 (at FP=0)
            wom::call::<F>(10, 11, 20, 100), // Call function at PC=20, FP offset=100
            wom::reveal(8, 0),               // wom::reveal x8 after return (should be 50)
            wom::halt(),
            // Function at PC = 20
            wom::const_32_imm(8, 1, 0), // x8 = 1 in new frame
            wom::ret(10, 11),           // Return using saved PC and FP
            wom::halt(),
        ];

        run_vm_test("CALL and RETURN sequence", instructions, 50, None).unwrap()
    }

    #[test]
    fn test_jump_instruction() {
        // Test unconditional JUMP
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),  // PC=0:
            wom::jump(20),               // PC=4: Jump to PC=20
            wom::add_imm(9, 0, 999_i16), // PC=8: This should be skipped
            wom::reveal(9, 0),           // PC=12: This should be skipped
            wom::halt(),                 // PC=16: Padding
            // PC = 20 (jump target)
            wom::add_imm(9, 0, 58_i16), // PC=20: x8 = 42 + 58 = 100
            wom::reveal(9, 0),          // PC=24: wom::reveal x8 (should be 100)
            wom::halt(),                // PC=28: End
        ];

        run_vm_test("JUMP instruction", instructions, 58, None).unwrap()
    }

    #[test]
    fn test_jump_if_instruction() {
        // Test conditional JUMP_IF (condition != 0)
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),  // PC=0
            wom::add_imm(9, 0, 5_i16),   // PC=4: x9 = 5 (condition != 0)
            wom::jump_if(9, 24),         // PC=8: Jump to PC=24 if x9 != 0 (should jump)
            wom::add_imm(8, 0, 999_i16), // PC=12: This should be skipped
            wom::reveal(8, 0),           // PC=16: This should be skipped
            wom::halt(),                 // PC=20: Padding
            // PC = 24 (jump target)
            wom::add_imm(8, 0, 15_i16), // PC=24: x8 = 15
            wom::reveal(8, 0),          // PC=28: wom::reveal x8 (should be 25)
            wom::halt(),                // PC=32: End
        ];

        run_vm_test(
            "JUMP_IF instruction (true condition)",
            instructions,
            15,
            None,
        )
        .unwrap()
    }

    #[test]
    fn test_jump_if_false_condition() {
        // Test conditional JUMP_IF with false condition (should not jump)
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::add_imm(9, 0, 0_i16), // PC=4: x9 = 0 (condition == 0, should not jump)
            wom::jump_if(9, 28),       // PC=8: Jump to PC=28 if x9 != 0 (should NOT jump)
            wom::add_imm(8, 0, 20_i16), // PC=12: x8 = 30 + 20 = 50 (this should execute)
            wom::reveal(8, 0),         // PC=16: wom::reveal x8 (should be 50)
            wom::halt(),               // PC=20: End
            // PC = 24 (jump target that should not be reached)
            wom::add_imm(8, 0, 999_i16), // PC=24: This should not execute
            wom::reveal(8, 0),           // PC=28: This should not execute
            wom::halt(),                 // PC=32: This should not execute
        ];

        run_vm_test(
            "JUMP_IF instruction (false condition)",
            instructions,
            20,
            None,
        )
        .unwrap()
    }

    #[test]
    fn test_jump_if_zero_instruction() {
        // Test conditional JUMP_IF_ZERO (condition == 0)
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::add_imm(9, 0, 0_i16),   // PC=4: x9 = 0 (condition == 0)
            wom::jump_if_zero(9, 24),    // PC=8: Jump to PC=24 if x9 == 0 (should jump)
            wom::add_imm(8, 0, 999_i16), // PC=12: This should be skipped
            wom::reveal(8, 0),           // PC=16: This should be skipped
            wom::halt(),                 // PC=20: Padding
            // PC = 24 (jump target)
            wom::add_imm(8, 0, 23_i16), // PC=24: x8 = 23
            wom::reveal(8, 0),          // PC=28: wom::reveal x8 (should be 100)
            wom::halt(),                // PC=32: End
        ];

        run_vm_test(
            "JUMP_IF_ZERO instruction (true condition)",
            instructions,
            23,
            None,
        )
        .unwrap()
    }

    #[test]
    fn test_jump_if_zero_false_condition() {
        // Test conditional JUMP_IF_ZERO with false condition (should not jump)
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::add_imm(9, 0, 7_i16), // PC=4: x9 = 7 (condition != 0, should not jump)
            wom::jump_if_zero(9, 28),  // PC=8: Jump to PC=28 if x9 == 0 (should NOT jump)
            wom::add_imm(8, 0, 40_i16), // PC=12: x8 = 40 (this should execute)
            wom::reveal(8, 0),         // PC=16: wom::reveal x8 (should be 100)
            wom::halt(),               // PC=20: End
            // PC = 24 (jump target that should not be reached)
            wom::add_imm(8, 0, 999_i16), // PC=24: This should not execute
            wom::reveal(8, 0),           // PC=28: This should not execute
            wom::halt(),                 // PC=32: This should not execute
        ];

        run_vm_test(
            "JUMP_IF_ZERO instruction (false condition)",
            instructions,
            40,
            None,
        )
        .unwrap()
    }

    #[test]
    fn test_const32_simple() {
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::const_32_imm(8, 0x1234, 0x5678), // Load 0x56781234 into x8
            wom::reveal(8, 0),
            wom::halt(),
        ];

        run_vm_test("CONST32 simple test", instructions, 0x56781234, None).unwrap()
    }

    #[test]
    fn test_const32_zero() {
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::const_32_imm(10, 0, 0), // Load 0 into x10
            wom::reveal(10, 0),
            wom::halt(),
        ];

        run_vm_test("CONST32 zero test", instructions, 0, None).unwrap()
    }

    #[test]
    fn test_const32_max_value() {
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::const_32_imm(12, 0xFFFF, 0xFFFF), // Load 0xFFFFFFFF into x12
            wom::reveal(12, 0),
            wom::halt(),
        ];

        run_vm_test("CONST32 max value test", instructions, 0xFFFFFFFF, None).unwrap()
    }

    #[test]
    fn test_const32_multiple_registers() {
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::const_32_imm(8, 100, 0), // Load 100 into x8
            wom::const_32_imm(9, 200, 0), // Load 200 into x9
            wom::add(11, 8, 9),           // x11 = x8 + x9 = 300
            wom::reveal(11, 0),
            wom::halt(),
        ];

        run_vm_test("CONST32 multiple registers test", instructions, 300, None).unwrap()
    }

    #[test]
    fn test_const32_with_arithmetic() {
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::const_32_imm(8, 1000, 0), // Load 1000 into x8
            wom::const_32_imm(9, 234, 0),  // Load 234 into x9
            wom::add(10, 8, 9),            // x10 = x8 + x9 = 1234
            wom::const_32_imm(11, 34, 0),  // Load 34 into x11
            wom::sub(12, 10, 11),          // x12 = x10 - x11 = 1200
            wom::reveal(12, 0),
            wom::halt(),
        ];

        run_vm_test("CONST32 with arithmetic test", instructions, 1200, None).unwrap()
    }

    #[test]
    fn test_lt_u_true() {
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::const_32_imm(8, 100, 0), // Load 100 into x8
            wom::const_32_imm(9, 200, 0), // Load 200 into x9
            wom::lt_u(10, 8, 9),          // x10 = (x8 < x9) = (100 < 200) = 1
            wom::reveal(10, 0),
            wom::halt(),
        ];

        run_vm_test("SLTU true test", instructions, 1, None).unwrap()
    }

    #[test]
    fn test_lt_u_false() {
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::const_32_imm(8, 200, 0), // Load 200 into x8
            wom::const_32_imm(9, 100, 0), // Load 100 into x9
            wom::lt_u(10, 8, 9),          // x10 = (x8 < x9) = (200 < 100) = 0
            wom::reveal(10, 0),
            wom::halt(),
        ];

        run_vm_test("SLTU false test", instructions, 0, None).unwrap()
    }

    #[test]
    fn test_lt_u_equal() {
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::const_32_imm(8, 150, 0), // Load 150 into x8
            wom::const_32_imm(9, 150, 0), // Load 150 into x9
            wom::lt_u(10, 8, 9),          // x10 = (x8 < x9) = (150 < 150) = 0
            wom::reveal(10, 0),
            wom::halt(),
        ];

        run_vm_test("SLTU equal test", instructions, 0, None).unwrap()
    }

    #[test]
    fn test_lt_s_positive() {
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::const_32_imm(8, 50, 0),  // Load 50 into x8
            wom::const_32_imm(9, 100, 0), // Load 100 into x9
            wom::lt_s(10, 8, 9),          // x10 = (x8 < x9) = (50 < 100) = 1
            wom::reveal(10, 0),
            wom::halt(),
        ];

        run_vm_test("SLT positive numbers test", instructions, 1, None).unwrap()
    }

    #[test]
    fn test_lt_s_negative() {
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::const_32_imm(8, 0xFFFF, 0xFFFF), // Load -1 into x8
            wom::const_32_imm(9, 5, 0),           // Load 5 into x9
            wom::lt_s(10, 8, 9),                  // x10 = (x8 < x9) = (-1 < 5) = 1
            wom::reveal(10, 0),
            wom::halt(),
        ];

        run_vm_test("SLT negative vs positive test", instructions, 1, None).unwrap()
    }

    #[test]
    fn test_lt_s_both_negative() {
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::const_32_imm(8, 0xFFFE, 0xFFFF), // Load -2 into x8
            wom::const_32_imm(9, 0xFFFC, 0xFFFF), // Load -4 into x9
            wom::lt_s(10, 8, 9),                  // x10 = (x8 < x9) = (-2 < -4) = 0
            wom::reveal(10, 0),
            wom::halt(),
        ];

        run_vm_test("SLT both negative test", instructions, 0, None).unwrap()
    }

    #[test]
    fn test_lt_comparison_chain() {
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::const_32_imm(8, 10, 0),  // x8 = 10
            wom::const_32_imm(9, 20, 0),  // x9 = 20
            wom::const_32_imm(10, 30, 0), // x10 = 30
            wom::lt_u(11, 8, 9),          // x11 = (10 < 20) = 1
            wom::lt_u(12, 9, 10),         // x12 = (20 < 30) = 1
            wom::and(13, 11, 12),         // x13 = x11 & x12 = 1 & 1 = 1
            wom::reveal(13, 0),
            wom::halt(),
        ];

        run_vm_test("Less than comparison chain test", instructions, 1, None).unwrap()
    }

    #[test]
    fn test_gt_u_true() {
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::const_32_imm(8, 200, 0), // Load 200 into x8
            wom::const_32_imm(9, 100, 0), // Load 100 into x9
            wom::gt_u(10, 8, 9),          // x10 = (x8 > x9) = (200 > 100) = 1
            wom::reveal(10, 0),
            wom::halt(),
        ];

        run_vm_test("GT_U true test", instructions, 1, None).unwrap()
    }

    #[test]
    fn test_gt_u_false() {
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::const_32_imm(8, 100, 0), // Load 100 into x8
            wom::const_32_imm(9, 200, 0), // Load 200 into x9
            wom::gt_u(10, 8, 9),          // x10 = (x8 > x9) = (100 > 200) = 0
            wom::reveal(10, 0),
            wom::halt(),
        ];

        run_vm_test("GT_U false test", instructions, 0, None).unwrap()
    }

    #[test]
    fn test_gt_u_equal() {
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::const_32_imm(8, 150, 0), // Load 150 into x8
            wom::const_32_imm(9, 150, 0), // Load 150 into x9
            wom::gt_u(10, 8, 9),          // x10 = (x8 > x9) = (150 > 150) = 0
            wom::reveal(10, 0),
            wom::halt(),
        ];

        run_vm_test("GT_U equal test", instructions, 0, None).unwrap()
    }

    #[test]
    fn test_gt_s_positive() {
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::const_32_imm(8, 100, 0), // Load 100 into x8
            wom::const_32_imm(9, 50, 0),  // Load 50 into x9
            wom::gt_s(10, 8, 9),          // x10 = (x8 > x9) = (100 > 50) = 1
            wom::reveal(10, 0),
            wom::halt(),
        ];

        run_vm_test("GT_S positive numbers test", instructions, 1, None).unwrap()
    }

    #[test]
    fn test_gt_s_negative() {
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::const_32_imm(8, 5, 0),           // Load 5 into x8
            wom::const_32_imm(9, 0xFFFF, 0xFFFF), // Load -1 into x9
            wom::gt_s(10, 8, 9),                  // x10 = (x8 > x9) = (5 > -1) = 1
            wom::reveal(10, 0),
            wom::halt(),
        ];

        run_vm_test("GT_S positive vs negative test", instructions, 1, None).unwrap()
    }

    #[test]
    fn test_gt_s_both_negative() {
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::const_32_imm(8, 0xFFFE, 0xFFFF), // Load -2 into x8
            wom::const_32_imm(9, 0xFFFC, 0xFFFF), // Load -4 into x9
            wom::gt_s(10, 8, 9),                  // x10 = (x8 > x9) = (-2 > -4) = 1
            wom::reveal(10, 0),
            wom::halt(),
        ];

        run_vm_test("GT_S both negative test", instructions, 1, None).unwrap()
    }

    #[test]
    fn test_gt_edge_cases() {
        let instructions = vec![
            // Test max unsigned value
            wom::const_32_imm(0, 0, 0),
            wom::const_32_imm(8, 0xFFFF, 0xFFFF), // Load 0xFFFFFFFF (max u32) into x8
            wom::const_32_imm(9, 0, 0),           // Load 0 into x9
            wom::gt_u(10, 8, 9),                  // x10 = (max > 0) = 1
            // Test with max signed positive
            wom::const_32_imm(11, 0xFFFF, 0x7FFF), // Load 0x7FFFFFFF (max positive) into x11
            wom::const_32_imm(12, 0, 0),           // Load 0 into x12
            wom::gt_s(13, 11, 12),                 // x13 = (max_pos > 0) = 1
            // Combine results
            wom::and(14, 10, 13), // x14 = 1 & 1 = 1
            wom::reveal(14, 0),
            wom::halt(),
        ];

        run_vm_test("GT edge cases test", instructions, 1, None).unwrap()
    }

    #[test]
    fn test_comparison_equivalence() {
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::const_32_imm(8, 25, 0), // x8 = 25
            wom::const_32_imm(9, 10, 0), // x9 = 10
            // Test that (a > b) == !(a <= b) == !((a < b) || (a == b))
            wom::gt_u(10, 8, 9), // x10 = (25 > 10) = 1
            wom::lt_u(11, 9, 8), // x11 = (10 < 25) = 1 (equivalent)
            // Test that gt_u and lt_u with swapped operands are equivalent
            wom::xor(12, 10, 11), // x12 = x10 XOR x11 = 1 XOR 1 = 0 (should be 0 if equivalent)
            wom::reveal(12, 0),
            wom::halt(),
        ];

        run_vm_test("Comparison equivalence test", instructions, 0, None).unwrap()
    }

    #[test]
    fn test_mixed_comparisons() {
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::const_32_imm(8, 0xFFFE, 0xFFFF), // x8 = -2 (signed)
            wom::const_32_imm(9, 2, 0),           // x9 = 2
            // Unsigned comparison: 0xFFFFFFFE > 2
            wom::gt_u(10, 8, 9), // x10 = 1 (large unsigned > small)
            // Signed comparison: -2 > 2
            wom::gt_s(11, 8, 9), // x11 = 0 (negative < positive)
            // Show the difference
            wom::sub(12, 10, 11), // x12 = 1 - 0 = 1
            wom::reveal(12, 0),
            wom::halt(),
        ];

        run_vm_test(
            "Mixed signed/unsigned comparison test",
            instructions,
            1,
            None,
        )
        .unwrap()
    }

    #[test]
    fn test_input_hint() {
        // hint_storew writes to memory AS, so we use a scratch register pointing to MEM[0].
        // Each prepare_read + hint_storew(skip len) + hint_storew(data) + loadw pattern
        // reads one u32 from stdin into a register.
        let scratch = 5; // scratch register pointing to MEM[0]
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::const_32_imm(scratch, 0, 0), // scratch_reg = 0 (points to MEM[0])
            wom::prepare_read(),
            wom::hint_storew(scratch),  // skip length word
            wom::hint_storew(scratch),  // write data to MEM[0]
            wom::loadw(10, scratch, 0), // load MEM[0] → r10
            wom::reveal(10, 0),
            wom::halt(),
        ];
        let mut stdin = StdIn::default();
        stdin.write(&42u32);

        run_vm_test("Input hint", instructions, 42, Some(stdin)).unwrap()
    }

    #[test]
    fn test_input_hint_with_frame_jump_and_xor() {
        let scratch = 5; // scratch register pointing to MEM[0]
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),       // PC=0
            wom::const_32_imm(scratch, 0, 0), // PC=4: scratch = 0
            // Read first value into r8
            wom::prepare_read(),       // PC=8
            wom::hint_storew(scratch), // PC=12: skip length
            wom::hint_storew(scratch), // PC=16: write data to MEM[0]
            wom::loadw(8, scratch, 0), // PC=20: load MEM[0] → r8
            // Store r8 to memory address 100 so the new frame can access it
            wom::add_imm(6, 0, 100_i16), // PC=24: r6 = 100
            wom::storew(8, 6, 0),        // PC=28: MEM[100] = r8
            // Call into a new frame
            wom::call::<F>(10, 11, 40, 200), // PC=32: jump to PC=40, FP += 200
            wom::halt(),                     // PC=36: padding (skipped)
            // === New frame (PC=40), FP = old_FP + 200 ===
            wom::const_32_imm(0, 0, 0),       // PC=40
            wom::const_32_imm(scratch, 0, 0), // PC=44: scratch = 0
            // Load the first value from memory into r2
            wom::add_imm(6, 0, 100_i16), // PC=48: r6 = 100
            wom::loadw(2, 6, 0),         // PC=52: r2 = MEM[100]
            // Read second value into r3
            wom::prepare_read(),       // PC=56
            wom::hint_storew(scratch), // PC=60: skip length
            wom::hint_storew(scratch), // PC=64: write data to MEM[0]
            wom::loadw(3, scratch, 0), // PC=68: load MEM[0] → r3
            // XOR the two values
            wom::xor(4, 2, 3), // PC=72
            wom::reveal(4, 0), // PC=76
            wom::halt(),       // PC=80
        ];

        let mut stdin = StdIn::default();
        stdin.write(&170u32); // First value: 170 in decimal
        stdin.write(&204u32); // Second value: 204 in decimal

        run_vm_test(
            "Input hint with frame jump and XOR",
            instructions,
            102,
            Some(stdin),
        )
        .unwrap()
    }

    #[test]
    fn test_loadw_basic() {
        // Test basic LOADW instruction
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::add_imm(8, 0, 100_i16), // x8 = 100 (base address)
            wom::add_imm(9, 0, 42_i16),  // x9 = 42 (value to store)
            wom::storew(9, 8, 0),        // MEM[x8 + 0] = x9 (store 42 at address 100)
            wom::loadw(10, 8, 0),        // x10 = MEM[x8 + 0] (load from address 100)
            wom::reveal(10, 0),          // wom::reveal x10 (should be 42)
            wom::halt(),
        ];

        run_vm_test("LOADW basic test", instructions, 42, None).unwrap()
    }

    #[test]
    fn test_storew_with_offset() {
        // Test STOREW with positive offset
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::add_imm(8, 0, 200_i16),  // x8 = 200 (base address)
            wom::add_imm(9, 0, 111_i16),  // x9 = 111 (first value)
            wom::add_imm(10, 0, 222_i16), // x10 = 222 (second value)
            wom::storew(9, 8, 0),         // MEM[x8 + 0] = 111
            wom::storew(10, 8, 4),        // MEM[x8 + 4] = 222
            wom::loadw(11, 8, 0),         // x11 = MEM[x8 + 0] (should be 111)
            wom::loadw(12, 8, 4),         // x12 = MEM[x8 + 4] (should be 222)
            // Test that we loaded the correct values
            wom::add(13, 11, 12), // x13 = x11 + x12 = 111 + 222 = 333
            wom::reveal(13, 0),   // wom::reveal x13 (should be 333)
            wom::halt(),
        ];

        run_vm_test("STOREW with offset test", instructions, 333, None).unwrap()
    }

    #[test]
    fn test_loadbu_basic() {
        // Test LOADBU instruction (load byte unsigned)
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::add_imm(8, 0, 300_i16),  // x8 = 300 (base address)
            wom::add_imm(9, 0, 0xFF_i16), // x9 = 255 (max byte value)
            wom::storeb(9, 8, 0),         // MEM[x8 + 0] = 255 (store as byte)
            wom::loadbu(10, 8, 0),        // x10 = MEM[x8 + 0] (load byte unsigned)
            wom::reveal(10, 0),           // Reveal x10 (should be 255)
            wom::halt(),
        ];
        run_vm_test("LOADBU basic test", instructions, 255, None).unwrap()
    }

    #[test]
    fn test_loadhu_basic() {
        // Test LOADHU instruction (load halfword unsigned)
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::add_imm(8, 0, 400_i16),     // x8 = 400 (base address)
            wom::const_32_imm(9, 0xABCD, 0), // x9 = 0xABCD (43981)
            wom::storeh(9, 8, 0),            // MEM[x8 + 0] = 0xABCD (store as halfword)
            wom::loadhu(10, 8, 0),           // x10 = MEM[x8 + 0] (load halfword unsigned)
            wom::reveal(10, 0),              // Reveal x10 (should be 0xABCD = 43981)
            wom::halt(),
        ];
        run_vm_test("LOADHU basic test", instructions, 0xABCD, None).unwrap()
    }

    #[test]
    fn test_storeb_with_offset() {
        // Test STOREB with offset and masking
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::add_imm(8, 0, 500_i16),     // x8 = 500 (base address)
            wom::const_32_imm(9, 0x1234, 0), // x9 = 0x1234 (only lowest byte 0x34 will be stored)
            wom::storeb(9, 8, 0),            // MEM[x8 + 0] = 0x34 (store lowest byte)
            wom::storeb(9, 8, 1),            // MEM[x8 + 1] = 0x34 (store at offset 1)
            wom::loadbu(10, 8, 0),           // x10 = MEM[x8 + 0] (should be 0x34 = 52)
            wom::loadbu(11, 8, 1),           // x11 = MEM[x8 + 1] (should be 0x34 = 52)
            wom::add(12, 10, 11),            // x12 = x10 + x11 = 52 + 52 = 104
            wom::reveal(12, 0),              // Reveal x12 (should be 104)
            wom::halt(),
        ];
        run_vm_test("STOREB with offset test", instructions, 104, None).unwrap()
    }

    #[test]
    fn test_storeh_with_offset() {
        // Test STOREH with offset
        let instructions = vec![
            wom::const_32_imm(0, 0, 0),
            wom::add_imm(8, 0, 600_i16),      // x8 = 600 (base address)
            wom::const_32_imm(9, 0x1111, 0),  // x9 = 0x1111
            wom::const_32_imm(10, 0x2222, 0), // x10 = 0x2222
            wom::storeh(9, 8, 0),             // MEM[x8 + 0] = 0x1111 (store halfword)
            wom::storeh(10, 8, 2),            // MEM[x8 + 2] = 0x2222 (store at offset 2)
            wom::loadhu(11, 8, 0),            // x11 = MEM[x8 + 0] (should be 0x1111 = 4369)
            wom::loadhu(12, 8, 2),            // x12 = MEM[x8 + 2] (should be 0x2222 = 8738)
            wom::add(13, 11, 12),             // x13 = 4369 + 8738 = 13107
            wom::reveal(13, 0),               // Reveal x13 (should be 13107)
            wom::halt(),
        ];
        run_vm_test("STOREH with offset test", instructions, 13107, None).unwrap()
    }
}

#[cfg(test)]
mod wast_tests {
    use super::*;
    use openvm_sdk::StdIn;
    use serde::Deserialize;
    use serde_json::Value;
    use std::fs;
    use std::path::Path;
    use std::process::Command;
    use tracing::Level;

    type TestCase = (String, Vec<u32>, Vec<u32>);
    type TestModule = (String, u32, Vec<TestCase>);

    #[derive(Debug, Deserialize)]
    struct TestFile {
        commands: Vec<CommandEntry>,
    }

    #[derive(Debug, Deserialize)]
    struct CommandEntry {
        #[serde(rename = "type")]
        cmd_type: String,
        filename: Option<String>,
        line: Option<u32>,
        action: Option<Action>,
        expected: Option<Vec<Expected>>,
    }

    #[derive(Debug, Deserialize)]
    struct Action {
        #[serde(rename = "type")]
        action_type: String,
        field: Option<String>,
        args: Option<Vec<Value>>,
        #[allow(dead_code)]
        module: Option<String>,
    }

    #[derive(Debug, Deserialize)]
    struct Expected {
        #[serde(rename = "type")]
        expected_type: String,
        #[allow(dead_code)]
        lane: Option<String>,
        value: Option<String>,
    }

    fn extract_wast_test_info(
        wast_file: &str,
    ) -> Result<(PathBuf, Vec<TestModule>), Box<dyn std::error::Error>> {
        // Convert .wast to .json using wast2json
        let target_dir =
            PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("../wast_target");
        fs::create_dir_all(&target_dir).unwrap();
        let wast_path = Path::new(wast_file).canonicalize()?;
        let json_path =
            target_dir.join(Path::new(wast_path.file_stem().unwrap()).with_extension("json"));

        let output = Command::new("wast2json")
            .arg(wast_path)
            .arg("--debug-names")
            .current_dir(&target_dir)
            .output()
            .unwrap();

        if !output.status.success() {
            return Err(format!(
                "wast2json failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )
            .into());
        }

        // Parse the JSON file
        let json_content = fs::read_to_string(&json_path)?;
        let test_file: TestFile = serde_json::from_str(&json_content)?;

        let mut test_cases = Vec::new();
        let mut current_module = None;
        let mut current_line = 0;
        let mut assert_cases = Vec::new();

        for cmd in test_file.commands {
            match cmd.cmd_type.as_str() {
                "module" => {
                    if let Some(module) = current_module.take()
                        && !assert_cases.is_empty()
                    {
                        test_cases.push((module, current_line, assert_cases.clone()));
                        assert_cases.clear();
                    }
                    current_module = cmd.filename;
                    current_line = cmd.line.unwrap_or(0);
                }
                "action" | "assert_return" => {
                    if let (Some(action), Some(expected)) = (cmd.action, cmd.expected)
                        && action.action_type == "invoke"
                        && let (Some(field), Some(args)) = (action.field, action.args)
                    {
                        let args_u32: Vec<u32> = args
                            .iter()
                            .filter_map(|v| {
                                if let Value::Object(obj) = v {
                                    if let Some(Value::String(val_str)) = obj.get("value") {
                                        // In OpenVM we read the inputs as u32s, so here we
                                        // need to parse the input as 32-bit limbs.
                                        if let Some(Value::String(ty_str)) = obj.get("type") {
                                            parse_as_vec_u32(ty_str, val_str)
                                        } else {
                                            Some(vec![val_str.parse::<u32>().unwrap()])
                                        }
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                }
                            })
                            .flatten()
                            .collect();

                        let expected_u32: Vec<u32> = expected
                            .iter()
                            .filter_map(|e| {
                                // Parse as 32-bit limbs for the same reason as
                                // above.
                                e.value
                                    .as_ref()
                                    .and_then(|v| parse_as_vec_u32(&e.expected_type, v))
                            })
                            .flatten()
                            .collect();

                        assert_cases.push((field, args_u32, expected_u32));
                    }
                }
                _ => {}
            }
        }

        if let Some(module) = current_module
            && !assert_cases.is_empty()
        {
            test_cases.push((module, current_line, assert_cases));
        }

        Ok((target_dir, test_cases))
    }

    fn parse_as_vec_u32(ty: &str, value: &str) -> Option<Vec<u32>> {
        if ty == "i32" {
            let v = value.parse::<u32>().unwrap();
            Some(vec![v])
        } else if ty == "i64" {
            let v = value.parse::<u64>().unwrap();
            Some(vec![v as u32, (v >> 32) as u32])
        } else {
            None
        }
    }

    #[allow(dead_code)]
    fn parse_val(s: &str) -> Result<u32, Box<dyn std::error::Error>> {
        if s.starts_with("i32.const ") {
            let val_str = s.trim_start_matches("i32.const ").trim();
            if val_str.starts_with("0x") {
                u32::from_str_radix(val_str.trim_start_matches("0x"), 16).map_err(|e| e.into())
            } else if val_str.starts_with("-0x") {
                u32::from_str_radix(val_str.trim_start_matches("-0x"), 16)
                    .map(|v| (!v).wrapping_add(1))
                    .map_err(|e| e.into())
            } else if val_str.starts_with("-") {
                val_str
                    .parse::<i32>()
                    .map(|v| v as u32)
                    .map_err(|e| e.into())
            } else {
                val_str.parse::<u32>().map_err(|e| e.into())
            }
        } else {
            Err("Unsupported value format".into())
        }
    }

    fn run_single_wasm_test(
        module_path: &str,
        function: &str,
        args: &[u32],
        expected: &[u32],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let wasm_bytes = std::fs::read(module_path).expect("Failed to read WASM file");
        let (module, functions) = load_wasm(&wasm_bytes);
        let mut module = LinkedProgram::new(module, functions);

        run_wasm_test_function(&mut module, function, args, expected)
    }

    fn run_wasm_test_function(
        module: &mut LinkedProgram<F>,
        function: &str,
        args: &[u32],
        expected: &[u32],
    ) -> Result<(), Box<dyn std::error::Error>> {
        setup_tracing_with_log_level(Level::WARN);
        println!("Running WASM test with {function}({args:?}): expected {expected:?}");

        let vm_config = WomirConfig::default();
        // Prepare input
        let mut stdin = StdIn::default();
        for &arg in args {
            stdin.write(&arg);
        }

        let output = module.execute(vm_config, function, stdin)?;

        // Verify output
        if !expected.is_empty() {
            // Read only as many bytes as expected by the test.
            let output: Vec<u32> = output[..expected.len() * 4]
                .chunks(4)
                .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            assert_eq!(
                output, expected,
                "Test failed for {function}({args:?}): expected {expected:?}, got {output:?}"
            );
        }

        Ok(())
    }

    #[test]
    fn test_i32() {
        run_wasm_test("../wasm_tests/i32.wast").unwrap()
    }

    #[test]
    #[should_panic]
    fn test_i64() {
        run_wasm_test("../wasm_tests/i64.wast").unwrap()
    }

    #[test]
    #[should_panic]
    fn test_address() {
        run_wasm_test("../wasm_tests/address.wast").unwrap()
    }

    #[test]
    #[should_panic]
    fn test_memory_grow() {
        run_wasm_test("../wasm_tests/memory_grow.wast").unwrap()
    }

    #[test]
    #[should_panic]
    fn test_call_indirect() {
        run_wasm_test("../wasm_tests/call_indirect.wast").unwrap()
    }

    #[test]
    #[should_panic]
    fn test_func() {
        run_wasm_test("../wasm_tests/func.wast").unwrap()
    }

    #[test]
    #[should_panic]
    fn test_call() {
        run_wasm_test("../wasm_tests/call.wast").unwrap()
    }

    #[test]
    #[should_panic]
    fn test_br_if() {
        run_wasm_test("../wasm_tests/br_if.wast").unwrap()
    }

    #[test]
    fn test_return() {
        run_wasm_test("../wasm_tests/return.wast").unwrap()
    }

    #[test]
    #[should_panic]
    fn test_loop() {
        run_wasm_test("../wasm_tests/loop.wast").unwrap()
    }

    #[test]
    #[should_panic]
    fn test_memory_fill() {
        run_wasm_test("../wasm_tests/memory_fill.wast").unwrap()
    }

    fn run_wasm_test(tf: &str) -> Result<(), Box<dyn std::error::Error>> {
        let (target_dir, test_cases) = extract_wast_test_info(tf)?;

        // Run all test cases
        for (module_path, _line, cases) in &test_cases {
            let full_module_path = target_dir.join(module_path);

            // Load the module to be executed multiple times.
            println!("Loading test module: {module_path}");
            let wasm_bytes = std::fs::read(full_module_path).expect("Failed to read WASM file");
            let (module, functions) = load_wasm(&wasm_bytes);
            let mut module = LinkedProgram::new(module, functions);

            for (function, args, expected) in cases {
                run_wasm_test_function(&mut module, function, args, expected)?;
            }
        }

        Ok(())
    }

    #[test]
    fn test_fib() {
        run_single_wasm_test("../sample-programs/fib_loop.wasm", "fib", &[10], &[55]).unwrap()
    }

    #[test]
    #[should_panic]
    fn test_n_first_sums() {
        run_single_wasm_test(
            "../sample-programs/n_first_sum.wasm",
            "n_first_sum",
            &[42, 0],
            &[903, 0],
        )
        .unwrap()
    }

    #[test]
    #[should_panic]
    fn test_call_indirect_wasm() {
        run_single_wasm_test("../sample-programs/call_indirect.wasm", "test", &[], &[1]).unwrap();
        run_single_wasm_test(
            "../sample-programs/call_indirect.wasm",
            "call_op",
            &[0, 10, 20],
            &[30],
        )
        .unwrap();
        run_single_wasm_test(
            "../sample-programs/call_indirect.wasm",
            "call_op",
            &[1, 10, 3],
            &[7],
        )
        .unwrap();
    }

    #[test]
    #[should_panic]
    fn test_keccak() {
        run_single_wasm_test("../sample-programs/keccak.wasm", "main", &[0, 0], &[]).unwrap()
    }

    #[test]
    #[should_panic]
    fn test_keeper_js() {
        // This is program is a stripped down version of geth, compiled for Go's js target.
        // Source: https://github.com/ethereum/go-ethereum/tree/master/cmd/keeper
        // Compile command:
        //   GOOS=js GOARCH=wasm go -gcflags=all=-d=softfloat build -tags "example" -o keeper.wasm
        run_single_wasm_test("../sample-programs/keeper_js.wasm", "run", &[0, 0], &[]).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_keccak_rust_womir() {
        run_womir_guest(
            "keccak_with_inputs",
            "main",
            &[0, 0],
            // keccak([0; 32]) = [41, ...]
            &[1, 41],
            &[],
        )
    }

    #[test]
    #[should_panic]
    fn test_keccak_rust_read_vec() {
        run_womir_guest("read_vec", "main", &[0, 0], &[0xffaabbcc, 0xeedd0066], &[])
    }

    #[test]
    fn test_keccak_rust_openvm() {
        let path = format!(
            "{}/../sample-programs/keccak_with_inputs",
            env!("CARGO_MANIFEST_DIR")
        );
        // TODO the outputs are not checked yet because powdr-openvm does not return the outputs.
        run_openvm_guest(&path, &[1], &[41]).unwrap();
    }

    fn run_womir_guest(
        case: &str,
        main_function: &str,
        func_inputs: &[u32],
        data_inputs: &[u32],
        outputs: &[u32],
    ) {
        let path = format!("{}/../sample-programs/{case}", env!("CARGO_MANIFEST_DIR"));
        build_wasm(&PathBuf::from(&path));
        let wasm_path = format!("{path}/target/wasm32-unknown-unknown/release/{case}.wasm",);
        let args = func_inputs
            .iter()
            .chain(data_inputs)
            .copied()
            .collect::<Vec<_>>();
        run_single_wasm_test(&wasm_path, main_function, &args, outputs).unwrap()
    }

    fn build_wasm(path: &PathBuf) {
        assert!(path.exists(), "Target directory does not exist: {path:?}",);

        let output = Command::new("cargo")
            .arg("build")
            .arg("--release")
            .arg("--target")
            .arg("wasm32-unknown-unknown")
            .current_dir(path)
            .output()
            .expect("Failed to run cargo build");

        if !output.status.success() {
            eprintln!("stderr:\n{}", String::from_utf8_lossy(&output.stderr));
            eprintln!("stdout:\n{}", String::from_utf8_lossy(&output.stdout));
        }

        assert!(output.status.success(), "cargo build failed for {path:?}",);
    }

    // We use powdr-openvm to run OpenVM RISC-V so we don't have to deal with
    // SdkConfig stuff and have access to autoprecompiles.
    fn run_openvm_guest(
        _guest: &str,
        _args: &[u32],
        _expected: &[u32],
    ) -> Result<(), Box<dyn std::error::Error>> {
        // setup_tracing_with_log_level(Level::WARN);
        // println!("Running OpenVM test {guest} with ({args:?}): expected {expected:?}");
        //
        // let compiled_program = powdr_openvm::compile_guest(
        //     guest,
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
        //
        // let mut stdin = StdIn::default();
        // for arg in args {
        //     stdin.write(arg);
        // }
        //
        // powdr_openvm::execute(compiled_program, stdin).unwrap();
        //
        Ok(())
    }
}
