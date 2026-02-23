use crate::*;
use openvm_circuit::arch::VmState;
use openvm_sdk::StdIn;
use serde::Deserialize;
use serde_json::Value;
use std::fs;
use std::path::Path;
use std::process::Command;
use tracing::Level;

use super::helpers;

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

fn load_wasm_module(wasm_bytes: &[u8]) -> LinkedProgram<'_, F> {
    let (module, functions) = load_wasm(wasm_bytes);
    LinkedProgram::new(module, functions)
}

fn run_and_prove_single_wasm_test(
    module_path: &str,
    function: &str,
    args: &[u32],
    expected: &[u32],
) -> Result<(), Box<dyn std::error::Error>> {
    let wasm_bytes = std::fs::read(module_path).expect("Failed to read WASM file");
    let mut module = load_wasm_module(&wasm_bytes);
    run_wasm_test_function(&mut module, function, args, expected, true)
}

/// Run a WASM program through execution with output verification.
/// When `prove` is true, also runs metered execution, preflight, and mock
/// proof (all stages). Supports multi-segment programs.
fn run_wasm_test_function(
    module: &mut LinkedProgram<F>,
    function: &str,
    args: &[u32],
    expected: &[u32],
    prove: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    setup_tracing_with_log_level(Level::WARN);
    println!("Running WASM test with {function}({args:?}): expected {expected:?}");

    // Capture the exe before module.execute() mutates memory_image.
    let exe = module.program_with_entry_point(function);
    let vm_config = WomirConfig::default();

    let make_state = || {
        let mut stdin = StdIn::default();
        for &arg in args {
            stdin.write(&arg);
        }
        VmState::initial(&vm_config.system, &exe.init_memory, exe.pc_start, stdin)
    };

    // Execution (also updates module.memory_image for wast test reuse)
    println!("  Execution");
    let mut stdin = StdIn::default();
    for &arg in args {
        stdin.write(&arg);
    }
    let output = module.execute(vm_config.clone(), function, stdin)?;

    if !expected.is_empty() {
        let output: Vec<u32> = output[..expected.len() * 4]
            .chunks(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(
            output, expected,
            "Test failed for {function}({args:?}): expected {expected:?}, got {output:?}"
        );
    }

    if !prove {
        return Ok(());
    }

    // Metered execution
    println!("  Metered execution");
    let (segments, _) = helpers::test_metered_execution(&exe, make_state)?;
    println!("    {} segment(s)", segments.len());

    // Preflight
    println!("  Preflight");
    helpers::test_preflight(&exe, make_state)?;

    // Mock proof
    println!("  Mock proof");
    helpers::test_prove(&exe, make_state)?;

    Ok(())
}

#[test]
fn test_i32() {
    run_wasm_test("../wasm_tests/i32.wast").unwrap()
}

#[test]
fn test_i64() {
    run_wasm_test("../wasm_tests/i64.wast").unwrap()
}

#[test]
fn test_address() {
    run_wasm_test("../wasm_tests/address.wast").unwrap()
}

#[test]
fn test_memory_grow() {
    run_wasm_test("../wasm_tests/memory_grow.wast").unwrap()
}

#[test]
fn test_call_indirect() {
    run_wasm_test("../wasm_tests/call_indirect.wast").unwrap()
}

#[test]
fn test_func() {
    run_wasm_test("../wasm_tests/func.wast").unwrap()
}

#[test]
fn test_call() {
    run_wasm_test("../wasm_tests/call.wast").unwrap()
}

#[test]
fn test_br_if() {
    run_wasm_test("../wasm_tests/br_if.wast").unwrap()
}

#[test]
fn test_return() {
    run_wasm_test("../wasm_tests/return.wast").unwrap()
}

#[test]
fn test_loop() {
    run_wasm_test("../wasm_tests/loop.wast").unwrap()
}

#[test]
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
        let mut module = load_wasm_module(&wasm_bytes);

        for (function, args, expected) in cases {
            run_wasm_test_function(&mut module, function, args, expected, false)?;
        }
    }

    Ok(())
}

#[test]
fn test_fib() {
    run_and_prove_single_wasm_test("../sample-programs/fib_loop.wasm", "fib", &[10], &[55]).unwrap()
}

#[test]
fn test_n_first_sums() {
    run_and_prove_single_wasm_test(
        "../sample-programs/n_first_sum.wasm",
        "n_first_sum",
        &[42, 0],
        &[903, 0],
    )
    .unwrap()
}

#[test]
fn test_call_indirect_wasm() {
    run_and_prove_single_wasm_test("../sample-programs/call_indirect.wasm", "test", &[], &[1])
        .unwrap();
    run_and_prove_single_wasm_test(
        "../sample-programs/call_indirect.wasm",
        "call_op",
        &[0, 10, 20],
        &[30],
    )
    .unwrap();
    run_and_prove_single_wasm_test(
        "../sample-programs/call_indirect.wasm",
        "call_op",
        &[1, 10, 3],
        &[7],
    )
    .unwrap();
}

#[test]
fn test_keccak() {
    run_and_prove_single_wasm_test("../sample-programs/keccak.wasm", "main", &[0, 0], &[]).unwrap()
}

#[test]
fn test_keeper_js() {
    // This is program is a stripped down version of geth, compiled for Go's js target.
    // Source: https://github.com/ethereum/go-ethereum/tree/master/cmd/keeper
    // Compile command:
    //   GOOS=js GOARCH=wasm go -gcflags=all=-d=softfloat build -tags "example" -o keeper.wasm
    run_and_prove_single_wasm_test("../sample-programs/keeper_js.wasm", "run", &[0, 0], &[])
        .unwrap();
}

fn keccak_rust_womir(iterations: u32, expected_first_byte: u32) {
    run_womir_guest(
        "keccak_with_inputs",
        "main",
        &[0, 0],
        &[iterations, expected_first_byte],
        &[],
    )
}

#[test]
fn test_keccak_rust_womir_1() {
    // keccak([0; 32]) = [0x29, ...], 0x29 = 41
    keccak_rust_womir(1, 41);
}

#[test]
fn test_keccak_rust_womir_2() {
    // keccak^2([0; 32]) = [0x51, ...], 0x51 = 81
    keccak_rust_womir(2, 81);
}

#[test]
fn test_keccak_rust_womir_3() {
    // keccak^3([0; 32]) = [0x35, ...], 0x35 = 53
    keccak_rust_womir(3, 53);
}

#[test]
#[should_panic]
fn test_keccak_rust_womir_1_wrong() {
    keccak_rust_womir(1, 42);
}

#[test]
#[should_panic]
fn test_keccak_rust_womir_2_wrong() {
    keccak_rust_womir(2, 82);
}

#[test]
#[should_panic]
fn test_keccak_rust_womir_3_wrong() {
    keccak_rust_womir(3, 54);
}

#[test]
fn test_keccak_rust_read_vec() {
    run_womir_guest("read_vec", "main", &[0, 0], &[0xffaabbcc, 0xeedd0066], &[])
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
    run_and_prove_single_wasm_test(&wasm_path, main_function, &args, outputs).unwrap()
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
