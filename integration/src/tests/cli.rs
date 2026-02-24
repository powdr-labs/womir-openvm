use std::path::PathBuf;
use std::process::Command;

fn cargo_bin() -> Command {
    // The test binary lives in target/<profile>/deps/; the CLI binary is one level up.
    let test_exe = std::env::current_exe().unwrap();
    let bin_dir = test_exe.parent().unwrap().parent().unwrap();
    let bin = bin_dir.join("womir-openvm-integration");
    assert!(bin.exists(), "CLI binary not found at {}", bin.display());
    Command::new(bin)
}

fn sample_program(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../sample-programs")
        .join(name)
}

/// Build a WASM crate if its output doesn't already exist.
fn build_wasm(crate_dir: &PathBuf) {
    let output = Command::new("cargo")
        .args(["build", "--release", "--target", "wasm32-unknown-unknown"])
        .current_dir(crate_dir)
        .output()
        .expect("Failed to run cargo build for WASM crate");
    assert!(
        output.status.success(),
        "cargo build failed for {}:\n{}",
        crate_dir.display(),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn test_cli_print_fib() {
    let output = cargo_bin()
        .args(["print", sample_program("fib_loop.wasm").to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Function 0:"));
}

#[test]
fn test_cli_run_fib() {
    let output = cargo_bin()
        .args([
            "run",
            sample_program("fib_loop.wasm").to_str().unwrap(),
            "fib",
            "--args",
            "10",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    // fib(10) = 55
    assert!(
        stdout.contains("output: [55,"),
        "expected fib(10) = 55, got: {stdout}"
    );
}

#[test]
fn test_cli_run_n_first_sum() {
    let output = cargo_bin()
        .args([
            "run",
            sample_program("n_first_sum.wasm").to_str().unwrap(),
            "n_first_sum",
            "--args",
            "42",
            "--args",
            "0",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn test_cli_run_keccak() {
    // keccak([0; 32]) = [0x29, ...], 0x29 = 41.
    // The WASM program internally asserts the first byte matches, so a successful
    // exit means the output is correct.
    build_wasm(&sample_program("keccak_with_inputs"));
    let wasm = sample_program(
        "keccak_with_inputs/target/wasm32-unknown-unknown/release/keccak_with_inputs.wasm",
    );
    let output = cargo_bin()
        .args([
            "run",
            wasm.to_str().unwrap(),
            "main",
            "--args",
            "0",
            "--args",
            "0",
            "--args",
            "1",
            "--args",
            "41",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn test_cli_run_keccak_wrong_output_fails() {
    build_wasm(&sample_program("keccak_with_inputs"));
    let wasm = sample_program(
        "keccak_with_inputs/target/wasm32-unknown-unknown/release/keccak_with_inputs.wasm",
    );
    let output = cargo_bin()
        .args([
            "run",
            wasm.to_str().unwrap(),
            "main",
            "--args",
            "0",
            "--args",
            "0",
            "--args",
            "1",
            "--args",
            "42",
        ])
        .output()
        .unwrap();
    assert!(
        !output.status.success(),
        "Expected failure with wrong expected byte, but process succeeded"
    );
}

#[test]
fn test_cli_prove_fib() {
    let output = cargo_bin()
        .args([
            "prove",
            sample_program("fib_loop.wasm").to_str().unwrap(),
            "fib",
            "--args",
            "10",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("Proof verified successfully."),
        "unexpected output: {stdout}"
    );
}

#[test]
fn test_cli_mock_prove_fib() {
    let output = cargo_bin()
        .args([
            "mock-prove",
            sample_program("fib_loop.wasm").to_str().unwrap(),
            "fib",
            "--args",
            "10",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("Mock proof verified successfully."),
        "unexpected output: {stdout}"
    );
}
