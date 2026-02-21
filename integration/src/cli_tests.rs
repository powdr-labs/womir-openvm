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
    assert!(stdout.contains("output:"));
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
