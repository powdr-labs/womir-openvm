use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn compile_wat_to_wasm(out_dir: &Path, name: &str) {
    let src_path = format!("builtin_src/{name}.wat");
    let wasm_output = out_dir.join(format!("{name}.wasm"));

    println!("cargo:rerun-if-changed={src_path}");

    let wasm_bytes = wat::parse_file(&src_path)
        .unwrap_or_else(|err| panic!("Failed to parse WAT source {src_path}: {err}"));

    fs::write(&wasm_output, wasm_bytes).unwrap_or_else(|err| {
        panic!(
            "Failed to write wasm output {}: {err}",
            wasm_output.display()
        )
    });

    println!(
        "cargo:rustc-env={}_WASM_PATH={}",
        name.to_uppercase(),
        wasm_output.display()
    );
}

fn compile_c_to_wasm(out_dir: &Path, name: &str) {
    let src_path = format!("builtin_src/{name}.c");
    let wasm_output = out_dir.join(format!("{name}.wasm"));

    println!("cargo:rerun-if-changed={src_path}");

    let status = Command::new("clang")
        .args([
            "--target=wasm32-unknown-unknown",
            "-O3",
            "-ffreestanding",
            "-nostdlib",
            "-Wl,--no-entry",
            "-o",
            wasm_output.to_str().unwrap(),
            &src_path,
        ])
        .status()
        .expect("Failed to execute clang");

    if !status.success() {
        panic!("Failed to compile {src_path} to WebAssembly. Please install clang and lld.");
    }

    println!(
        "cargo:rustc-env={}_WASM_PATH={}",
        name.to_uppercase(),
        wasm_output.display()
    );
}

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    compile_c_to_wasm(&out_dir, "memory_copy");
    compile_c_to_wasm(&out_dir, "memory_fill");
    compile_c_to_wasm(&out_dir, "i32_clz");
    compile_c_to_wasm(&out_dir, "i64_clz");
    compile_wat_to_wasm(&out_dir, "i32_popcnt");
    compile_wat_to_wasm(&out_dir, "i64_popcnt");
    compile_wat_to_wasm(&out_dir, "i32_ctz");
    compile_wat_to_wasm(&out_dir, "i64_ctz");
}
