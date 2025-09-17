use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn compile_c_to_wasm(out_dir: &Path, name: &str, export_name: &str) {
    let src_path = format!("builtin_src/{name}.c");
    let wasm_output = out_dir.join(format!("{name}.wasm"));

    // Tell cargo to re-run build.rs if the C source file changes
    println!("cargo:rerun-if-changed={src_path}");

    // Compile the C file to WASM
    let status = Command::new("clang")
        .args([
            "--target=wasm32-unknown-unknown",
            "-O3",
            "-ffreestanding",
            "-nostdlib",
            "-Wl,--no-entry",
            &format!("-Wl,--export={export_name}"),
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

    compile_c_to_wasm(&out_dir, "memory_copy", "memory_copy");
    compile_c_to_wasm(&out_dir, "memory_fill", "memory_fill");
}
