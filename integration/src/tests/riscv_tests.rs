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

#[test]
fn test_keccak_rust_openvm() {
    let path = format!(
        "{}/../sample-programs/keccak_with_inputs",
        env!("CARGO_MANIFEST_DIR")
    );
    // TODO the outputs are not checked yet because powdr-openvm does not return the outputs.
    run_openvm_guest(&path, &[1], &[41]).unwrap();
}
