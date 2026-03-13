use autoprecompiles::WomirISA;
use openvm_instructions::instruction::Instruction;
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use powdr_autoprecompiles::blocks::SuperBlock;
use powdr_openvm::extraction_utils::OriginalVmConfig;
use powdr_openvm::test_utils;
use std::path::Path;
use wasm_circuit::WomirConfig;

pub fn assert_machine_output(
    program: SuperBlock<Instruction<BabyBear>>,
    module_name: &str,
    test_name: &str,
) {
    let original_config = OriginalVmConfig::<WomirISA>::new(WomirConfig::default());
    let snapshot_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("apc_snapshots");
    test_utils::assert_apc_machine_output::<WomirISA>(
        &original_config,
        program,
        &snapshot_dir,
        module_name,
        test_name,
    );
}
