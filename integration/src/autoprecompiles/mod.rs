#![allow(dead_code)]

mod adapter;
mod basic_blocks;
mod instruction_handler;
mod opcodes;

use openvm_instructions::exe::VmExe;
use openvm_sdk::config::DEFAULT_APP_LOG_BLOWUP;
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use powdr_autoprecompiles::DegreeBound;

use adapter::Prog;
use basic_blocks::{collect_womir_basic_blocks, extract_jump_targets};
use instruction_handler::WomirOriginalAirs;

/// Run the autoprecompiles pipeline on a WOMIR program.
///
/// Currently panics at `WomirOriginalAirs::from_womir_config()` (TODO).
/// Once that is implemented, this will extract basic blocks and create APCs.
pub fn run_autoprecompiles(exe: &VmExe<BabyBear>, _num_apcs: u64) {
    let prog = Prog(&exe.program);
    let degree_bound = DegreeBound {
        identities: 2 * DEFAULT_APP_LOG_BLOWUP + 1,
        bus_interactions: 2 * DEFAULT_APP_LOG_BLOWUP,
    };

    // Step 1: Extract jump targets (works today)
    let jump_targets = extract_jump_targets(&prog);
    println!(
        "Program: {} instructions, {} jump targets",
        prog.0.instructions_and_debug_infos.len(),
        jump_targets.len()
    );
    println!("Jump targets: {jump_targets:?}");

    // Step 2: Build instruction handler (currently panics at todo!())
    let instruction_handler = WomirOriginalAirs::from_womir_config(degree_bound);

    // Step 3: Collect basic blocks
    let blocks = collect_womir_basic_blocks(&prog, &instruction_handler);
    println!("{} basic blocks found", blocks.len());

    // TODO Step 4: Create VmConfig, run PGO filtering, create APCs, and modify the program
}
