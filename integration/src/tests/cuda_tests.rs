//! CUDA GPU proving tests for WOMIR.
//!
//! These tests verify GPU proving works with WOMIR's memory configuration (FP_AS).
//! GPU uses WomirPreparingGpuConfig which supports: BaseAlu32 + system instructions.
//!
//! Tests run all stages:
//! 1. Execution - verifies instruction semantics and final state
//! 2. Metered execution - verifies segmentation
//! 3. Preflight - verifies trace generation
//! 4. Mock prove - verifies circuit constraints on CPU and GPU

use super::isolated_tests::{TestSpec, test_spec_all_backends};
use crate::instruction_builder::add;
use crate::setup_tracing_with_log_level;
use tracing::Level;

/// Test halt instruction on all backends (CPU and GPU).
#[test]
fn test_gpu_halt() {
    setup_tracing_with_log_level(Level::DEBUG);

    let spec = TestSpec {
        program: vec![], // halt is appended automatically
        ..Default::default()
    };
    test_spec_all_backends(spec);
}

/// Test BaseAlu32 ADD instruction on all backends (CPU and GPU).
#[test]
fn test_gpu_add32() {
    setup_tracing_with_log_level(Level::DEBUG);

    // r2 = r0 + r1: 5 + 7 = 12
    let spec = TestSpec {
        program: vec![add(2, 0, 1)], // halt is appended automatically
        start_registers: vec![(0, 5), (1, 7)],
        expected_registers: vec![(2, 12)],
        ..Default::default()
    };
    test_spec_all_backends(spec);
}
