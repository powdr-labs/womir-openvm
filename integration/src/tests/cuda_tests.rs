//! CUDA GPU proving tests for WOMIR.
//!
//! These tests verify GPU proving works with WOMIR's memory configuration (FP_AS).
//! Currently only system instructions (halt) can be tested on GPU because
//! WOMIR GPU tracegen is not yet implemented.

use super::isolated_tests::{TestSpec, test_prove};
use crate::instruction_builder::halt;
use crate::proving::ALL_BACKENDS;
use crate::setup_tracing_with_log_level;
use tracing::Level;

/// Test halt instruction on all backends (CPU and GPU).
#[test]
fn test_gpu_halt() {
    setup_tracing_with_log_level(Level::DEBUG);

    let spec = TestSpec {
        program: vec![halt()],
        ..Default::default()
    };
    test_prove(&spec, ALL_BACKENDS).expect("halt on all backends");
}
