use openvm_circuit::{arch::*, system::memory::online::TracingMemory};
use openvm_instructions::{LocalOpcode, instruction::Instruction};
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_womir_transpiler::HintStoreOpcode::{HINT_BUFFER, HINT_STOREW};

// Core executor that implements FpPreflightExecutor
#[derive(Clone, Copy, derive_new::new)]
pub struct HintStoreCoreExecutor {
    pub pointer_max_bits: usize,
    pub offset: usize,
}

// FpPreflightExecutor implementation for HintStore
// This is a stub - proving path not yet implemented
// Execution works through InterpreterExecutor in execution.rs
impl<F, RA> crate::FpPreflightExecutor<F, RA> for HintStoreCoreExecutor
where
    F: PrimeField32,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        if opcode == HINT_STOREW.global_opcode().as_usize() {
            String::from("HINT_STOREW")
        } else if opcode == HINT_BUFFER.global_opcode().as_usize() {
            String::from("HINT_BUFFER")
        } else {
            unreachable!("unsupported opcode: {opcode}")
        }
    }

    fn execute_with_fp(
        &self,
        _state: VmStateMut<F, TracingMemory, RA>,
        _instruction: &Instruction<F>,
        _fp: u32,
    ) -> Result<Option<u32>, ExecutionError> {
        // TODO: Implement proper proving support for HintStore
        // Execution works through InterpreterExecutor, this is only for proving
        panic!("HintStoreCoreExecutor proving path not yet implemented");
    }
}
