use std::collections::HashSet;

use openvm_instructions::{LocalOpcode, VmOpcode};
use openvm_womir_transpiler::{CallOpcode, JumpOpcode};
use strum::IntoEnumIterator;

/// Returns the set of WOMIR opcodes that are branching instructions.
pub fn branch_opcodes_set() -> HashSet<VmOpcode> {
    let mut set = HashSet::new();
    set.extend(JumpOpcode::iter().map(|x| x.global_opcode()));
    set.extend(CallOpcode::iter().map(|x| x.global_opcode()));
    set
}
