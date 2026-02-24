use std::collections::HashSet;

use openvm_instructions::{LocalOpcode, VmOpcode};
use openvm_womir_transpiler::{
    BaseAlu64Opcode, BaseAluOpcode, CallOpcode, ConstOpcodes, DivRem64Opcode, DivRemOpcode,
    Eq64Opcode, EqOpcode, JumpOpcode, LessThan64Opcode, LessThanOpcode, LoadStoreOpcode,
    Mul64Opcode, MulOpcode, Shift64Opcode, ShiftOpcode,
};
use strum::IntoEnumIterator;

/// Returns the set of WOMIR opcodes allowed in autoprecompiles.
///
/// Excludes jump/call opcodes (they are branching) and hint store opcodes
/// (they use next-row references incompatible with APCs).
pub fn instruction_allowlist() -> HashSet<VmOpcode> {
    let mut allowed = HashSet::new();

    // 32-bit ALU
    allowed.extend(BaseAluOpcode::iter().map(|x| x.global_opcode()));
    // 64-bit ALU
    allowed.extend(BaseAlu64Opcode::iter().map(|x| x.global_opcode()));
    // 32-bit shifts
    allowed.extend(ShiftOpcode::iter().map(|x| x.global_opcode()));
    // 64-bit shifts
    allowed.extend(Shift64Opcode::iter().map(|x| x.global_opcode()));
    // 32-bit comparisons
    allowed.extend(LessThanOpcode::iter().map(|x| x.global_opcode()));
    // 64-bit comparisons
    allowed.extend(LessThan64Opcode::iter().map(|x| x.global_opcode()));
    // 32-bit equality
    allowed.extend(EqOpcode::iter().map(|x| x.global_opcode()));
    // 64-bit equality
    allowed.extend(Eq64Opcode::iter().map(|x| x.global_opcode()));
    // 32-bit multiplication
    allowed.extend(MulOpcode::iter().map(|x| x.global_opcode()));
    // 64-bit multiplication
    allowed.extend(Mul64Opcode::iter().map(|x| x.global_opcode()));
    // 32-bit division/remainder
    allowed.extend(DivRemOpcode::iter().map(|x| x.global_opcode()));
    // 64-bit division/remainder
    allowed.extend(DivRem64Opcode::iter().map(|x| x.global_opcode()));
    // Load/store (all variants including sign-extending loads)
    allowed.extend(LoadStoreOpcode::iter().map(|x| x.global_opcode()));
    // Const32
    allowed.extend(ConstOpcodes::iter().map(|x| x.global_opcode()));

    allowed
}

/// Returns the set of WOMIR opcodes that are branching instructions.
pub fn branch_opcodes_set() -> HashSet<VmOpcode> {
    let mut set = HashSet::new();
    set.extend(JumpOpcode::iter().map(|x| x.global_opcode()));
    set.extend(CallOpcode::iter().map(|x| x.global_opcode()));
    set
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instruction_allowlist() {
        let allowlist = instruction_allowlist();
        // Should include all ALU, shift, comparison, eq, mul, divrem, loadstore, const opcodes
        assert!(allowlist.contains(&BaseAluOpcode::ADD.global_opcode()));
        assert!(allowlist.contains(&BaseAlu64Opcode::ADD.global_opcode()));
        assert!(allowlist.contains(&ShiftOpcode::SLL.global_opcode()));
        assert!(allowlist.contains(&Shift64Opcode::SLL.global_opcode()));
        assert!(allowlist.contains(&LessThanOpcode::SLT.global_opcode()));
        assert!(allowlist.contains(&LessThan64Opcode::SLT.global_opcode()));
        assert!(allowlist.contains(&EqOpcode::EQ.global_opcode()));
        assert!(allowlist.contains(&Eq64Opcode::EQ.global_opcode()));
        assert!(allowlist.contains(&MulOpcode::MUL.global_opcode()));
        assert!(allowlist.contains(&Mul64Opcode::MUL.global_opcode()));
        assert!(allowlist.contains(&DivRemOpcode::DIV.global_opcode()));
        assert!(allowlist.contains(&DivRem64Opcode::DIV.global_opcode()));
        assert!(allowlist.contains(&LoadStoreOpcode::LOADW.global_opcode()));
        assert!(allowlist.contains(&LoadStoreOpcode::STOREW.global_opcode()));
        assert!(allowlist.contains(&ConstOpcodes::CONST32.global_opcode()));

        // Should NOT include jumps, calls, or hint store
        assert!(!allowlist.contains(&JumpOpcode::JUMP.global_opcode()));
        assert!(!allowlist.contains(&JumpOpcode::JUMP_IF.global_opcode()));
        assert!(!allowlist.contains(&CallOpcode::CALL.global_opcode()));
        assert!(!allowlist.contains(&CallOpcode::RET.global_opcode()));

        // HintStore opcodes are excluded implicitly since we only add the opcodes listed above
    }

    #[test]
    fn test_branch_opcodes_set() {
        let branches = branch_opcodes_set();
        assert!(branches.contains(&JumpOpcode::JUMP.global_opcode()));
        assert!(branches.contains(&JumpOpcode::SKIP.global_opcode()));
        assert!(branches.contains(&JumpOpcode::JUMP_IF.global_opcode()));
        assert!(branches.contains(&JumpOpcode::JUMP_IF_ZERO.global_opcode()));
        assert!(branches.contains(&CallOpcode::RET.global_opcode()));
        assert!(branches.contains(&CallOpcode::CALL.global_opcode()));
        assert!(branches.contains(&CallOpcode::CALL_INDIRECT.global_opcode()));

        // ALU opcodes should NOT be branching
        assert!(!branches.contains(&BaseAluOpcode::ADD.global_opcode()));
    }

    #[test]
    fn test_allowlist_and_branches_disjoint() {
        let allowlist = instruction_allowlist();
        let branches = branch_opcodes_set();
        assert!(
            allowlist.is_disjoint(&branches),
            "Allowlist and branch set should be disjoint"
        );
    }
}
