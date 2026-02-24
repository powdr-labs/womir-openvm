use std::collections::BTreeSet;

use openvm_instructions::LocalOpcode;
use openvm_instructions::program::DEFAULT_PC_STEP;
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use openvm_womir_transpiler::{CallOpcode, JumpOpcode};
use powdr_autoprecompiles::blocks::{BasicBlock, Program, collect_basic_blocks};

use super::adapter::{Instr, Prog, WomirApcAdapter};
use super::instruction_handler::WomirOriginalAirs;

/// Collects basic blocks from a WOMIR program.
///
/// Extracts static jump targets from the instruction stream, then delegates
/// to the powdr-autoprecompiles `collect_basic_blocks` function.
pub fn collect_womir_basic_blocks(
    program: &Prog<'_, BabyBear>,
    instruction_handler: &WomirOriginalAirs<BabyBear>,
) -> Vec<BasicBlock<Instr<BabyBear>>> {
    let jumpdest_set = extract_jump_targets(program);
    collect_basic_blocks::<WomirApcAdapter>(program, &jumpdest_set, instruction_handler)
}

/// Extracts static jump targets from a WOMIR program's instruction stream.
///
/// WOMIR doesn't have ELF labels, so we scan instructions for jump/call targets.
/// Dynamic targets (SKIP, CALL_INDIRECT, RET) cannot be determined statically.
///
/// Instruction operand encoding (from instruction_builder.rs):
/// - JUMP:         a = to_pc_imm (absolute PC target)
/// - JUMP_IF:      a = to_pc_imm, b = condition_reg
/// - JUMP_IF_ZERO: a = to_pc_imm, b = condition_reg
/// - CALL:         c = to_pc_imm (immediate PC target)
/// - SKIP:         b = offset register (dynamic, not extractable)
/// - CALL_INDIRECT: c = to_pc register (dynamic, not extractable)
/// - RET:          c/d = registers (dynamic, not extractable)
fn extract_jump_targets<F: PrimeField32>(program: &Prog<'_, F>) -> BTreeSet<u64> {
    let mut targets = BTreeSet::new();

    let jump_opcode = JumpOpcode::JUMP.global_opcode();
    let jump_if_opcode = JumpOpcode::JUMP_IF.global_opcode();
    let jump_if_zero_opcode = JumpOpcode::JUMP_IF_ZERO.global_opcode();
    let call_opcode = CallOpcode::CALL.global_opcode();

    for (i, instr) in program.instructions().enumerate() {
        let pc = program.instruction_index_to_pc(i);
        let opcode = instr.0.opcode;

        if opcode == jump_opcode {
            // JUMP: target = operand a (absolute PC)
            let target = instr.0.a.as_canonical_u32() as u64;
            targets.insert(target);
        } else if opcode == jump_if_opcode || opcode == jump_if_zero_opcode {
            // JUMP_IF / JUMP_IF_ZERO: target = operand a (absolute PC)
            let target = instr.0.a.as_canonical_u32() as u64;
            targets.insert(target);
            // Fall-through is also a block boundary
            targets.insert(pc + DEFAULT_PC_STEP as u64);
        } else if opcode == call_opcode {
            // CALL: target = operand c (immediate PC)
            let target = instr.0.c.as_canonical_u32() as u64;
            targets.insert(target);
        }
        // SKIP, CALL_INDIRECT, RET have dynamic targets — they will still
        // create block boundaries via is_branching() in collect_basic_blocks
    }

    targets
}
