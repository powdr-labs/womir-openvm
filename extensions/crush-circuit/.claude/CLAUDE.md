# crush-circuit

Circuit implementations for the crush extension of OpenVM.

A chip = *Adapter* (fetching operands, PC/FP updates, writebacks) + *Core* (main computation).

## Structure

- `src/extension/` - Extension definition; where new chips get registered
- `src/adapters/` - Adapters connecting cores to the VM
- `src/{base_alu,loadstore,...}/` - Chip implementations (core chips + execution logic)

## Adding a Chip

Check powdr-labs/powdr-wasm#130 for the current status of existing chips and some additional information. Also, read `extensions/openvm-transpiler/src/instructions.rs` to understand the opcode → chip mapping.

**1. Collect context**

Understand how instruction arguments are encoded:
powdr-wasm/src/instruction_builder.rs

Understand which chips already exist:
extensions/crush-circuit/src/extension/mod.rs

Understand the opcode → chip mapping:
extensions/openvm-transpiler/src/instructions.rs

Check crush instruction semantics:
`<crush>/src/interpreter/mod.rs`

**2. Find relevant RISC-V chips** in OpenVM to reuse or learn from. Read the opcode reference:
`<openvm>/extensions/rv32im/transpiler/src/instructions.rs`

Opcode enums map to chip directories: `BaseAluOpcode` → `base_alu/`, `LessThanOpcode` → `less_than/`, etc.
Look for related chips too (e.g., for comparison logic, check both `less_than/` and `branch_lt/`).

**3. Read chip files** to study:
```
<openvm>/extensions/rv32im/circuit/src/{chip_name}/{mod,core,execution}.rs
<openvm>/extensions/rv32im/circuit/src/adapters/{alu,branch,jalr,loadstore,...}.rs
```
(Use the `resolve-dep-paths` skill to resolve these to local paths.)

**4. Study existing crush chips** as examples. The `base_alu/` directory shows the pattern:
- `core.rs` - Often reused directly from OpenVM (Base ALU does this)
- `execution.rs` - Interpreter logic; main difference is frame pointer handling. Remove AOT/TCO code.
- `mod.rs` - Type aliases and re-exports

Compare adapters: `<openvm>/extensions/rv32im/circuit/src/adapters/alu.rs` vs `src/adapters/alu.rs`

**5. Make sure your changes are tested**:
There should be a single instruction test in `powdr-wasm/src/isolated_tests.rs` that covers the new instruction(s). Add more if needed. There might also be relevant tests in `powdr-wasm/src/main.rs` marked as should_panic if the instruction was not previously implemented; remove the should_panic. There is also a snapshot test in `extensions/crush-circuit/tests/machine_extraction.rs`. If it fails, it'll print instructions on how to update the snapshot.

**Key differences from RISC-V:** registers are frame-pointer-relative; no AOT/TCO/CUDA support needed.

Use the `openvm-constraints` skill for constraint-writing guidance.

## Testing

```bash
cargo test -p crush-circuit --release
```
