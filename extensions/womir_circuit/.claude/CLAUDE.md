# womir-circuit

Circuit implementations for the WOMIR extension of OpenVM.

A chip = *Adapter* (fetching operands, PC/FP updates, writebacks) + *Core* (main computation).

## Structure

- `src/extension/` - Extension definition; where new chips get registered
- `src/adapters/` - Adapters connecting cores to the VM
- `src/{base_alu,loadstore,...}/` - Chip implementations (core chips + execution logic)

## Adding a Chip

Check powdr-labs/womir-openvm#130 for the current status of existing chips and some additional information. Also, read `extensions/transpiler/src/instructions.rs` to understand the opcode → chip mapping.

**1. Collect context**

Understand how instruction arguments are encoded:
integration/src/instruction_builder.rs

Understand which chips already exist:
extensions/womir_circuit/src/extension/mod.rs

Understand the opcode → chip mapping:
extensions/transpiler/src/instructions.rs

Check WOMIR instruction semantics:
https://raw.githubusercontent.com/powdr-labs/womir/refs/heads/main/src/wom_interpreter/mod.rs

**2. Find relevant RISC-V chips** in OpenVM to reuse or learn from. Fetch the opcode reference:
https://raw.githubusercontent.com/powdr-labs/openvm/refs/tags/v1.4.2-powdr-rc.1/extensions/rv32im/transpiler/src/instructions.rs

Opcode enums map to chip directories: `BaseAluOpcode` → `base_alu/`, `LessThanOpcode` → `less_than/`, etc.
Look for related chips too (e.g., for comparison logic, check both `less_than/` and `branch_lt/`).

**3. Fetch chip files** to study:
```
.../extensions/rv32im/circuit/src/{chip_name}/{mod,core,execution}.rs
.../extensions/rv32im/circuit/src/adapters/{alu,branch,jalr,loadstore,...}.rs
```
(Base URL: `https://raw.githubusercontent.com/powdr-labs/openvm/refs/tags/v1.4.2-powdr-rc.1`)

**4. Study existing WOMIR chips** as examples. The `base_alu/` directory shows the pattern:
- `core.rs` - Often reused directly from OpenVM (Base ALU does this)
- `execution.rs` - Interpreter logic; main difference is frame pointer handling. Remove AOT/TCO code.
- `mod.rs` - Type aliases and re-exports

Compare adapters: [OpenVM alu.rs](https://raw.githubusercontent.com/powdr-labs/openvm/refs/tags/v1.4.2-powdr-rc.1/extensions/rv32im/circuit/src/adapters/alu.rs) vs `src/adapters/alu.rs`

**5. Make sure your changes are tested**:
There should be an single instruction test in `integration/src/isolated_tests.rs` that covers the new instruction(s). Add more if needed. It might also exist but marked as should_panic if the instruction was not previously implemented; remove the should_panic.

**Key differences from RISC-V:** registers are frame-pointer-relative; no AOT/TCO/CUDA support needed.

Use the `openvm-constraints` skill for constraint-writing guidance.

## Testing

```bash
cargo test -p womir-circuit --release
```
