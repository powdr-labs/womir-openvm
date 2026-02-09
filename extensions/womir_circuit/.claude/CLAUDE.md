# womir-circuit

Circuit implementations for the WOMIR extension of OpenVM.

The extension consists of a collection of chips, each responsible for a subset of the instruction set. A chip is usually structured into an *Adapter* (responsible for fetching operands, updating the PC and frame pointer, writing back results) and a *Core* (responsible for the main computation).

## Structure

- `src/extension/` - The main OpenVM extension. This is where new chips get instantiated.
- `src/adapters/` - Adapters connecting core chips to the rest of the VM.
- `src/{base_alu,loadstore,...}/` - Chip implementations.

## Instruction spec

There is none. But you can check the WOMIR interpreter code for the semantics of each instruction:
https://raw.githubusercontent.com/powdr-labs/womir/refs/heads/main/src/wom_interpreter/mod.rs

## Patterns

The general structure of a chip should follow the same patterns of the RISC-V extension in OpenVM. The main differences are:
- Some instructions might not exist in RISC-V (e.g. `const32`)
- The adapters generally work differently, because registers are relative to a *frame pointer* (which doesn't exist natively in RISC-V).
- At this point we don't want to support AOT, TCO and CUDA, which are all behind feature flags in OpenVM. So the chips should be simplified to only support interpreted execution on CPU.

### Finding Relevant RISC-V Chips

Before adding any chip, find relevant chips in the OpenVM RISC-V extension to reuse or learn from.

**Step 1: Fetch the opcode reference** to see all available chips and their instructions:
```
https://raw.githubusercontent.com/powdr-labs/openvm/refs/tags/v1.4.2-powdr-rc.1/extensions/rv32im/transpiler/src/instructions.rs
```

This file defines all opcode enums. Each enum maps to a chip directory:
- `BaseAluOpcode` (ADD, SUB, XOR, OR, AND) → `base_alu/`
- `ShiftOpcode` (SLL, SRL, SRA) → `shift/`
- `LessThanOpcode` (SLT, SLTU) → `less_than/`
- `BranchEqualOpcode` (BEQ, BNE) → `branch_eq/`
- `BranchLessThanOpcode` (BLT, BLTU, BGE, BGEU) → `branch_lt/`
- `Rv32LoadStoreOpcode` (LOADW, STOREW, etc.) → `loadstore/`
- `Rv32JalLuiOpcode` (JAL, LUI) → `jal_lui/`
- `Rv32JalrOpcode` (JALR) → `jalr/`
- `Rv32AuipcOpcode` (AUIPC) → `auipc/`
- `MulOpcode` (MUL) → `mul/`
- `MulHOpcode` (MULH, MULHSU, MULHU) → `mulh/`
- `DivRemOpcode` (DIV, DIVU, REM, REMU) → `divrem/`

**Step 2: Identify relevant chips** - not just exact matches:
- For a comparison instruction → look at both `less_than/` AND `branch_lt/` (they share comparison logic)
- For a conditional branch → look at `branch_eq/` or `branch_lt/` depending on the condition type
- For arithmetic → look at `base_alu/` for patterns

**Step 3: Fetch the chip's files** to understand the code and be inspired or identify reusable components:
E.g.:
```
https://raw.githubusercontent.com/powdr-labs/openvm/refs/tags/v1.4.2-powdr-rc.1/extensions/rv32im/circuit/src/{chip_name}/{mod,core,execution}.rs

https://raw.githubusercontent.com/powdr-labs/openvm/refs/tags/v1.4.2-powdr-rc.1/extensions/rv32im/circuit/src/adapters/{mode,alu,branch,jalr,loadstore,mul,rdwrite}.rs
```


### Chip Structure

This is the general structure of a chip, explained using "Base ALU" as an example, which exists both in RISC-V and WOMIR.
Study the `extensions/womir_circuit/src/base_alu` directory. The main files are:
- `core.rs` ([OpenVM](https://raw.githubusercontent.com/powdr-labs/openvm/refs/tags/v1.4.2-powdr-rc.1/extensions/rv32im/circuit/src/base_alu/core.rs)) The core chip. This file might not exist at all, because we'll often be able to reuse the OpenVM RISC-V core chips without modification. This is the case for Base ALU.
- `execution.rs` ([OpenVM](https://raw.githubusercontent.com/powdr-labs/openvm/refs/tags/v1.4.2-powdr-rc.1/extensions/rv32im/circuit/src/base_alu/execution.rs), ours at `extensions/womir_circuit/src/base_alu/execution.rs`) - Execution of the instruction (both interpreting and metered execution). OpenVM will have additional logic of TCO and AOT. Remove it. Besides that, the main difference is the handling of the frame pointer.
- `mod.rs` - File tying everything together.

This chip is used with the Base ALU adapter. Study the differences between the [OpenVM variant](https://raw.githubusercontent.com/powdr-labs/openvm/refs/tags/v1.4.2-powdr-rc.1/extensions/rv32im/circuit/src/adapters/alu.rs) and ours at `extensions/womir_circuit/src/adapters/alu.rs`.

Use the `openvm-constraints` skill for guidance on writing OpenVM circuit constraints.

## Testing

```bash
# Run tests for this crate
cargo test -p womir-circuit --release
```