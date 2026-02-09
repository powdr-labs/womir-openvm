# womir-circuit

Circuit implementations for the WOMIR extension of OpenVM.

The extension consists of a collection of chips, each responsible for a subset of the instruction set. A chip is usually structured into an *Adapter* (responsible for fetching operands, updating the PC and frame pointer, writing back results) and a *Core* (responsible for the main computation).

## Structure

- `src/extension/` - The main OpenVM extension. This is where new chips get instantiated.
- `src/adapters/` - Adapters connecting core chips to the rest of the VM.
- `src/{base_alu,loadstore,...}/` - Chip implementations.

## Patterns

The general structure of a chip should follow the same patterns of the RISC-V extension in OpenVM. The main differences are:
- Some instructions might not exist in RISC-V (e.g. `const32`)
- The adapters generally work differently, because registers are relative to a *frame pointer* (which doesn't exist natively in RISC-V).
- At this point we don't want to support AOT, TCO and CUDA, which are all behind feature flags in OpenVM. So the chips should be simplified to only support interpreted execution on CPU.

Before adding any chip, check https://github.com/powdr-labs/openvm/tree/v1.4.2-powdr-rc.1/extensions/rv32im/circuit/src to see if a similar chip exists.

This is the general structure of a chip, explained using "Base ALU" as an example, which exists both in RISC-V and WOMIR.
Study the `extensions/womir_circuit/src/base_alu` directory. The main files are:
- `core.rs` ([OpenVM](https://raw.githubusercontent.com/powdr-labs/openvm/refs/tags/v1.4.2-powdr-rc.1/extensions/rv32im/circuit/src/base_alu/core.rs)) The core chip. This file might not exist at all, because we'll often be able to reuse the OpenVM RISC-V core chips without modification. This is the case for Base ALU.
- `execution.rs` ([OpenVM](https://raw.githubusercontent.com/powdr-labs/openvm/refs/tags/v1.4.2-powdr-rc.1/extensions/rv32im/circuit/src/base_alu/execution.rs), ours at `extensions/womir_circuit/src/base_alu/execution.rs`) - Execution of the instruction (both interpreting and metered execution). OpenVM will have additional logic of TCO and AOT. Remove it. Besides that, the main difference is the handling of the frame pointer.
- `mod.rs` - File tying everything together.

This chip is used with the Base ALU adapter. Study the differences between the [OpenVM variant](https://raw.githubusercontent.com/powdr-labs/openvm/refs/tags/v1.4.2-powdr-rc.1/extensions/rv32im/circuit/src/adapters/alu.rs) and ours at `extensions/womir_circuit/src/adapters/alu.rs`.

## Testing

```bash
# Run tests for this crate
cargo test -p womir-circuit --release
```

## Writing Constraints

Use the `openvm-constraints` skill for guidance on writing OpenVM circuit constraints.
