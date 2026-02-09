---
name: openvm-constraints
description: Reference for writing OpenVM circuit constraints. Covers VmAirWrapper, adapters, core chips, and bus interactions.
autoContext:
  - glob: extensions/**/womir_circuit/**/*.rs
---

# OpenVM Circuit Constraints

## Architecture

Each chip is a `VmAirWrapper<Adapter, Core>`:
- **Adapter**: Handles VM interactions (buses, memory, PC/fp)
- **Core**: Constrains the computation (e.g., `a = b + c`)

The `Air::eval()` implementation ([source](https://raw.githubusercontent.com/powdr-labs/openvm/refs/tags/v1.4.2-powdr-rc.1/crates/vm/src/arch/integration_api.rs), lines 268-278):

```rust
impl<AB, A, M> Air<AB> for VmAirWrapper<A, M>
where
    AB: AirBuilder,
    A: VmAdapterAir<AB>,
    M: VmCoreAir<AB, A::Interface>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();

        // 1. Split row into adapter and core columns
        let (local_adapter, local_core) = local.split_at(self.adapter.width());

        // 2. Core evaluates first, returns context with reads/writes/instruction
        let ctx = self.core.eval(builder, local_core, self.adapter.get_from_pc(local_adapter));

        // 3. Adapter evaluates using that context
        self.adapter.eval(builder, local_adapter, ctx);
    }
}
```

## Key Traits

### VmAdapterAir
```rust
pub trait VmAdapterAir<AB: AirBuilder>: BaseAir<AB::F> {
    type Interface: VmAdapterInterface<AB::Expr>;
    fn eval(&self, builder: &mut AB, local: &[AB::Var], ctx: AdapterAirContext<AB::Expr, Self::Interface>);
    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var;
}
```

### VmCoreAir
```rust
pub trait VmCoreAir<AB, I>: BaseAirWithPublicValues<AB::F> {
    fn eval(&self, builder: &mut AB, local_core: &[AB::Var], from_pc: AB::Var) -> AdapterAirContext<AB::Expr, I>;
}
```

### AdapterAirContext
```rust
pub struct AdapterAirContext<T, I: VmAdapterInterface<T>> {
    pub to_pc: Option<T>,              // None = adapter determines (e.g., PC + 4)
    pub reads: I::Reads,               // e.g., [[T; 4]; 2] for two 4-limb reads
    pub writes: I::Writes,             // e.g., [[T; 4]; 1] for one 4-limb write
    pub instruction: I::ProcessedInstruction,
}
```

## Buses

| Bus | Purpose |
|-----|---------|
| `ExecutionBridge` | PC/fp state transitions, timestamps, instruction verification |
| `FpBus` | Frame pointer state tracking (local extension) |
| `MemoryBridge` | Memory/register reads/writes |
| `BitwiseOperationLookupBus` | Range checks via lookup tables |

## Local Extensions (Frame Pointer)

This project adds frame pointer (`fp`) support. Read `extensions/womir_circuit/src/execution.rs` for:
- `ExecutionState { pc, fp, timestamp }` - extended with `fp`
- `FpBus` - tracks fp transitions
- `ExecutionBridge` - wraps OpenVM's `ExecutionBus` + our `FpBus`
- `FpKeepOrSet::Keep | Set(value)` - control fp in transitions

**Register access is fp-relative**: `register_ptr + from_state.fp`

## Common Patterns

### Timestamp Management
```rust
let mut timestamp = cols.from_state.timestamp;
let mut timestamp_pp = || {
    let ts = timestamp.clone();
    timestamp = timestamp + AB::Expr::ONE;
    ts
};
```

### Conditional Constraints
```rust
builder.when(flag).assert_eq(a, b);
builder.when(x).when(y).assert_eq(a, b);  // both conditions
builder.when(x).when_ne(y, AB::Expr::ONE).assert_eq(a, b);
```

### Memory Operations (fp-relative)
```rust
self.memory_bridge.read(
    MemoryAddress::new(RV32_REGISTER_AS, reg_ptr + from_state.fp),
    data, timestamp, &aux,
).eval(builder, is_valid);
```

### Execution Bridge
```rust
self.execution_bridge.execute_and_increment_or_set_pc(
    opcode, operands, from_state, timestamp_delta,
    (DEFAULT_PC_STEP, ctx.to_pc),  // PcIncOrSet
    FpKeepOrSet::<AB::Expr>::Keep,
).eval(builder, is_valid);
```

### Bitwise via Lookup
```rust
self.bitwise_lookup_bus.send_xor(b, c, a).eval(builder, flag);
```

## Example: BaseAluAir

### Adapter (Local)

See `extensions/womir_circuit/src/adapters/alu.rs`. Read it completely when writing adapters, to have a reference.

Include `crate::execution::{ExecutionBridge, ExecutionState}`, NOT the OpenVM variants (which don't handle `fp`):

### Core (OpenVM Upstream)

The core ([source](https://raw.githubusercontent.com/openvm-org/openvm/refs/heads/main/extensions/rv32im/circuit/src/base_alu/core.rs)) in this case can be re-used from OpenVM. It implements `VmCoreAir` and constrains the ALU operations.

Columns include the operand and result limbs, plus some internal columns (in this case flags):
```rust
#[repr(C)]
#[derive(AlignedBorrow, Debug)]
pub struct BaseAluCoreCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub a: [T; NUM_LIMBS],
    pub b: [T; NUM_LIMBS],
    pub c: [T; NUM_LIMBS],

    pub opcode_add_flag: T,
    pub opcode_sub_flag: T,
    pub opcode_xor_flag: T,
    pub opcode_or_flag: T,
    pub opcode_and_flag: T,
}
```

This is a sketch of the `VmCoreAir` implementation:
```rust
fn eval(&self, builder: &mut AB, local: &[AB::Var], _from_pc: AB::Var) -> AdapterAirContext<...> {
    // Parse the columns as the struct for easier access
    let cols: &BaseAluCoreCols<_, NUM_LIMBS, LIMB_BITS> = local_core.borrow();

    // is_valid indicates whether the current row is used.
    // It's an expression which in this case is the sum of the flags (which are mutually exclusive and boolean).
    let flags = [
        cols.opcode_add_flag,
        cols.opcode_sub_flag,
        cols.opcode_xor_flag,
        cols.opcode_or_flag,
        cols.opcode_and_flag,
    ];
    let is_valid = flags.iter().fold(AB::Expr::ZERO, |acc, &flag| {
        builder.assert_bool(flag);
        acc + flag.into()
    });
    // an is_valid column should always be boolean. Note that this ensures that flags are mutually exclusive (only one can be 1 at a time), which is also important for the constraints below.
    builder.assert_bool(is_valid.clone());

    // Some add/sub logic omitted here...

    // Some xor/and/or logic omitted here...

    // The expression the opcode should be matched to.
    // In this case, the expression is of the form: opcode_add_flag * ADD_OPCODE + opcode_sub_flag * SUB_OPCODE + ...
    // Because flags are mutually exclusive and boolean, this will match exactly one opcode, and ensure that the flags
    // and opcode (enforced by the adapter) are consistent with each other.
    let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
        self,
        flags.iter().zip(BaseAluOpcode::iter()).fold(
            AB::Expr::ZERO,
            |acc, (flag, local_opcode)| {
                acc + (*flag).into() * AB::Expr::from_u8(local_opcode as u8)
            },
        ),
    );

    AdapterAirContext {
        // Default PC update (PC -> PC + 4)
        to_pc: None,
        // I/O columns
        reads: [cols.b.map(Into::into), cols.c.map(Into::into)].into(),
        writes: [cols.a.map(Into::into)].into(),
        // Simple instruction (this might be different e.g. for conditional jumps)
        instruction: MinimalInstruction {
            is_valid,
            opcode: expected_opcode,
        }
        .into(),
    }
}
```

## Source Files

### Local
| Component | Path |
|-----------|------|
| ExecutionBridge/State/FpBus | `extensions/womir_circuit/src/execution.rs` |
| ALU Adapter | `extensions/womir_circuit/src/adapters/alu.rs` |

### OpenVM Upstream
| Component | Link | Lines |
|-----------|------|-------|
| VmAirWrapper | [integration_api.rs](https://raw.githubusercontent.com/powdr-labs/openvm/refs/tags/v1.4.2-powdr-rc.1/crates/vm/src/arch/integration_api.rs) | 268-278 |
| BaseAluCoreAir | [core.rs](https://raw.githubusercontent.com/openvm-org/openvm/refs/heads/main/extensions/rv32im/circuit/src/base_alu/core.rs) | 68-161 |

**Raw URL format**: `https://raw.githubusercontent.com/org/repo/refs/tags/TAG/path`

## Instantiation

All chips are instantiated in `extensions/womir_circuit/src/extension/mod.rs`, in `WomirCpuProverExt::extend_prover`.