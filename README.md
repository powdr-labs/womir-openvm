# womir-openvm

This repository consists of:

- [WOMIR](https://github.com/powdr-labs/womir) ISA extension implementation for [OpenVM](https://github.com/openvm-org/openvm/).
- WOMIR -> OpenVM transpiler.
- OpenVM extension integration.

## Usage

### Print OpenVM instructions

```bash
cargo run -r -- print sample-programs/fib_loop.wasm
```

### Run a WASM program

```bash
# Fibonacci: fib(10) = 55
cargo run -r -- run sample-programs/fib_loop.wasm fib --args 10

# Keccak (build first, then run):
cargo build --manifest-path sample-programs/keccak_with_inputs/Cargo.toml \
    --target wasm32-unknown-unknown --release
cargo run -r -- run \
    sample-programs/keccak_with_inputs/target/wasm32-unknown-unknown/release/keccak_with_inputs.wasm \
    main --args 0 --args 0 --args 1 --args 42
```

### Mock proof (constraint verification)

```bash
cargo run -r -- prove sample-programs/fib_loop.wasm fib 10
```
