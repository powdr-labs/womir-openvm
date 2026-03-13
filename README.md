# powdr-wasm

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
cargo run -r -- run sample-programs/fib_loop.wasm fib --input 10

# Keccak (build first, then run):
cargo build --manifest-path sample-programs/keccak_with_inputs/Cargo.toml \
    --target wasm32-unknown-unknown --release

# The guest arguments are, in order:
# 0, 0: the arguments to the `main` function (argcv and argv). This is needed for any guest compiled from Rust.
# 1: the number of iterations to run Keccak (from the guest logic).
# 41: the first byte of the resulting hash (from the guest logic).
cargo run -r -- run \
    sample-programs/keccak_with_inputs/target/wasm32-unknown-unknown/release/keccak_with_inputs.wasm \
    main --input 0 --input 0 --input 1 --input 41
```

Each `--input` is either a u32 literal or `file:<path>` for binary file contents, given in the order the guest reads them.

### Mock proof (constraint verification only)

```bash
cargo run -r -- mock-prove sample-programs/fib_loop.wasm fib --input 10
```

### Prove (full cryptographic proof)

```bash
cargo run -r -- prove sample-programs/fib_loop.wasm fib --input 10
```
