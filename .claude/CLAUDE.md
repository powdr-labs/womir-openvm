# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

[WOMIR-OpenVM](https://github.com/powdr-labs/womir-openvm) (this repository) implements the [WOMIR](https://github.com/powdr-labs/womir) ISA extension for [OpenVM](https://github.com/openvm-org/openvm/), enabling WebAssembly programs to be proven in a zkVM.

## Build & Test Commands

```bash
# Build
cargo build --release

# Run all tests
cargo test --release

# Run a single test
cargo test --release test_name

# Lint
cargo clippy --all-targets -- -D warnings

# Format
cargo fmt --all -- --check
```

## Architecture

### Workspace Crates

- **integration** - CLI for loading WASM, translating to WOMIR, and proving execution
- **extensions/transpiler** (`openvm-womir-transpiler`) - Transpiles WOMIR instructions to OpenVM format
- **extensions/womir_circuit** (`womir-circuit`) - Circuit implementation for the WOMIR extension

### Key Dependencies

Before any task, read `Cargo.toml` to understand the key dependencies.
