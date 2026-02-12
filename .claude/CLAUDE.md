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
  - `integration/src/main.rs` also includes integration tests
- **extensions/transpiler** (`openvm-womir-transpiler`) - Transpiles WOMIR instructions to OpenVM format
- **extensions/womir_circuit** (`womir-circuit`) - Circuit implementation for the WOMIR extension

### Key Dependencies

Before any task, read `Cargo.toml` to understand the key dependencies.

## GitHub workflow

Use the `gh` CLI tool to interact with GitHub when needed. For example, to access issues, pull requests, comments, files, etc.

When asked to push your changes:
- Make sure you ran clippy, fmt, and all relevant tests.
- Check what the current branch is and remember it.
- Create a new branch, like `feature-name`
- Commit your changes. If possible, make many small commits that make it easy to follow the history.
- Never force push, unless explicitly asked.
- Open a draft PR, using the previous branch as the base.
- Keep your description minimal, but include:
  - A link to the relevant issue (if it exists)
  - A brief motivation (unless obvious)
  - A brief description of the changes you made
  - If the PR contains anything unexpected, explain it.

You might also be asked to check CI results. Make sure it passes.