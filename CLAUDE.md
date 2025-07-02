# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

womir-openvm is an OpenVM extension that implements the Womir IR instruction set. Womir is an intermediate representation compiled from WebAssembly that flattens stack and locals into infinite registers, optimized for zkVMs with write-once memory regions. This project was forked from OpenVM's rv32im extension and is being adapted to support frame pointer-based register access.

## Key Commands

- **Build Check**: `cargo check` - Verifies the code compiles without building binaries
- **Run E2E Test**: `cargo run` - Executes the main example that creates and runs Womir instructions in the VM

## Architecture

### Crate Structure
The project is organized as a Cargo workspace with three main components:

1. **womir-openvm** (root binary crate) - Main executable with example VM usage
2. **rv32im-wom/circuit** - Zero-knowledge circuit implementations for Womir operations (to be renamed from rv32im)
3. **rv32im-wom/transpiler** - Converts Womir instructions to VM-compatible format

### Key Technical Components

- **Instruction Builder** (`womir-openvm/src/instruction_builder.rs`) - Helper functions for constructing Womir instructions
- **Circuit Operations** (`rv32im-wom/circuit/src/`) - Implements various operations as ZK circuits (adapters need modification for frame pointer support):
  - ALU operations (add, sub, and, or, xor)
  - Branching (beq, bne, blt, bge)
  - Multiplication and division
  - Memory operations (load/store)
  - Jump instructions (jal, jalr)
- **Transpiler** (`rv32im-wom/transpiler/src/`) - Handles instruction conversion and VM integration

### Dependencies

The project relies heavily on the OpenVM SDK ecosystem and uses local path overrides for development. Key dependencies include:
- OpenVM SDK and runtime
- STARK backend for zero-knowledge proofs
- BabyBear prime field for cryptographic operations

### Development Notes

- Uses Rust nightly toolchain (2025-05-14)
- Currently has some unused import warnings in the build
- The main.rs file contains a working example that demonstrates basic add operations
- Local dependency overrides in Cargo.toml suggest parallel development with the main OpenVM project
- Adapters from the rv32im extension need to be reimplemented to support frame pointer-relative register access
- Need to implement Womir-specific Directives like jump and activate frame