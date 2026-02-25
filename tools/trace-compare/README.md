# Trace Comparison Tools

Python scripts for comparing execution traces between the **womir interpreter** and **womir-openvm**. These were developed during debugging of the DivRem immediate-mode bug and are useful for diagnosing any future divergence between the two backends.

## Prerequisites

- Python 3.8+
- No external dependencies (stdlib only)

## Generating Traces

### OpenVM trace

Add `eprintln!` tracing to chip executors (see each chip's `execute_e12_impl`), then run:

```bash
cargo run -r -p womir-openvm-integration -- run <wasm> "main" \
    --args 0 --args 0 --binary-input-files <input.bin> 2> openvm-trace.txt
```

### Womir trace

```bash
womir -e rw --trace <wasm> 2> womir-trace.txt
```

For memory writes only:

```bash
grep '^MW ' womir-trace.txt > womir-memwrites.txt
```

## Tools

### `verify_alu.py` -- Verify arithmetic correctness (single-backend)

Checks that every traced ALU, comparison, shift, and MUL operation in an openvm trace computes the correct result. Use this first to rule out basic arithmetic bugs.

```bash
python verify_alu.py openvm-trace.txt
python verify_alu.py openvm-trace.txt --64bit-only
```

### `compare_calls.py` -- Compare function call sequences (cross-backend)

Compares CALL/RET event sequences between openvm and womir, filtering out builtin function calls. Shows where the first function-level divergence occurs.

```bash
python compare_calls.py openvm-trace.txt womir-trace.txt
python compare_calls.py openvm-trace.txt womir-trace.txt --builtin-threshold 1565
```

### `compare_muls.py` -- Compare MUL results by input pairs (cross-backend)

Groups MUL operations by their `(in1, in2)` input pairs and checks that both backends produce the same outputs. Needed because builtin MULs interleave in openvm, making sequential comparison unreliable.

```bash
python compare_muls.py openvm-trace.txt womir-trace.txt
```

### `find_first_divergent_write.py` -- Word-level write divergence (cross-backend)

For each heap address, compares the sequence of STOREW writes from each backend. Fast and lightweight.

```bash
python find_first_divergent_write.py openvm-trace.txt womir-memwrites.txt
python find_first_divergent_write.py openvm-trace.txt womir-memwrites.txt --max-line 473404
```

### `find_first_divergence.py` -- Byte-level memory divergence (cross-backend)

Replays all stores (STOREW/STOREH/STOREB) from both traces into byte-level memory images, compares final states, and identifies the first divergent range with full store attribution. Most thorough but slowest.

```bash
python find_first_divergence.py openvm-trace.txt womir-memwrites.txt
python find_first_divergence.py openvm-trace.txt womir-memwrites.txt --offset 0x2120
```

### `trace_crash.py` -- Trace backward from crash (single-backend)

Shows the last N detailed trace lines before a crash. Useful for tracing backward from an OOB access to find the chain of computations that produced the bad value.

```bash
python trace_crash.py openvm-trace.txt
python trace_crash.py openvm-trace.txt --lines 100
```

## Address Space Offset

OpenVM addresses are offset from womir addresses by a fixed amount (the linear memory start). The default offset is `0x2120`. Use `--offset` to override if your binary has a different layout.

## Typical Debugging Workflow

1. **Generate traces** from both backends for the same input
2. **`verify_alu.py`** -- rule out basic arithmetic errors
3. **`compare_calls.py`** -- find which function call diverges first
4. **`find_first_divergent_write.py`** -- find which memory write diverges first
5. **`find_first_divergence.py`** -- get byte-level precision on the divergence
6. **`trace_crash.py`** -- if there's a crash, trace backward from the crash point
