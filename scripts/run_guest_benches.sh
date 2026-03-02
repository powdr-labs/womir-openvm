#!/bin/bash

# Script to collect some numbers from our OpenVM WOMIR vs RISC-V guest examples.
# Mostly for CI usage, but can be easily modified for manual tests.

# NOTE: The script expects the python environment to be set up with the required
# dependencies. Should be run from the project root, will create a `results`
# directory.

set -e

SCRIPT_PATH=$(realpath "${BASH_SOURCE[0]}")
SCRIPTS_DIR=$(dirname "$SCRIPT_PATH")

# Convert space-separated values into --args flags for the CLI.
# E.g. "0 0 10 155" -> "--args 0 --args 0 --args 10 --args 155"
make_args_flags() {
    local flags=()
    for val in $1; do
        flags+=(--args "$val")
    done
    echo "${flags[@]}"
}

run_bench_wasm() {
    local guest="$1"
    local input="$2"
    local run_name="$3"

    echo ""
    echo "==== ${run_name} ===="
    echo ""

    mkdir -p "${run_name}"

    local args_flags
    args_flags=($(make_args_flags "$input"))

    cargo run -r -- prove --recursion "$guest" "main" "${args_flags[@]}" --metrics "${run_name}/metrics.json" &> "${run_name}/log.txt"

    python3 "$SCRIPTS_DIR"/plot_trace_cells.py -o "${run_name}"/trace_cells.png "${run_name}"/metrics.json > "${run_name}"/trace_cells.txt
}

run_bench_riscv() {
    local guest="$1"
    local input="$2"
    local run_name="$3"

    echo ""
    echo "==== ${run_name} ===="
    echo ""

    mkdir -p "${run_name}"

    local args_flags
    args_flags=($(make_args_flags "$input"))

    cargo run -r -- prove-riscv "$guest" "${args_flags[@]}" --metrics "${run_name}/metrics.json" &> "${run_name}/log.txt"

    python3 "$SCRIPTS_DIR"/plot_trace_cells.py -o "${run_name}"/trace_cells.png "${run_name}"/metrics.json > "${run_name}"/trace_cells.txt

    # Clean up some files that we don't want to push.
    rm -f debug.pil
}

### Keccak 10 iterations
dir="results/keccak_10"
input_riscv="10"
input_wasm="0 0 10 155"

ROOT_DIR=$(pwd)

mkdir -p "$dir"
pushd "$dir"

run_bench_riscv "$ROOT_DIR/sample-programs/keccak_with_inputs" "$input_riscv" "riscv"
run_bench_wasm "$ROOT_DIR/sample-programs/keccak_with_inputs/target/wasm32-unknown-unknown/release/keccak_with_inputs.wasm" "$input_wasm" "womir"

python3 "$SCRIPTS_DIR"/basic_metrics.py --csv */metrics.json > basic_metrics.csv
python3 "$SCRIPTS_DIR"/womir_vs_riscv.py "womir/metrics.json" "riscv/metrics.json" > womir_vs_riscv.txt
popd

### Keccak 500 iterations
dir="results/keccak_500"
input_riscv="500"
input_wasm="0 0 500 57"

mkdir -p "$dir"
pushd "$dir"

run_bench_riscv "$ROOT_DIR/sample-programs/keccak_with_inputs" "$input_riscv" "riscv"
run_bench_wasm "$ROOT_DIR/sample-programs/keccak_with_inputs/target/wasm32-unknown-unknown/release/keccak_with_inputs.wasm" "$input_wasm" "womir"

python3 "$SCRIPTS_DIR"/basic_metrics.py --csv */metrics.json > basic_metrics.csv
python3 "$SCRIPTS_DIR"/womir_vs_riscv.py "womir/metrics.json" "riscv/metrics.json" > womir_vs_riscv.txt
popd
