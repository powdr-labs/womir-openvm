#!/bin/bash

# Script to collect some numbers from our OpenVM WOMIR vs RISC-V guest examples.
# Mostly for CI usage, but can be easily modified for manual tests.

# NOTE: The script expects the python environment to be set up with the required
# dependencies. Should be run from the project root, will create a `results`
# directory.

set -e

# Parse --cuda flag
CUDA_FLAGS=""
if [[ "$1" == "--cuda" ]]; then
    CUDA_FLAGS="--features cuda"
    shift
fi

SCRIPT_PATH=$(realpath "${BASH_SOURCE[0]}")
SCRIPTS_DIR=$(dirname "$SCRIPT_PATH")

# Download shared analysis scripts from powdr upstream.
POWDR_SCRIPTS_URL="https://raw.githubusercontent.com/powdr-labs/powdr/main/openvm/scripts"
curl -sL "$POWDR_SCRIPTS_URL/basic_metrics.py" -o "$SCRIPTS_DIR/basic_metrics.py"
curl -sL "$POWDR_SCRIPTS_URL/metrics_utils.py" -o "$SCRIPTS_DIR/metrics_utils.py"
curl -sL "$POWDR_SCRIPTS_URL/plot_trace_cells.py" -o "$SCRIPTS_DIR/plot_trace_cells.py"

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

    cargo run -r $CUDA_FLAGS -- prove --recursion "$guest" "main" "${args_flags[@]}" --metrics "${run_name}/metrics.json" &> "${run_name}/log.txt"

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

    cargo run -r $CUDA_FLAGS -- prove-riscv "$guest" "${args_flags[@]}" --metrics "${run_name}/metrics.json" &> "${run_name}/log.txt"

    python3 "$SCRIPTS_DIR"/plot_trace_cells.py -o "${run_name}"/trace_cells.png "${run_name}"/metrics.json > "${run_name}"/trace_cells.txt

    # Clean up some files that we don't want to push.
    rm -f debug.pil
}

### Keccak 1000 iterations
dir="results/keccak_1000"
# The RISC-V guest takes as input the number of iterations.
# It returns the first byte of the result as a public.
input_riscv="1000"
# The WASM guest takes as input the main arguments [argc, argv], the input number of iterations,
# and the first byte of the result to be asserted inside the guest program.
input_wasm="0 0 1000 39"

ROOT_DIR=$(pwd)

mkdir -p "$dir"
pushd "$dir"

run_bench_riscv "$ROOT_DIR/sample-programs/keccak_with_inputs" "$input_riscv" "riscv"
run_bench_wasm "$ROOT_DIR/sample-programs/keccak_with_inputs/target/wasm32-unknown-unknown/release/keccak_with_inputs.wasm" "$input_wasm" "womir"

python3 "$SCRIPTS_DIR"/basic_metrics.py summary-table --csv */metrics.json > basic_metrics.csv
python3 "$SCRIPTS_DIR"/womir_vs_riscv.py "womir/metrics.json" "riscv/metrics.json" > womir_vs_riscv.txt
popd
