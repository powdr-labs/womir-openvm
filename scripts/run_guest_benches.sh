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
POWDR_SCRIPTS_URL="https://raw.githubusercontent.com/powdr-labs/powdr/main/openvm-riscv/scripts"
curl -sL "$POWDR_SCRIPTS_URL/basic_metrics.py" -o "$SCRIPTS_DIR/basic_metrics.py"
curl -sL "$POWDR_SCRIPTS_URL/metrics_utils.py" -o "$SCRIPTS_DIR/metrics_utils.py"
curl -sL "$POWDR_SCRIPTS_URL/plot_trace_cells.py" -o "$SCRIPTS_DIR/plot_trace_cells.py"

# Convert space-separated values into --input flags for the CLI.
# E.g. "0 0 10 155" -> "--input 0 --input 0 --input 10 --input 155"
make_input_flags() {
    local flags=()
    for val in $1; do
        flags+=(--input "$val")
    done
    echo "${flags[@]}"
}

# Run a command, capture its wall time, and append {label: seconds} to a JSON file.
# Usage: timed <json_file> <label> <command...>
timed() {
    local json_file="$1"; shift
    local label="$1"; shift
    local start end elapsed
    start=$(date +%s.%N)
    "$@"
    local exit_code=$?
    end=$(date +%s.%N)
    elapsed=$(echo "$end - $start" | bc)
    # Append to JSON file (builds up object entries, finalized later)
    echo "\"${label}\": ${elapsed}" >> "${json_file}.tmp"
    return $exit_code
}

# Finalize wall_times JSON from collected entries.
finalize_wall_times() {
    local json_file="$1"
    echo "{" > "$json_file"
    # Join entries with commas
    sed '$!s/$/,/' "${json_file}.tmp" >> "$json_file"
    echo "}" >> "$json_file"
    rm -f "${json_file}.tmp"
}

# Compile+prove pipeline: compile is excluded from metrics, prove is measured.
run_bench_wasm() {
    local guest="$1"
    local input="$2"
    local run_name="$3"
    local apc_count="$4"

    echo ""
    echo "==== ${run_name} ===="
    echo ""

    mkdir -p "${run_name}"

    local input_flags
    input_flags=($(make_input_flags "$input"))

    local compiled_dir="${run_name}/compiled"
    local wall_times="${run_name}/wall_times.json"

    # Compile step (not included in metrics)
    timed "$wall_times" "compile" \
        cargo run -r $CUDA_FLAGS -- compile --apc-count "$apc_count" --output-dir "$compiled_dir" "$guest" "main" "${input_flags[@]}" &> "${run_name}/compile_log.txt"

    # Prove step (metrics captured here)
    timed "$wall_times" "prove" \
        cargo run -r $CUDA_FLAGS -- prove --compiled-dir "$compiled_dir" --recursion "${input_flags[@]}" --metrics "${run_name}/metrics.json" &> "${run_name}/log.txt"

    finalize_wall_times "$wall_times"

    python3 "$SCRIPTS_DIR"/plot_trace_cells.py -o "${run_name}"/trace_cells.png "${run_name}"/metrics.json > "${run_name}"/trace_cells.txt
}

run_bench_riscv() {
    local guest="$1"
    local input="$2"
    local run_name="$3"
    local apc_count="$4"

    echo ""
    echo "==== ${run_name} ===="
    echo ""

    mkdir -p "${run_name}"

    local input_flags
    input_flags=($(make_input_flags "$input"))

    local compiled_dir="${run_name}/compiled"
    local wall_times="${run_name}/wall_times.json"

    # Compile step (not included in metrics)
    timed "$wall_times" "compile" \
        cargo run -r $CUDA_FLAGS -- compile-riscv --apc-count "$apc_count" --output-dir "$compiled_dir" "$guest" "${input_flags[@]}" &> "${run_name}/compile_log.txt"

    # Prove step (metrics captured here)
    timed "$wall_times" "prove" \
        cargo run -r $CUDA_FLAGS -- prove-riscv --compiled-dir "$compiled_dir" "${input_flags[@]}" --metrics "${run_name}/metrics.json" &> "${run_name}/log.txt"

    finalize_wall_times "$wall_times"

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

run_bench_riscv "$ROOT_DIR/sample-programs/keccak_with_inputs" "$input_riscv" "riscv" "0"
run_bench_riscv "$ROOT_DIR/sample-programs/keccak_with_inputs" "$input_riscv" "riscv_apc_1" "1"
run_bench_riscv "$ROOT_DIR/sample-programs/keccak_with_inputs" "$input_riscv" "riscv_apc_10" "10"
run_bench_wasm "$ROOT_DIR/sample-programs/keccak_with_inputs/target/wasm32-unknown-unknown/release/keccak_with_inputs.wasm" "$input_wasm" "womir" "0"
run_bench_wasm "$ROOT_DIR/sample-programs/keccak_with_inputs/target/wasm32-unknown-unknown/release/keccak_with_inputs.wasm" "$input_wasm" "womir_apc_1" "1"
run_bench_wasm "$ROOT_DIR/sample-programs/keccak_with_inputs/target/wasm32-unknown-unknown/release/keccak_with_inputs.wasm" "$input_wasm" "womir_apc_10" "10"

python3 "$SCRIPTS_DIR"/basic_metrics.py summary-table --csv */metrics.json > basic_metrics.csv
python3 "$SCRIPTS_DIR"/womir_vs_riscv.py "womir/metrics.json" "riscv/metrics.json" > womir_apc_0_vs_riscv.txt
python3 "$SCRIPTS_DIR"/womir_vs_riscv.py "womir_apc_1/metrics.json" "riscv_apc_1/metrics.json" > womir_apc_1_vs_riscv_apc_1.txt
python3 "$SCRIPTS_DIR"/womir_vs_riscv.py "womir_apc_10/metrics.json" "riscv_apc_10/metrics.json" > womir_apc_10_vs_riscv_apc_10.txt
popd
