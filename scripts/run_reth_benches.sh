#!/bin/bash

# Script to benchmark Reth (eth-block) via crush and RISC-V.
# Mostly for CI usage, but can be easily modified for manual tests.

# NOTE: The script expects the python environment to be set up with the required
# dependencies. Should be run from the project root, will create a `results`
# directory.

set -e

# Parse flags
CUDA_FLAGS=""
BLOCK=""
APC_COUNT="0"
RETH_BENCH_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda) CUDA_FLAGS="--features cuda"; shift ;;
        --block-number) BLOCK="$2"; shift 2 ;;
        --apc-count) APC_COUNT="$2"; shift 2 ;;
        --reth-bench-dir) RETH_BENCH_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$BLOCK" ]]; then
    echo "Usage: $0 [--cuda] --block-number <number> [--apc-count <apcs>] [--reth-bench-dir <path>]"
    exit 1
fi

CUDA_FLAG=""
if [[ -n "$CUDA_FLAGS" ]]; then
    CUDA_FLAG="--cuda"
fi

SCRIPT_PATH=$(realpath "${BASH_SOURCE[0]}")
SCRIPTS_DIR=$(dirname "$SCRIPT_PATH")

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
    echo "\"${label}\": ${elapsed}" >> "${json_file}.tmp"
    return $exit_code
}

# Finalize wall_times JSON from collected entries.
finalize_wall_times() {
    local json_file="$1"
    echo "{" > "$json_file"
    sed '$!s/$/,/' "${json_file}.tmp" >> "$json_file"
    echo "}" >> "$json_file"
    rm -f "${json_file}.tmp"
}

# Download shared analysis scripts from powdr upstream.
POWDR_SCRIPTS_URL="https://raw.githubusercontent.com/powdr-labs/powdr/main/openvm-riscv/scripts"
curl -sL "$POWDR_SCRIPTS_URL/basic_metrics.py" -o "$SCRIPTS_DIR/basic_metrics.py"
curl -sL "$POWDR_SCRIPTS_URL/metrics_utils.py" -o "$SCRIPTS_DIR/metrics_utils.py"
curl -sL "$POWDR_SCRIPTS_URL/plot_trace_cells.py" -o "$SCRIPTS_DIR/plot_trace_cells.py"

### Reth eth-block
dir="results/reth_${BLOCK}"

ROOT_DIR=$(pwd)

COMPILED_DIR="$ROOT_DIR/.cache/crush-compiled-reth-${BLOCK}"

# Compile step (not included in benchmark metrics)
echo ""
echo "==== crush Compile ===="
echo ""

mkdir -p "$dir"
pushd "$dir"

### crush benchmark
run_name="crush_apc_${APC_COUNT}"
mkdir -p "${run_name}"
wall_times="${run_name}/wall_times.json"

timed "$wall_times" "compile" \
    cargo run -r $CUDA_FLAGS -- compile \
    "$ROOT_DIR/sample-programs/eth-block/openvm-client-eth.wasm" "main" \
    --input 0 --input 0 --input "file:$ROOT_DIR/sample-programs/eth-block/${BLOCK}.bin" \
    --apc-count "$APC_COUNT" --apc-candidates-dir "${run_name}" --output-dir "$COMPILED_DIR" &> "${run_name}/compile_log.txt"

echo ""
echo "==== ${run_name} ===="
echo ""

# Prove step (metrics captured here)
timed "$wall_times" "prove" \
    cargo run -r $CUDA_FLAGS -- prove \
    --compiled-dir "$COMPILED_DIR" \
    --input 0 --input 0 --input "file:$ROOT_DIR/sample-programs/eth-block/${BLOCK}.bin" \
    --metrics "${run_name}/metrics.json" \
    --recursion \
    &> "${run_name}/log.txt"

finalize_wall_times "$wall_times"

python3 "$SCRIPTS_DIR"/plot_trace_cells.py -o "${run_name}"/trace_cells.png "${run_name}"/metrics.json > "${run_name}"/trace_cells.txt

### RISC-V benchmark (via openvm-reth-benchmark)
if [[ -n "$RETH_BENCH_DIR" ]]; then
    run_name="riscv_apc_${APC_COUNT}"
    echo ""
    echo "==== ${run_name} ===="
    echo ""

    mkdir -p "${run_name}"
    riscv_wall_times="${run_name}/wall_times.json"

    # Compile (keygen) first so it doesn't appear in prove metrics
    pushd "$RETH_BENCH_DIR"
    timed "$OLDPWD/$riscv_wall_times" "compile" \
        ./run.sh --no-precompiles $CUDA_FLAG --apcs "$APC_COUNT" --mode compile --block-number "$BLOCK" &> "$OLDPWD/${run_name}/compile_log.txt"
    # Prove
    timed "$OLDPWD/$riscv_wall_times" "prove" \
        ./run.sh --no-precompiles $CUDA_FLAG --mode prove-stark --block-number "$BLOCK" &> "$OLDPWD/${run_name}/log.txt"
    cp metrics.json "$OLDPWD/${run_name}/metrics.json"
    popd

    finalize_wall_times "$riscv_wall_times"

    python3 "$SCRIPTS_DIR"/plot_trace_cells.py -o "${run_name}"/trace_cells.png "${run_name}"/metrics.json > "${run_name}"/trace_cells.txt

    python3 "$SCRIPTS_DIR"/crush_vs_riscv.py "crush/metrics.json" "riscv/metrics.json" > crush_vs_riscv.txt
fi

python3 "$SCRIPTS_DIR"/basic_metrics.py summary-table --csv */metrics.json > basic_metrics.csv
popd
