#!/bin/bash

# Script to benchmark Reth (eth-block) via WOMIR and RISC-V.
# Mostly for CI usage, but can be easily modified for manual tests.

# NOTE: The script expects the python environment to be set up with the required
# dependencies. Should be run from the project root, will create a `results`
# directory.

set -e

# Parse flags
CUDA_FLAGS=""
BLOCK=""
RETH_BENCH_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda) CUDA_FLAGS="--features cuda"; shift ;;
        --block-number) BLOCK="$2"; shift 2 ;;
        --reth-bench-dir) RETH_BENCH_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$BLOCK" ]]; then
    echo "Usage: $0 [--cuda] --block-number <number> [--reth-bench-dir <path>]"
    exit 1
fi

CUDA_FLAG=""
if [[ -n "$CUDA_FLAGS" ]]; then
    CUDA_FLAG="--cuda"
fi

SCRIPT_PATH=$(realpath "${BASH_SOURCE[0]}")
SCRIPTS_DIR=$(dirname "$SCRIPT_PATH")

# Download shared analysis scripts from powdr upstream.
POWDR_SCRIPTS_URL="https://raw.githubusercontent.com/powdr-labs/powdr/main/openvm-riscv/scripts"
curl -sL "$POWDR_SCRIPTS_URL/basic_metrics.py" -o "$SCRIPTS_DIR/basic_metrics.py"
curl -sL "$POWDR_SCRIPTS_URL/metrics_utils.py" -o "$SCRIPTS_DIR/metrics_utils.py"
curl -sL "$POWDR_SCRIPTS_URL/plot_trace_cells.py" -o "$SCRIPTS_DIR/plot_trace_cells.py"

### Reth eth-block
dir="results/reth_${BLOCK}"

ROOT_DIR=$(pwd)

CACHE_DIR="$ROOT_DIR/.cache/womir-keys"

# Generate and cache WOMIR proving keys (not included in benchmark metrics)
echo ""
echo "==== WOMIR Keygen ===="
echo ""
cargo run -r $CUDA_FLAGS -- keygen "$CACHE_DIR"

mkdir -p "$dir"
pushd "$dir"

### WOMIR benchmark
run_name="womir"
echo ""
echo "==== ${run_name} ===="
echo ""

mkdir -p "${run_name}"

cargo run -r $CUDA_FLAGS -- prove \
    "$ROOT_DIR/sample-programs/eth-block/openvm-client-eth.wasm" "main" \
    --input 0 --input 0 --input "file:$ROOT_DIR/sample-programs/eth-block/${BLOCK}.bin" \
    --metrics "${run_name}/metrics.json" \
    --recursion \
    --cache-dir "$CACHE_DIR" \
    &> "${run_name}/log.txt"

python3 "$SCRIPTS_DIR"/plot_trace_cells.py -o "${run_name}"/trace_cells.png "${run_name}"/metrics.json > "${run_name}"/trace_cells.txt

### RISC-V benchmark (via openvm-reth-benchmark)
if [[ -n "$RETH_BENCH_DIR" ]]; then
    run_name="riscv"
    echo ""
    echo "==== ${run_name} ===="
    echo ""

    mkdir -p "${run_name}"

    # Compile (keygen) first so it doesn't appear in prove metrics
    pushd "$RETH_BENCH_DIR"
    ./run.sh --no-precompiles $CUDA_FLAG --mode compile --block-number "$BLOCK" &> /dev/null
    # Prove
    ./run.sh --no-precompiles $CUDA_FLAG --mode prove-stark --block-number "$BLOCK" &> "$OLDPWD/${run_name}/log.txt"
    cp metrics.json "$OLDPWD/${run_name}/metrics.json"
    popd

    python3 "$SCRIPTS_DIR"/plot_trace_cells.py -o "${run_name}"/trace_cells.png "${run_name}"/metrics.json > "${run_name}"/trace_cells.txt

    python3 "$SCRIPTS_DIR"/womir_vs_riscv.py "womir/metrics.json" "riscv/metrics.json" > womir_vs_riscv.txt
fi

python3 "$SCRIPTS_DIR"/basic_metrics.py summary-table --csv */metrics.json > basic_metrics.csv
popd
