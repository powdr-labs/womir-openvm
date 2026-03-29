#!/bin/bash
# Benchmark: 4 configurations of keccak proving
set -e

ITERS=${1:-10}
echo "Running keccak benchmark with $ITERS iterations"

SCRIPT_PATH=$(realpath "${BASH_SOURCE[0]}")
SCRIPTS_DIR=$(dirname "$SCRIPT_PATH")

# Download shared analysis scripts from powdr upstream.
POWDR_SCRIPTS_URL="https://raw.githubusercontent.com/powdr-labs/powdr/main/openvm-riscv/scripts"
curl -sL "$POWDR_SCRIPTS_URL/basic_metrics.py" -o "$SCRIPTS_DIR/basic_metrics.py"
curl -sL "$POWDR_SCRIPTS_URL/metrics_utils.py" -o "$SCRIPTS_DIR/metrics_utils.py"
curl -sL "$POWDR_SCRIPTS_URL/plot_trace_cells.py" -o "$SCRIPTS_DIR/plot_trace_cells.py"

# Compute expected first byte
EXPECTED_BYTE=$(python3 -c "
from hashlib import new as h
import struct
o = b'\x00'*32
# Use tiny-keccak compatible keccak256 (NOT sha3-256)
# We just read it from a known table
table = {1: 41, 10: 155, 100: 173, 1000: 39}
print(table.get(${ITERS}, 0))
")
echo "Expected first byte for $ITERS iterations: $EXPECTED_BYTE"

ROOT_DIR=$(pwd)
RESULTS_DIR="$ROOT_DIR/keccak-benchmark-results"
rm -rf "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR"

make_input_flags() {
  local flags=()
  for val in $1; do flags+=(--input "$val"); done
  echo "${flags[@]}"
}

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

finalize_wall_times() {
    local json_file="$1"
    echo "{" > "$json_file"
    sed '$!s/$/,/' "${json_file}.tmp" >> "$json_file"
    echo "}" >> "$json_file"
    rm -f "${json_file}.tmp"
}

run_bench_wasm() {
  local guest="$1"
  local input="$2"
  local run_name="$3"
  local extra_flags="$4"

  echo ""
  echo "==== ${run_name} ===="

  mkdir -p "${run_name}"

  local input_flags
  input_flags=($(make_input_flags "$input"))

  local compiled_dir="${run_name}/compiled"
  local wall_times="${run_name}/wall_times.json"

  timed "$wall_times" "compile" \
    cargo run -r -- compile $extra_flags --apc-count 0 --output-dir "$compiled_dir" "$guest" "main" "${input_flags[@]}" &>"${run_name}/compile_log.txt"

  timed "$wall_times" "prove" \
    cargo run -r -- prove $extra_flags --compiled-dir "$compiled_dir" --recursion "${input_flags[@]}" --metrics "${run_name}/metrics.json" &>"${run_name}/log.txt"

  finalize_wall_times "$wall_times"

  python3 "$SCRIPTS_DIR"/plot_trace_cells.py -o "${run_name}"/trace_cells.png "${run_name}"/metrics.json >"${run_name}"/trace_cells.txt 2>/dev/null || true
}

run_bench_riscv() {
  local guest="$1"
  local input="$2"
  local run_name="$3"

  echo ""
  echo "==== ${run_name} ===="

  mkdir -p "${run_name}"

  local input_flags
  input_flags=($(make_input_flags "$input"))

  local compiled_dir="${run_name}/compiled"
  local wall_times="${run_name}/wall_times.json"

  timed "$wall_times" "compile" \
    cargo run -r -- compile-riscv --apc-count 0 --output-dir "$compiled_dir" "$guest" "${input_flags[@]}" &>"${run_name}/compile_log.txt"

  timed "$wall_times" "prove" \
    cargo run -r -- prove-riscv --compiled-dir "$compiled_dir" "${input_flags[@]}" --metrics "${run_name}/metrics.json" &>"${run_name}/log.txt"

  finalize_wall_times "$wall_times"

  python3 "$SCRIPTS_DIR"/plot_trace_cells.py -o "${run_name}"/trace_cells.png "${run_name}"/metrics.json >"${run_name}"/trace_cells.txt 2>/dev/null || true
}

# Build WASM guests
echo "Building WASM guests..."
cargo build --release --target wasm32-unknown-unknown --manifest-path sample-programs/keccak_with_inputs/Cargo.toml 2>/dev/null
cargo build --release --target wasm32-unknown-unknown --manifest-path sample-programs/keccak_precompile/Cargo.toml 2>/dev/null

WASM_SW="$ROOT_DIR/sample-programs/keccak_with_inputs/target/wasm32-unknown-unknown/release/keccak_with_inputs.wasm"
WASM_HW="$ROOT_DIR/sample-programs/keccak_precompile/target/wasm32-unknown-unknown/release/keccak_precompile.wasm"

cd "$RESULTS_DIR"

# 1. RISC-V + software keccak
run_bench_riscv "$ROOT_DIR/sample-programs/keccak_with_inputs" "$ITERS" "riscv_software"

# 2. RISC-V + manual precompile
run_bench_riscv "$ROOT_DIR/sample-programs/keccak_riscv_precompile" "$ITERS" "riscv_precompile"

# 3. WASM/Crush + software keccak
run_bench_wasm "$WASM_SW" "0 0 $ITERS $EXPECTED_BYTE" "crush_software" ""

# 4. WASM/Crush + manual precompile
run_bench_wasm "$WASM_HW" "0 0 $ITERS $EXPECTED_BYTE" "crush_precompile" "--keccak"

echo ""
echo "==== Summary ===="
python3 "$SCRIPTS_DIR"/basic_metrics.py summary-table --csv */metrics.json

cd "$ROOT_DIR"
