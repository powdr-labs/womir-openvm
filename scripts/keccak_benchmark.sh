#!/bin/bash
# Benchmark: 4 configurations of keccak proving
# 1. RISC-V + software keccak
# 2. RISC-V + manual precompile
# 3. WASM/Crush + software keccak
# 4. WASM/Crush + manual precompile

set -e

ITERS=${1:-10}
echo "Running keccak benchmark with $ITERS iterations"

# Compute expected first byte for the given iteration count
EXPECTED_BYTE=$(cd /tmp && mkdir -p keccak_expected && cd keccak_expected && \
  cat > Cargo.toml << 'TOML'
[package]
name = "keccak_expected"
version = "0.1.0"
edition = "2021"
[dependencies]
tiny-keccak = { version = "2.0.2", features = ["keccak"] }
TOML
  mkdir -p src && cat > src/main.rs << RUST
use tiny_keccak::{Hasher, Keccak};
fn main() {
    let n: u32 = ${ITERS};
    let mut output = [0u8; 32];
    for _ in 0..n {
        let mut hasher = Keccak::v256();
        hasher.update(&output);
        hasher.finalize(&mut output);
    }
    print!("{}", output[0]);
}
RUST
  cargo run -q 2>/dev/null)

echo "Expected first byte for $ITERS iterations: $EXPECTED_BYTE"

ROOT_DIR=$(pwd)
RESULTS_DIR="$ROOT_DIR/keccak-benchmark-results"
mkdir -p "$RESULTS_DIR"

# Build WASM guests
echo ""
echo "=== Building WASM guests ==="
cargo build --release --target wasm32-unknown-unknown --manifest-path sample-programs/keccak_with_inputs/Cargo.toml
cargo build --release --target wasm32-unknown-unknown --manifest-path sample-programs/keccak_precompile/Cargo.toml

WASM_SW="sample-programs/keccak_with_inputs/target/wasm32-unknown-unknown/release/keccak_with_inputs.wasm"
WASM_HW="sample-programs/keccak_precompile/target/wasm32-unknown-unknown/release/keccak_precompile.wasm"

# Helper: run a timed prove and extract metrics
run_prove() {
  local name="$1"
  shift
  local dir="$RESULTS_DIR/$name"
  mkdir -p "$dir"
  echo ""
  echo "=== $name ==="
  local start end elapsed
  start=$(date +%s.%N)
  "$@" --metrics "$dir/metrics.json" >"$dir/log.txt" 2>&1
  local exit_code=$?
  end=$(date +%s.%N)
  elapsed=$(echo "$end - $start" | bc)
  echo "  wall time: ${elapsed}s (exit: $exit_code)"
  if [ $exit_code -ne 0 ]; then
    echo "  FAILED - see $dir/log.txt"
    tail -5 "$dir/log.txt"
    return $exit_code
  fi
  # Extract key metrics
  if [ -f "$dir/metrics.json" ]; then
    echo "  metrics saved to $dir/metrics.json"
  fi
  return 0
}

# 1. RISC-V + software keccak
run_prove "riscv_software" \
  cargo run -r -- prove-riscv \
    "$ROOT_DIR/sample-programs/keccak_with_inputs" \
    --input "$ITERS"

# 2. RISC-V + manual precompile
run_prove "riscv_precompile" \
  cargo run -r -- prove-riscv \
    "$ROOT_DIR/sample-programs/keccak_riscv_precompile" \
    --input "$ITERS"

# 3. WASM/Crush + software keccak
run_prove "crush_software" \
  cargo run -r -- prove \
    "$WASM_SW" main \
    --input 0 --input 0 --input "$ITERS" --input "$EXPECTED_BYTE"

# 4. WASM/Crush + manual precompile
run_prove "crush_precompile" \
  cargo run -r -- prove --keccak \
    "$WASM_HW" main \
    --input 0 --input 0 --input "$ITERS" --input "$EXPECTED_BYTE"

echo ""
echo "=== Summary ==="
echo ""
for name in riscv_software riscv_precompile crush_software crush_precompile; do
  f="$RESULTS_DIR/$name/metrics.json"
  if [ -f "$f" ]; then
    main_cells=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('main_trace_cells', d.get('total_main_cells', 'N/A')))" 2>/dev/null || echo "N/A")
    echo "  $name: main_cells=$main_cells"
  else
    echo "  $name: no metrics"
  fi
done
