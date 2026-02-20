# Codex Web test execution plan (`wat2wasm` / `wast2json` constraints)

## Problem summary

Today, `integration/build.rs` always shells out to `wat2wasm` while compiling the `womir-openvm-integration` package.
Because build scripts run before any tests are compiled, **all tests are blocked** when `wat2wasm` is unavailable (including non-WASM tests such as `isolated_tests.rs`).

Additionally, the WAST test path in `integration/src/main.rs` shells out to `wast2json`, which is also not available in Codex Web sandbox environments.

## Goals

1. Codex Web can run `isolated_tests.rs` without requiring host-installed WABT binaries.
2. Codex Web can run the full suite, including WAST tests, without requiring host-installed WABT binaries.

## Plan overview

### Phase 1 — Unblock `isolated_tests.rs` quickly

**Intent:** remove hard dependency on `wat2wasm` from package build.

1. Replace build-time `wat2wasm` calls in `integration/build.rs` with pure-Rust compilation using the `wat` crate:
   - Add build dependency: `wat = "*"` (pin exact version during implementation).
   - In `compile_wat_to_wasm`, read `builtin_src/*.wat`, parse with `wat::parse_str` (or `parse_bytes`), and write output bytes to `${OUT_DIR}/*.wasm`.
2. Keep existing `clang` flow unchanged for now (current blocker is specifically WABT).
3. Add an explicit error message if WAT parsing fails that points to source file + parse error.
4. Validate in Codex-like environment:
   - `cargo test -p womir-openvm-integration --tests isolated_tests`

**Why this works:** `build.rs` no longer needs external `wat2wasm`, so tests that do not use WAST runner can compile and execute.

---

### Phase 2 — Make WAST tests sandbox-compatible

**Intent:** remove runtime dependency on `wast2json`.

Two implementation paths:

#### Preferred path (fully native Rust)

1. Replace `wast2json` subprocess path in `wast_tests` with Rust parsing/execution:
   - Parse `.wast` using the `wast` crate AST.
   - Translate each directive (`module`, `assert_return`, `assert_trap`, etc.) into current test-case structures used by `run_wasm_test`.
2. Reuse existing execution/assertion logic where possible; only swap input front-end.
3. Add support incrementally for directives present in `wasm_tests/*.wast` (the repo subset), and error clearly on unsupported directives.
4. Add focused unit tests for parser-to-testcase conversion.

#### Transitional fallback (if delivery needs to be staged)

1. Add a repo script (run in CI/dev machines that have WABT) to pre-generate JSON test vectors from `.wast`.
2. Commit generated JSON fixtures into the repo (or publish as release artifacts fetched in CI).
3. In test runtime, prefer checked-in JSON fixtures; only call `wast2json` when fixture is missing and binary is present.

**Recommendation:** implement preferred path to completely remove external tool dependency in Codex Web.

---

### Phase 3 — Improve test topology so unrelated tests stay runnable

Even after removing WABT usage, separating test domains reduces future coupling.

1. Move reusable integration logic out of `src/main.rs` into `src/lib.rs`.
2. Keep CLI entrypoint minimal in `main.rs`.
3. Split tests into dedicated files under `integration/tests/`:
   - `tests/isolated_tests.rs`
   - `tests/wast_tests.rs`
   - other logical groups as needed
4. Ensure heavy/runtime-gated suites are tagged and filterable (e.g., `#[ignore]` or feature gates) without blocking compile of other tests.

This step is optional for solving the immediate blocker, but strongly recommended for long-term maintainability.

---

## Validation matrix

After implementation, validate with the following matrix:

1. **No WABT binaries present (`wat2wasm`, `wast2json` absent):**
   - `cargo test -p womir-openvm-integration --tests isolated_tests` ✅
   - `cargo test -p womir-openvm-integration` (includes WAST) ✅
2. **Developer workstation with WABT installed:**
   - same commands above ✅ (ensures no regression)
3. **CI:**
   - run full test command in job that does not install WABT tools.

## Suggested rollout order

1. Land Phase 1 first (smallest change, immediate unblock for isolated tests).
2. Land Phase 2 next (enables full test suite in Codex Web).
3. Land Phase 3 opportunistically as refactor hardening.

## Risk notes

- WAST semantics can be broad; constrain scope to directives used by this repo first, then expand.
- Keep failures explicit and deterministic when unsupported directives are encountered.
- Preserve current behavior parity by running old/new runners side-by-side during migration (where WABT is available).
