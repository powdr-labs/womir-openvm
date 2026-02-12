---
name: resolve-dep-paths
description: Resolves `<dep>/path` references to local filesystem paths. Use this whenever you encounter a `<dep>/path` reference in documentation and need to read the file.
autoInvoke: true
---

# Resolve Dependency Paths

When you encounter a `<dep>/<path>` reference (e.g. `<openvm>/crates/vm/src/arch/integration_api.rs`), resolve it by running the script in this skill's directory:

```bash
# List all dependency roots:
.claude/skills/resolve-dep-paths/resolve.sh

# Get a single dependency root:
.claude/skills/resolve-dep-paths/resolve.sh openvm
```

Then use the Read tool with the resolved path. For example, if the script prints `/Users/.../.cargo/git/checkouts/openvm-HASH/REV`, read `<openvm>/crates/vm/src/arch/integration_api.rs` as `/Users/.../.cargo/git/checkouts/openvm-HASH/REV/crates/vm/src/arch/integration_api.rs`.
