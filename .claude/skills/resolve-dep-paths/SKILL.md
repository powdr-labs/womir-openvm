---
name: resolve-dep-paths
description: Resolves `<dep>/path` references to local filesystem paths. Use this whenever you encounter a `<dep>/path` reference in documentation and need to read the file.
autoInvoke: true
---

# Resolve Dependency Paths

When you encounter a `<dep>/<path>` reference (e.g. `<openvm>/crates/vm/src/arch/integration_api.rs`), resolve it to a local path by running:

```bash
cargo metadata --format-version 1 | python3 -c "
import json, sys, os
meta = json.load(sys.stdin)
seen = {}
for pkg in meta['packages']:
    p = pkg['manifest_path']
    if '/.cargo/git/checkouts/' in p:
        parts = p.split('/.cargo/git/checkouts/')[1]
        repo_checkout = '/'.join(parts.split('/')[:2])
        repo_name = parts.split('-')[0]
        if repo_name not in seen:
            seen[repo_name] = os.path.expanduser('~') + '/.cargo/git/checkouts/' + repo_checkout
for n, r in sorted(seen.items()):
    print(f'{n}: {r}')
"
```

This prints lines like:
```
openvm: /Users/.../.cargo/git/checkouts/openvm-HASH/REV
womir: /Users/.../.cargo/git/checkouts/womir-HASH/REV
```

Then replace `<openvm>` with the printed openvm root, etc. For example:
- `<openvm>/crates/vm/src/arch/integration_api.rs` becomes `/Users/.../.cargo/git/checkouts/openvm-HASH/REV/crates/vm/src/arch/integration_api.rs`

Use the Read tool to read the resolved path.
