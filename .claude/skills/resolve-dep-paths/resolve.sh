#!/usr/bin/env bash
# Resolves git dependency names to local cargo checkout paths.
# Usage: ./resolve.sh [dep_name]
#   No args: prints all dep roots (e.g. "openvm: /path/to/checkout")
#   With arg: prints just the path for that dep (e.g. "/path/to/checkout")

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Navigate to the workspace root (three levels up from .claude/skills/resolve-dep-paths/)
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

output=$(cargo metadata --manifest-path "$WORKSPACE_ROOT/Cargo.toml" --format-version 1 | python3 -c "
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
")

if [ $# -eq 0 ]; then
    echo "$output"
else
    path=$(echo "$output" | grep "^$1: " | cut -d' ' -f2-)
    if [ -z "$path" ]; then
        echo "Error: dependency '$1' not found" >&2
        echo "Available dependencies:" >&2
        echo "$output" >&2
        exit 1
    fi
    echo "$path"
fi
