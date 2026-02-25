#!/usr/bin/env python3
"""Compare call/ret event sequences between openvm and womir traces.

Filters out builtin function calls in openvm (function index >= threshold).
Useful for narrowing down which function call diverges between backends.

Usage:
    python compare_calls.py <openvm_trace> <womir_trace> [--builtin-threshold 1565]
"""
import argparse
import re


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("openvm_trace", help="Path to openvm trace file")
    p.add_argument("womir_trace", help="Path to womir trace file")
    p.add_argument("--builtin-threshold", type=int, default=1565,
                   help="Function index >= this are builtins to skip (default: 1565)")
    return p.parse_args()


def is_builtin(func_name, threshold):
    m = re.match(r'__func_(\d+)', func_name)
    return m and int(m.group(1)) >= threshold


def main():
    args = parse_args()

    # Build openvm function PC-to-name map
    func_pc_map = {}
    with open(args.openvm_trace) as f:
        for line in f:
            m = re.match(r'FUNC (__func_\d+): pc=(\d+)\.\.(\d+)', line)
            if m:
                func_pc_map[int(m.group(2))] = m.group(1)
    print(f"Loaded {len(func_pc_map)} function PC ranges")

    # Extract openvm events, skipping builtin calls and their returns
    ovm_events = []
    entry_skipped = False
    builtin_depth = 0
    with open(args.openvm_trace) as f:
        for i, line in enumerate(f, 1):
            if ' CALL ' in line:
                m = re.search(r'new_pc=(\d+)', line)
                if m:
                    target_pc = int(m.group(1))
                    func_name = func_pc_map.get(target_pc, f"?pc={target_pc}")
                    if not entry_skipped:
                        entry_skipped = True
                        continue
                    if builtin_depth > 0 or is_builtin(func_name, args.builtin_threshold):
                        builtin_depth += 1
                        continue
                    ovm_events.append(('CALL', func_name, i))
            elif ' RET ' in line:
                if not entry_skipped:
                    continue
                if builtin_depth > 0:
                    builtin_depth -= 1
                    continue
                ovm_events.append(('RET', '', i))
    print(f"OpenVM: {len(ovm_events)} call/ret events (after filtering)")

    # Extract womir events
    wom_events = []
    with open(args.womir_trace) as f:
        for i, line in enumerate(f, 1):
            if 'RwCall __func_' in line:
                m = re.search(r'RwCall (__func_\d+)', line)
                if m:
                    wom_events.append(('CALL', m.group(1), i))
            elif ':     Return' in line:
                wom_events.append(('RET', '', i))
    print(f"Womir: {len(wom_events)} call/ret events")

    # Compare
    first_diff = None
    for idx in range(min(len(ovm_events), len(wom_events))):
        otype, ofunc, oline = ovm_events[idx]
        wtype, wfunc, wline = wom_events[idx]
        if otype != wtype or (otype == 'CALL' and ofunc != wfunc):
            first_diff = idx
            break

    if first_diff is not None:
        print(f"\nFirst divergence at event {first_diff}:")
        start = max(0, first_diff - 5)
        for idx in range(start, min(first_diff + 5, len(ovm_events), len(wom_events))):
            otype, ofunc, oline = ovm_events[idx]
            wtype, wfunc, wline = wom_events[idx]
            match_str = "OK" if (otype == wtype and (otype != 'CALL' or ofunc == wfunc)) else "DIFFER"
            print(f"  [{idx}] ovm: {otype} {ofunc:20s} L{oline}  |  wom: {wtype} {wfunc:20s} L{wline}  [{match_str}]")
    else:
        shorter = min(len(ovm_events), len(wom_events))
        print(f"\nAll {shorter} events match!")
        if len(ovm_events) != len(wom_events):
            print(f"  (ovm has {len(ovm_events)}, wom has {len(wom_events)})")


if __name__ == "__main__":
    main()
