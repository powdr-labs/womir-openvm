#!/usr/bin/env python3
"""Trace backward from a crash point in an openvm trace.

Shows the last N detailed trace lines before the crash, filtering out
bare PC-stepping lines. Useful for understanding the chain of computations
leading to an out-of-bounds access or other failure.

Usage:
    python trace_crash.py <openvm_trace> [--lines 50] [--max-line 0]
"""
import argparse
import re


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("openvm_trace", help="Path to openvm trace file")
    p.add_argument("--lines", type=int, default=50, help="Number of detailed lines to show (default: 50)")
    p.add_argument("--max-line", type=int, default=0, help="Stop at this line (0=read entire file)")
    return p.parse_args()


DETAIL_KEYWORDS = [
    'LOADW', 'STOREW', 'STOREH', 'STOREB', 'LOADH', 'LOADB',
    'CALL', 'RET', 'AddOp', 'SubOp', 'XorOp', 'OrOp', 'AndOp',
    'SLTU', 'SLT', 'EQ', 'NEQ', 'JIF', 'JIFZ',
    'CONST32', 'MUL', 'SHIFT', 'DIVREM',
    'fp=',
]


def main():
    args = parse_args()
    detailed_lines = []

    with open(args.openvm_trace) as f:
        for i, line in enumerate(f, 1):
            if args.max_line and i > args.max_line:
                break
            line = line.rstrip()
            if any(kw in line for kw in DETAIL_KEYWORDS):
                detailed_lines.append((i, line))

    print(f"=== Last {args.lines} detailed trace lines ===")
    for i, line in detailed_lines[-args.lines:]:
        print(f"  L{i}: {line[:200]}")


if __name__ == "__main__":
    main()
