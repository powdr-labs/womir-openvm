#!/usr/bin/env python3
"""Find the first divergent memory write between openvm and womir.

For each heap address, compares the sequence of STOREW writes from each backend.
The first mismatch (earliest in openvm time) reveals the root cause.
Lighter-weight than find_first_divergence.py (word-level only, no byte replay).

Usage:
    python find_first_divergent_write.py <openvm_trace> <womir_memwrites> [options]

Options:
    --offset N        Address offset: openvm_ptr - womir_addr (default: 0x2120)
    --heap-start N    Womir heap start address (default: 0x11695c)
    --max-line N      Stop reading openvm trace at this line (default: unlimited)
"""
import argparse
import re
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("openvm_trace", help="Path to openvm trace file")
    p.add_argument("womir_memwrites", help="Path to womir memory writes file")
    p.add_argument("--offset", type=lambda x: int(x, 0), default=0x2120)
    p.add_argument("--heap-start", type=lambda x: int(x, 0), default=0x11695c)
    p.add_argument("--max-line", type=int, default=0, help="Stop at this openvm line (0=unlimited)")
    return p.parse_args()


def main():
    args = parse_args()

    print("=== Building OpenVM per-address write map ===")
    ovm_writes = defaultdict(list)
    ovm_count = 0
    with open(args.openvm_trace) as f:
        for i, line in enumerate(f, 1):
            if args.max_line and i > args.max_line:
                break
            if 'STOREW e=2' in line:
                m = re.search(r'STOREW e=2 .* ptr=0x([0-9a-f]+) shift=0 val=0x([0-9a-f]+)', line)
                if m:
                    ptr = int(m.group(1), 16)
                    val = int(m.group(2), 16)
                    womir_addr = ptr - args.offset
                    if womir_addr >= args.heap_start:
                        ovm_writes[womir_addr].append((i, val))
                        ovm_count += 1
    print(f"  {ovm_count} heap writes to {len(ovm_writes)} unique addresses")

    print("=== Building Womir per-address write map ===")
    wom_writes = defaultdict(list)
    wom_count = 0
    with open(args.womir_memwrites) as f:
        for idx, line in enumerate(f):
            if line.startswith('MW '):
                m = re.match(r'MW addr=0x([0-9a-f]+) val=0x([0-9a-f]+)', line)
                if m:
                    addr = int(m.group(1), 16)
                    val = int(m.group(2), 16)
                    if addr >= args.heap_start:
                        wom_writes[addr].append((idx, val))
                        wom_count += 1
    print(f"  {wom_count} heap writes to {len(wom_writes)} unique addresses")

    common = set(ovm_writes.keys()) & set(wom_writes.keys())
    print(f"\nCommon heap addresses: {len(common)}")

    divergent = []
    for addr in sorted(common):
        oseq = ovm_writes[addr]
        wseq = wom_writes[addr]
        for idx in range(min(len(oseq), len(wseq))):
            oln, oval = oseq[idx]
            wln, wval = wseq[idx]
            if oval != wval:
                divergent.append((addr, idx, oln, oval, wln, wval))
                break

    divergent.sort(key=lambda x: x[2])  # sort by openvm line
    print(f"Addresses with STOREW divergence: {len(divergent)}")
    print(f"\nFirst 30 divergent STOREWs (sorted by openvm line):")
    for addr, idx, oln, oval, wln, wval in divergent[:30]:
        print(f"  addr=0x{addr:06x} write#{idx} ovm_L{oln}=0x{oval:08x} wom_MW{wln}=0x{wval:08x}")

    ovm_only = set(ovm_writes.keys()) - set(wom_writes.keys())
    wom_only = set(wom_writes.keys()) - set(ovm_writes.keys())
    print(f"\nAddresses only in openvm: {len(ovm_only)}")
    print(f"Addresses only in womir: {len(wom_only)}")


if __name__ == "__main__":
    main()
