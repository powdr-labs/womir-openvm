#!/usr/bin/env python3
"""Compare MUL results between openvm and womir by matching (in1, in2) input pairs.

Since builtin functions interleave MUL operations in openvm but not in womir,
sequential comparison doesn't work. Instead, this groups MUL results by their
input pairs and checks that both backends produce the same outputs.

Usage:
    python compare_muls.py <openvm_trace> <womir_trace>
"""
import argparse
import re
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("openvm_trace", help="Path to openvm trace file")
    p.add_argument("womir_trace", help="Path to womir trace file")
    return p.parse_args()


def main():
    args = parse_args()

    # Extract openvm 64-bit MULs
    ovm_muls = defaultdict(list)
    with open(args.openvm_trace) as f:
        for i, line in enumerate(f, 1):
            m = re.search(
                r'MUL a=(\d+) b=(\d+) c=(\d+) in1=0x([0-9a-f]+) in2=0x([0-9a-f]+) out=0x([0-9a-f]+)',
                line)
            if m:
                in1 = int(m.group(4), 16)
                in2 = int(m.group(5), 16)
                out = int(m.group(6), 16)
                if in1 > 0xFFFFFFFF or in2 > 0xFFFFFFFF or out > 0xFFFFFFFF:
                    ovm_muls[(in1, in2)].append((i, out))

    # Extract womir I64Mul
    wom_muls = defaultdict(list)
    with open(args.womir_trace) as f:
        for i, line in enumerate(f, 1):
            if 'I64Mul' in line:
                reads = re.findall(r'r:(\d+)=(\d+)', line)
                writes = re.findall(r'w:(\d+)=(\d+)', line)
                if len(reads) >= 4 and len(writes) >= 2:
                    r_vals = {int(r): int(v) for r, v in reads}
                    w_vals = {int(r): int(v) for r, v in writes}
                    r_regs = [int(r) for r, _ in reads]
                    w_regs = [int(r) for r, _ in writes]
                    a = (r_vals[r_regs[1]] << 32) | r_vals[r_regs[0]]
                    b = (r_vals[r_regs[3]] << 32) | r_vals[r_regs[2]]
                    out = (w_vals[w_regs[1]] << 32) | w_vals[w_regs[0]]
                    wom_muls[(a, b)].append((i, out))

    common = set(ovm_muls.keys()) & set(wom_muls.keys())
    print(f"OpenVM: {sum(len(v) for v in ovm_muls.values())} 64-bit MULs, {len(ovm_muls)} unique input pairs")
    print(f"Womir:  {sum(len(v) for v in wom_muls.values())} I64Muls, {len(wom_muls)} unique input pairs")
    print(f"Common input pairs: {len(common)}")

    mismatches = []
    for pair in common:
        ovm_outs = set(out for _, out in ovm_muls[pair])
        wom_outs = set(out for _, out in wom_muls[pair])
        if ovm_outs != wom_outs:
            mismatches.append((pair, ovm_outs, wom_outs))

    print(f"Input pairs with different outputs: {len(mismatches)}")
    for (in1, in2), ovm_outs, wom_outs in mismatches[:10]:
        print(f"  in1=0x{in1:016x} in2=0x{in2:016x}")
        print(f"    ovm: {[f'0x{o:016x}' for o in ovm_outs]}")
        print(f"    wom: {[f'0x{o:016x}' for o in wom_outs]}")

    # Verify openvm results
    print(f"\nVerifying all OpenVM 64-bit MUL results...")
    wrong = 0
    for (in1, in2), results in ovm_muls.items():
        expected = (in1 * in2) & 0xFFFFFFFFFFFFFFFF
        for line, out in results:
            if out != expected:
                wrong += 1
                if wrong <= 5:
                    print(f"  WRONG L{line}: 0x{in1:x} * 0x{in2:x} = 0x{out:x} (expected 0x{expected:x})")
    print(f"  {wrong} incorrect out of {sum(len(v) for v in ovm_muls.values())}")


if __name__ == "__main__":
    main()
