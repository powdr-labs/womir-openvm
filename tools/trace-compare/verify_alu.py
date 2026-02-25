#!/usr/bin/env python3
"""Verify ALU, comparison, shift, and MUL operations in an openvm trace.

Checks that every traced arithmetic/logic operation produces the correct result.
Supports: Add, Sub, Xor, Or, And, EQ, NEQ, SLT, SLTU, SHIFT, MUL.

Usage:
    python verify_alu.py <openvm_trace> [--64bit-only]
"""
import argparse
import re
import sys


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("openvm_trace", help="Path to openvm trace file")
    p.add_argument("--64bit-only", action="store_true", help="Only check 64-bit operations")
    return p.parse_args()


def verify_alu_ops(trace_path, only_64bit):
    alu_ops = {
        'AddOp': lambda a, b: (a + b) & 0xFFFFFFFFFFFFFFFF,
        'SubOp': lambda a, b: (a - b) & 0xFFFFFFFFFFFFFFFF,
        'XorOp': lambda a, b: a ^ b,
        'OrOp': lambda a, b: a | b,
        'AndOp': lambda a, b: a & b,
    }
    cmp_ops = ['EQ', 'NEQ', 'SLT', 'SLTU']

    counts = {}
    errors = {}
    for name in list(alu_ops) + cmp_ops + ['SHIFT', 'MUL']:
        counts[name] = 0
        errors[name] = 0

    with open(trace_path) as f:
        for i, line in enumerate(f, 1):
            # ALU ops
            for op_name, fn in alu_ops.items():
                if f' {op_name} a=' in line:
                    m = re.search(
                        rf'{op_name} a=(\d+) b=(\d+) c=(\d+) in1=0x([0-9a-f]+) in2=0x([0-9a-f]+) out=0x([0-9a-f]+)',
                        line)
                    if m:
                        in1 = int(m.group(4), 16)
                        in2 = int(m.group(5), 16)
                        out = int(m.group(6), 16)
                        is_64 = in1 > 0xFFFFFFFF or in2 > 0xFFFFFFFF or out > 0xFFFFFFFF
                        if only_64bit and not is_64:
                            break
                        counts[op_name] += 1
                        expected = fn(in1, in2)
                        if not is_64:
                            expected &= 0xFFFFFFFF
                        if out != expected:
                            errors[op_name] += 1
                            if errors[op_name] <= 3:
                                print(f"  ERROR L{i}: {op_name} in1=0x{in1:x} in2=0x{in2:x} out=0x{out:x} expected=0x{expected:x}")
                    break

            # Comparison ops
            for op in cmp_ops:
                if f' {op} a=' in line:
                    m = re.search(
                        rf'{op} a=\d+ b=\d+ c=\d+ in1=0x([0-9a-f]+) in2=0x([0-9a-f]+) out=(true|false)',
                        line)
                    if m:
                        in1 = int(m.group(1), 16)
                        in2 = int(m.group(2), 16)
                        out = m.group(3) == 'true'
                        counts[op] += 1
                        if op == 'EQ':
                            expected = in1 == in2
                        elif op == 'NEQ':
                            expected = in1 != in2
                        elif op == 'SLT':
                            if in1 > 0xFFFFFFFF or in2 > 0xFFFFFFFF:
                                s1 = in1 if in1 < (1 << 63) else in1 - (1 << 64)
                                s2 = in2 if in2 < (1 << 63) else in2 - (1 << 64)
                            else:
                                s1 = in1 if in1 < (1 << 31) else in1 - (1 << 32)
                                s2 = in2 if in2 < (1 << 31) else in2 - (1 << 32)
                            expected = s1 < s2
                        elif op == 'SLTU':
                            expected = in1 < in2
                        if out != expected:
                            errors[op] += 1
                            if errors[op] <= 3:
                                print(f"  ERROR L{i}: {op} in1=0x{in1:x} in2=0x{in2:x} out={out} expected={expected}")
                    break

            # Shift ops
            m = re.search(r'SHIFT a=(\d+) b=(\d+) c=(\d+) in=0x([0-9a-f]+) shamt=(\d+) out=0x([0-9a-f]+)', line)
            if m:
                inp = int(m.group(4), 16)
                shamt = int(m.group(5))
                out = int(m.group(6), 16)
                is_64 = inp > 0xFFFFFFFF or out > 0xFFFFFFFF or shamt > 31
                if only_64bit and not is_64:
                    continue
                counts['SHIFT'] += 1
                mask = 0xFFFFFFFFFFFFFFFF if is_64 else 0xFFFFFFFF
                bits = 64 if is_64 else 32
                expected_sll = (inp << shamt) & mask
                expected_srl = inp >> shamt
                if inp & (1 << (bits - 1)):
                    expected_sra = (inp >> shamt) | (~((1 << (bits - shamt)) - 1) & mask) if shamt < bits else mask
                else:
                    expected_sra = inp >> shamt
                if out not in (expected_sll, expected_srl, expected_sra):
                    errors['SHIFT'] += 1
                    if errors['SHIFT'] <= 3:
                        print(f"  ERROR L{i}: SHIFT in=0x{inp:x} shamt={shamt} out=0x{out:x}")
                        print(f"    expected SLL=0x{expected_sll:x} SRL=0x{expected_srl:x} SRA=0x{expected_sra:x}")

            # MUL ops
            m = re.search(r'MUL a=(\d+) b=(\d+) c=(\d+) in1=0x([0-9a-f]+) in2=0x([0-9a-f]+) out=0x([0-9a-f]+)', line)
            if m:
                in1 = int(m.group(4), 16)
                in2 = int(m.group(5), 16)
                out = int(m.group(6), 16)
                is_64 = in1 > 0xFFFFFFFF or in2 > 0xFFFFFFFF or out > 0xFFFFFFFF
                if only_64bit and not is_64:
                    continue
                counts['MUL'] += 1
                mask = 0xFFFFFFFFFFFFFFFF if is_64 else 0xFFFFFFFF
                expected = (in1 * in2) & mask
                if out != expected:
                    errors['MUL'] += 1
                    if errors['MUL'] <= 3:
                        print(f"  ERROR L{i}: MUL 0x{in1:x} * 0x{in2:x} = 0x{out:x} (expected 0x{expected:x})")

    # Summary
    print("\nOperation verification summary:")
    all_ok = True
    for name in list(alu_ops) + cmp_ops + ['SHIFT', 'MUL']:
        if counts[name] == 0:
            continue
        status = "OK" if errors[name] == 0 else f"{errors[name]} ERRORS"
        print(f"  {name:8s}: {counts[name]:8d} ops, {status}")
        if errors[name] > 0:
            all_ok = False

    if all_ok:
        total = sum(counts.values())
        print(f"\nAll {total} operations verified correct.")
    else:
        total_errors = sum(errors.values())
        print(f"\n{total_errors} total errors found!")
        sys.exit(1)


def main():
    args = parse_args()
    verify_alu_ops(args.openvm_trace, args.__dict__['64bit_only'])


if __name__ == "__main__":
    main()
