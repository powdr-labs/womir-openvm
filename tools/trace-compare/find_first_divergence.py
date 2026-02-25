#!/usr/bin/env python3
"""Build byte-level memory maps from both openvm and womir traces, find first divergent store.

Replays all STOREW/STOREH/STOREB from the openvm trace and all MW from the womir trace
into byte-level memory images, then compares the final states to find the first divergent
range. For that range, shows which stores from each backend wrote to those addresses.

Usage:
    python find_first_divergence.py <openvm_trace> <womir_memwrites> [--offset 0x2120] [--mem-size 0x300000]

Arguments:
    openvm_trace      Path to openvm trace file (containing STOREW/STOREH/STOREB lines)
    womir_memwrites   Path to womir memory writes file (containing MW lines)

Options:
    --offset N        Address offset between openvm and womir (openvm_ptr - womir_addr), default 0x2120
    --mem-size N      Memory image size in bytes, default 0x300000 (3MB)
"""
import argparse
import re
import struct


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("openvm_trace", help="Path to openvm trace file")
    p.add_argument("womir_memwrites", help="Path to womir memory writes file (MW lines)")
    p.add_argument("--offset", type=lambda x: int(x, 0), default=0x2120,
                   help="Address offset: openvm_ptr - womir_addr (default: 0x2120)")
    p.add_argument("--mem-size", type=lambda x: int(x, 0), default=0x300000,
                   help="Memory image size in bytes (default: 0x300000)")
    return p.parse_args()


def replay_openvm(trace_path, offset, mem_size):
    mem = bytearray(mem_size)
    stores = []
    count = 0
    with open(trace_path) as f:
        for i, line in enumerate(f, 1):
            if 'STOREW e=2' in line:
                m = re.search(r'STOREW e=2 .* ptr=0x([0-9a-f]+) shift=0 val=0x([0-9a-f]+)', line)
                if m:
                    ptr = int(m.group(1), 16)
                    val = int(m.group(2), 16)
                    addr = ptr - offset
                    if 0 <= addr < mem_size - 3:
                        old = struct.unpack_from('<I', mem, addr)[0]
                        struct.pack_into('<I', mem, addr, val & 0xFFFFFFFF)
                        stores.append((i, addr, 4, old, val & 0xFFFFFFFF))
                        count += 1
            elif 'STOREH e=2' in line:
                m = re.search(r'STOREH e=2 .* ptr=0x([0-9a-f]+) shift=(\d+) val=0x([0-9a-f]+)', line)
                if m:
                    ptr = int(m.group(1), 16)
                    shift = int(m.group(2))
                    val = int(m.group(3), 16) & 0xFFFF
                    addr = ptr - offset + shift
                    if 0 <= addr < mem_size - 1:
                        old = struct.unpack_from('<H', mem, addr)[0]
                        struct.pack_into('<H', mem, addr, val)
                        stores.append((i, addr, 2, old, val))
                        count += 1
            elif 'STOREB e=2' in line:
                m = re.search(r'STOREB e=2 .* ptr=0x([0-9a-f]+) shift=(\d+) val=0x([0-9a-f]+)', line)
                if m:
                    ptr = int(m.group(1), 16)
                    shift = int(m.group(2))
                    val = int(m.group(3), 16) & 0xFF
                    addr = ptr - offset + shift
                    if 0 <= addr < mem_size:
                        old = mem[addr]
                        mem[addr] = val
                        stores.append((i, addr, 1, old, val))
                        count += 1
    return mem, stores, count


def replay_womir(memwrites_path, mem_size):
    mem = bytearray(mem_size)
    stores = []
    count = 0
    line_idx = 0
    with open(memwrites_path) as f:
        for line in f:
            if line.startswith('MW '):
                m = re.match(r'MW addr=0x([0-9a-f]+) val=0x([0-9a-f]+)', line)
                if m:
                    addr = int(m.group(1), 16)
                    val = int(m.group(2), 16)
                    if 0 <= addr < mem_size - 3:
                        old = struct.unpack_from('<I', mem, addr)[0]
                        struct.pack_into('<I', mem, addr, val & 0xFFFFFFFF)
                        stores.append((line_idx, addr, old, val & 0xFFFFFFFF))
                        count += 1
                line_idx += 1
    return mem, stores, count


def main():
    args = parse_args()

    print("=== Replaying OpenVM stores ===")
    ovm_mem, ovm_stores, ovm_count = replay_openvm(args.openvm_trace, args.offset, args.mem_size)
    print(f"  {ovm_count} stores replayed")

    print("=== Replaying Womir stores ===")
    wom_mem, wom_stores, wom_count = replay_womir(args.womir_memwrites, args.mem_size)
    print(f"  {wom_count} stores replayed")

    # Compare final memory states
    print("\n=== Comparing final memory states ===")
    divergent_ranges = []
    in_divergent = False
    div_start = None
    for addr in range(min(len(ovm_mem), len(wom_mem))):
        if ovm_mem[addr] != wom_mem[addr]:
            if not in_divergent:
                div_start = addr
                in_divergent = True
        else:
            if in_divergent:
                divergent_ranges.append((div_start, addr))
                in_divergent = False
    if in_divergent:
        divergent_ranges.append((div_start, len(ovm_mem)))

    print(f"  {len(divergent_ranges)} divergent ranges")
    for start, end in divergent_ranges[:30]:
        size = end - start
        ovm_bytes = ovm_mem[start:end][:16]
        wom_bytes = wom_mem[start:end][:16]
        print(f"  0x{start:06x}-0x{end:06x} ({size} bytes)  ovm={ovm_bytes.hex()}  wom={wom_bytes.hex()}")

    # For the first divergent byte, find what wrote it
    if divergent_ranges:
        first_div_addr = divergent_ranges[0][0]
        first_div_end = min(divergent_ranges[0][1], first_div_addr + 16)
        print(f"\n=== First divergent range: 0x{first_div_addr:06x}-0x{first_div_end:06x} ===")

        print(f"\nOpenVM stores touching 0x{first_div_addr:06x}-0x{first_div_end:06x}:")
        count = 0
        for line, addr, size, old, new in ovm_stores:
            store_end = addr + size
            if store_end > first_div_addr and addr < first_div_end:
                print(f"  L{line}: addr=0x{addr:06x} size={size} 0x{old:x} -> 0x{new:x}")
                count += 1
                if count >= 40:
                    break

        print(f"\nWomir stores touching 0x{first_div_addr:06x}-0x{first_div_end:06x}:")
        count = 0
        for idx, addr, old, new in wom_stores:
            store_end = addr + 4
            if store_end > first_div_addr and addr < first_div_end:
                print(f"  MW#{idx}: addr=0x{addr:06x} 0x{old:x} -> 0x{new:x}")
                count += 1
                if count >= 40:
                    break


if __name__ == "__main__":
    main()
