#!/usr/bin/env python3

import os
import sys
import argparse

from plot_trace_cells import compute_cells_by_air

# Fixed tables
range_tuple_checker = "RangeTupleCheckerAir<2>"
var_range_checker = "VariableRangeCheckerAir"
bitwise_lookup = "BitwiseOperationLookupAir<8>"
program = "ProgramAir"

# Common
poseidon = "Poseidon2PeripheryAir<BabyBearParameters>, 1>"
memory_merkle = "MemoryMerkleAir<8>"
access_adapter = "AccessAdapterAir<8>"
persistent_boundary = "PersistentBoundaryAir<8>"
vm_connector = "VmConnectorAir"
phantom = "PhantomAir"

# WOMIR specific
womir_alu_32 = "VmAirWrapper<WomBaseAluAdapterAir<2, 4, 4>, BaseAluCoreAir<4, 8>"
womir_alu_64 = "VmAirWrapper<WomBaseAluAdapterAir<4, 8, 8>, BaseAluCoreAir<8, 8>"
womir_mul_32 = "VmAirWrapper<WomBaseAluAdapterAir<2, 4, 4>, MultiplicationCoreAir<4, 8>"
womir_mul_64 = "VmAirWrapper<WomBaseAluAdapterAir<4, 8, 8>, MultiplicationCoreAir<8, 8>"
womir_shift_32 = "VmAirWrapper<WomBaseAluAdapterAir<2, 4, 4>, ShiftCoreAir<4, 8>"
womir_shift_64 = "VmAirWrapper<WomBaseAluAdapterAir<4, 8, 8>, ShiftCoreAir<8, 8>"
womir_loadstore = "VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4>"
womir_copy_into_frame = "VmAirWrapper<CopyIntoFrameAdapterAirWom, CopyIntoFrameCoreAir>"
womir_less_than = "VmAirWrapper<WomBaseAluAdapterAir<2, 4, 4>, LessThanCoreAir<4, 4, 8>"
womir_jaaf = "VmAirWrapper<JaafAdapterAirWom, JaafCoreAir>"
womir_allocate_frame = "VmAirWrapper<AllocateFrameAdapterAirWom, AllocateFrameCoreAir>"
womir_jump = "VmAirWrapper<JumpAdapterAirWom, JumpCoreAir>"
womir_consts = "VmAirWrapper<ConstsAdapterAirWom, ConstsCoreAir>"
womir_eq_32 = "VmAirWrapper<WomBaseAluAdapterAir<2, 4, 4>, EqCoreAir<4, 4, 8>"
womir_eq_64 = "VmAirWrapper<WomBaseAluAdapterAir<4, 8, 4>, EqCoreAir<8, 4, 8>"
womir_hintstore = "HintStoreAir"

# RISC-V specific
riscv_loadstore = "VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4>"
riscv_alu = "VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8>"
riscv_shift = "VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8>"
riscv_branch_eq = "VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4>"
riscv_branch_less_than = "VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8>"
riscv_cond_rd_write = "VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir>"
riscv_jalr = "VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir>"
riscv_less_than = "VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8>"
riscv_auipc = "VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir>"
riscv_mul = "VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8>"
riscv_hintstore = "Rv32HintStoreAir"

def compare(cat, a, b):
    print(f"\nComparing {cat}:")
    print(f"  WOMIR: {a} cells")
    print(f"  RISC-V: {b} cells")
    sa = sum([int(x) for x in a])
    sb = sum([int(x) for x in b])
    if sa < sb:
        d = sb - sa
        if sa > 0:
            print(f"  WOMIR uses fewer cells by {d} ({sb/sa:.2f}x)")
        else:
            print(f"  WOMIR uses fewer cells by {d} (100%)")
    elif sa > sb:
        d = sa - sb
        if sb > 0:
            print(f"  RISC-V uses fewer cells by {d} ({sa/sb:.2f}x)")
        else:
            print(f"  RISC-V uses fewer cells by {d} (100%)")
    else:
        print("  Both use the same number of cells")

def main(womir_metrics_path, riscv_metrics_path):
    print(f"Analysing WOMIR...")
    w = compute_cells_by_air(womir_metrics_path).to_dict()
    w_total = sum(int(v) for v in w.values())

    print(f"\nAnalysing RISC-V...")
    r = compute_cells_by_air(riscv_metrics_path).to_dict()
    r_total = sum(int(v) for v in r.values())

    compare("Total WOMIR vs RISC-V", [w_total], [r_total])

    compare("ALU WOMIR (32, 64) vs RISC-V 32", [w[womir_alu_32], w[womir_alu_64]], [r[riscv_alu]])
    compare("Shift WOMIR (32, 64) vs RISC-V 32", [w[womir_shift_32], w[womir_shift_64]], [r[riscv_shift]])
    compare("Mul WOMIR (32, 64) vs RISC-V 32", [w[womir_mul_32], w[womir_mul_64]], [r[riscv_mul]])
    compare("LoadStore", [w[womir_loadstore]], [r[riscv_loadstore]])
    compare("Comparison WOMIR (lt, eq_32, eq_64) vs RISC-V lt", [w[womir_less_than], w[womir_eq_32], w[womir_eq_64]], [r[riscv_less_than]])
    compare("Branch/Jump WOMIR (jaaf, jump) vs RISC-V (beq, blt, jalr, auipc)", [w[womir_jaaf], w[womir_jump]], [r[riscv_branch_eq], r[riscv_branch_less_than], r[riscv_jalr], r[riscv_auipc]])
    compare("Comparison + Branch/Jump (often combined) WOMIR (lt, eq_32, eq_64, jaaf, jump) vs RISC-V (lt, beq, blt, jalr, auipc)", [w[womir_less_than], w[womir_eq_32], w[womir_eq_64], w[womir_jaaf], w[womir_jump]], [r[riscv_less_than], r[riscv_branch_eq], r[riscv_branch_less_than], r[riscv_jalr], r[riscv_auipc]])
    compare("HintStore", [w[womir_hintstore]], [r[riscv_hintstore]])
    compare("Frame WOMIR (copy, allocate)", [w[womir_copy_into_frame], w[womir_allocate_frame]], [])
    compare("Consts WOMIR", [w[womir_consts]], [])
    compare("Lookups (range tuple checker, var range checker, bitwise)", [w[range_tuple_checker], w[var_range_checker], w[bitwise_lookup]], [r[range_tuple_checker], r[var_range_checker], r[bitwise_lookup]])
    compare("Continuations (poseidon, merkle, access, boundary, vm_connector)", [w[poseidon], w[memory_merkle], w[access_adapter], w[persistent_boundary], w[vm_connector]], [r[poseidon], r[memory_merkle], r[access_adapter], r[persistent_boundary], r[vm_connector]])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare trace cells between WOMIR and RISC-V in OpenVM proofs.")
    parser.add_argument("womir_metrics_path", help="Path to the WOMIR metrics.json file")
    parser.add_argument("riscv_metrics_path", help="Path to the RISCV metrics.json file")
    args = parser.parse_args()

    main(args.womir_metrics_path, args.riscv_metrics_path)
