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
womir_alu_32 = "VmAirWrapper<BaseAluAdapterAirDifferentInputsOutputs<4, 1, 1>, BaseAluCoreAir<4, 8>"
womir_alu_64 = "VmAirWrapper<BaseAluAdapterAirDifferentInputsOutputs<8, 2, 2>, BaseAluCoreAir<8, 8>"
womir_mul_32 = "VmAirWrapper<BaseAluAdapterAirDifferentInputsOutputs<4, 1, 1>, MultiplicationCoreAir<4, 8>"
womir_mul_64 = "VmAirWrapper<BaseAluAdapterAirDifferentInputsOutputs<8, 2, 2>, MultiplicationCoreAir<8, 8>"
womir_divrem_32 = "VmAirWrapper<BaseAluAdapterAirDifferentInputsOutputs<4, 1, 1>, DivRemCoreAir<4, 8>"
womir_divrem_64 = "VmAirWrapper<BaseAluAdapterAirDifferentInputsOutputs<8, 2, 2>, DivRemCoreAir<8, 8>"
womir_shift_32 = "VmAirWrapper<BaseAluAdapterAirDifferentInputsOutputs<4, 1, 1>, ShiftCoreAir<4, 8>"
womir_shift_64 = "VmAirWrapper<BaseAluAdapterAirDifferentInputsOutputs<8, 2, 2>, ShiftCoreAir<8, 8>"
womir_loadstore = "VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4>"
womir_loadsignextend = "VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8>"
womir_less_than_32 = "VmAirWrapper<BaseAluAdapterAirDifferentInputsOutputs<4, 1, 1>, LessThanCoreAir<4, 8>"
womir_less_than_64 = "VmAirWrapper<BaseAluAdapterAirDifferentInputsOutputs<8, 2, 1>, LessThanCoreAir<8, 8>"
womir_call = "VmAirWrapper<CallAdapterAir, CallCoreAir>"
womir_jump = "VmAirWrapper<JumpAdapterAir, JumpCoreAir>"
womir_consts = "Const32AdapterAir<4>"
womir_eq_32 = "VmAirWrapper<BaseAluAdapterAirDifferentInputsOutputs<4, 1, 1>, EqCoreAir<4>"
womir_eq_64 = "VmAirWrapper<BaseAluAdapterAirDifferentInputsOutputs<8, 2, 1>, EqCoreAir<8>"
womir_hintstore = "Rv32HintStoreAir"

# RISC-V specific
riscv_loadstore = "VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4>"
riscv_loadsignextend = "VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8>"
riscv_alu = "VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8>"
riscv_shift = "VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8>"
riscv_branch_eq = "VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4>"
riscv_branch_less_than = "VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8>"
riscv_cond_rd_write = "VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir>"
riscv_jalr = "VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir>"
riscv_less_than = "VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8>"
riscv_auipc = "VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir>"
riscv_mul = "VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8>"
riscv_mulh = "VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8>"
riscv_divrem = "VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8>"
riscv_hintstore = "Rv32HintStoreAir"

def get(d, key):
    """Get a value from the dict, returning 0 if key is missing."""
    return d.get(key, 0)

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

    compare("ALU WOMIR (32, 64) vs RISC-V 32", [get(w, womir_alu_32), get(w, womir_alu_64)], [get(r, riscv_alu)])
    compare("Shift WOMIR (32, 64) vs RISC-V 32", [get(w, womir_shift_32), get(w, womir_shift_64)], [get(r, riscv_shift)])
    compare("Mul WOMIR (32, 64) vs RISC-V 32", [get(w, womir_mul_32), get(w, womir_mul_64)], [get(r, riscv_mul)])
    compare("DivRem WOMIR (32, 64) vs RISC-V 32", [get(w, womir_divrem_32), get(w, womir_divrem_64)], [get(r, riscv_divrem)])
    compare("LoadStore", [get(w, womir_loadstore), get(w, womir_loadsignextend)], [get(r, riscv_loadstore), get(r, riscv_loadsignextend)])
    compare("Comparison WOMIR (lt_32, lt_64, eq_32, eq_64) vs RISC-V lt", [get(w, womir_less_than_32), get(w, womir_less_than_64), get(w, womir_eq_32), get(w, womir_eq_64)], [get(r, riscv_less_than)])
    compare("Branch/Jump WOMIR (call, jump) vs RISC-V (beq, blt, jal/lui, jalr, auipc)", [get(w, womir_call), get(w, womir_jump)], [get(r, riscv_branch_eq), get(r, riscv_branch_less_than), get(r, riscv_cond_rd_write), get(r, riscv_jalr), get(r, riscv_auipc)])
    compare("Comparison + Branch/Jump", [get(w, womir_less_than_32), get(w, womir_less_than_64), get(w, womir_eq_32), get(w, womir_eq_64), get(w, womir_call), get(w, womir_jump)], [get(r, riscv_less_than), get(r, riscv_branch_eq), get(r, riscv_branch_less_than), get(r, riscv_cond_rd_write), get(r, riscv_jalr), get(r, riscv_auipc)])
    compare("HintStore", [get(w, womir_hintstore)], [get(r, riscv_hintstore)])
    compare("Consts WOMIR", [get(w, womir_consts)], [])
    compare("Lookups (range tuple checker, var range checker, bitwise)", [get(w, range_tuple_checker), get(w, var_range_checker), get(w, bitwise_lookup)], [get(r, range_tuple_checker), get(r, var_range_checker), get(r, bitwise_lookup)])
    compare("Continuations (poseidon, merkle, access, boundary, vm_connector)", [get(w, poseidon), get(w, memory_merkle), get(w, access_adapter), get(w, persistent_boundary), get(w, vm_connector)], [get(r, poseidon), get(r, memory_merkle), get(r, access_adapter), get(r, persistent_boundary), get(r, vm_connector)])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare trace cells between WOMIR and RISC-V in OpenVM proofs.")
    parser.add_argument("womir_metrics_path", help="Path to the WOMIR metrics.json file")
    parser.add_argument("riscv_metrics_path", help="Path to the RISCV metrics.json file")
    args = parser.parse_args()

    main(args.womir_metrics_path, args.riscv_metrics_path)
