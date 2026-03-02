#!/usr/bin/env python3

"""Render a Markdown summary from benchmark result directories.

Usage: python3 render_summary.py results/keccak_* > summary.md
"""

import argparse
import csv
import os
import sys


def render_bench(bench_dir: str) -> str:
    lines = []
    name = os.path.basename(bench_dir)
    lines.append(f"## {name}\n")

    # Basic metrics
    csv_path = os.path.join(bench_dir, "basic_metrics.csv")
    if os.path.isfile(csv_path):
        lines.append("### Basic Metrics\n")
        lines.append("```")
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                for k, v in row.items():
                    lines.append(f"{k}: {v}")
                lines.append("")
        lines.append("```\n")

    # WOMIR vs RISC-V comparison
    cmp_path = os.path.join(bench_dir, "womir_vs_riscv.txt")
    if os.path.isfile(cmp_path):
        lines.append("### WOMIR vs RISC-V Comparison\n")
        lines.append("```")
        lines.append(open(cmp_path).read().rstrip())
        lines.append("```\n")

    # Trace cell breakdowns (one per run: womir/, riscv/)
    for run_dir in sorted(os.listdir(bench_dir)):
        txt_path = os.path.join(bench_dir, run_dir, "trace_cells.txt")
        if os.path.isfile(txt_path):
            lines.append(f"### Trace Cells - {run_dir}\n")
            lines.append("```")
            lines.append(open(txt_path).read().rstrip())
            lines.append("```\n")

    lines.append("_Trace cell plots (PNG) are available in the uploaded artifacts._\n")
    lines.append("---\n")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Render a Markdown summary from benchmark result directories.")
    parser.add_argument("bench_dirs", nargs="+", help="Benchmark result directories (e.g. results/keccak_*)")
    args = parser.parse_args()

    for bench_dir in sorted(args.bench_dirs):
        print(render_bench(bench_dir))


if __name__ == "__main__":
    main()
