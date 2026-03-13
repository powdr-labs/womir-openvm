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
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if rows:
            headers = list(rows[0].keys())
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join("---" for _ in headers) + " |")
            for row in rows:
                lines.append("| " + " | ".join(row[h] for h in headers) + " |")
            lines.append("")

    # CRUSH vs RISC-V comparison
    cmp_path = os.path.join(bench_dir, "crush_vs_riscv.txt")
    if os.path.isfile(cmp_path):
        lines.append("### CRUSH vs RISC-V Comparison\n")
        lines.append("```")
        lines.append(open(cmp_path).read().rstrip())
        lines.append("```\n")

    # Trace cell breakdowns (one per run: crush/, riscv/)
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


def group_label(bench_dir: str) -> str:
    """Derive a human-readable group label from the directory name prefix."""
    name = os.path.basename(bench_dir.rstrip("/"))
    prefix = name.split("_")[0]
    labels = {
        "keccak": "Keccak",
        "reth": "Reth (eth-block)",
    }
    return labels.get(prefix, prefix.capitalize())


def main():
    parser = argparse.ArgumentParser(description="Render a Markdown summary from benchmark result directories.")
    parser.add_argument("bench_dirs", nargs="+", help="Benchmark result directories (e.g. results/keccak_*)")
    args = parser.parse_args()

    current_group = None
    for bench_dir in sorted(args.bench_dirs):
        group = group_label(bench_dir)
        if group != current_group:
            print(f"# {group}\n")
            current_group = group
        print(render_bench(bench_dir))


if __name__ == "__main__":
    main()
