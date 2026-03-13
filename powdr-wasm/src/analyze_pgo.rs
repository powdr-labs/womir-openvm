//! PGO analysis: collect execution profiles and basic block statistics.

use std::collections::HashMap;

use powdr_autoprecompiles::{
    blocks::BasicBlock,
    execution_profile::ExecutionProfile,
};
use powdr_openvm::customize_exe::Instr;
use openvm_stark_sdk::p3_baby_bear::BabyBear;

/// Print comprehensive PGO statistics for a set of basic blocks and an execution profile.
pub fn print_pgo_analysis<ISA>(
    isa_name: &str,
    blocks: &[BasicBlock<Instr<BabyBear, ISA>>],
    profile: &ExecutionProfile,
) {
    // Index blocks by start_pc
    let block_by_pc: HashMap<u64, &BasicBlock<Instr<BabyBear, ISA>>> =
        blocks.iter().map(|b| (b.start_pc, b)).collect();

    // For each block, compute execution count (how many times the block's start_pc was hit)
    // Since PGO records every PC, a block's execution count = count of its start_pc in pc_count
    let mut block_stats: Vec<BlockStat> = blocks
        .iter()
        .map(|b| {
            let exec_count = profile.pc_count.get(&b.start_pc).copied().unwrap_or(0);
            BlockStat {
                start_pc: b.start_pc,
                num_instructions: b.instructions.len(),
                exec_count,
            }
        })
        .collect();

    // Total instructions in the program (static)
    let total_static_instructions: usize = blocks.iter().map(|b| b.instructions.len()).sum();
    let total_static_blocks = blocks.len();

    // Total dynamic instructions executed
    let total_dynamic_instructions: u64 = profile.pc_list.len() as u64;

    // Compute dynamic instruction count per block (exec_count * block_len)
    let total_dynamic_instructions_from_blocks: u64 = block_stats
        .iter()
        .map(|b| b.exec_count as u64 * b.num_instructions as u64)
        .sum();

    // How many blocks were actually executed
    let executed_blocks: Vec<&BlockStat> = block_stats.iter().filter(|b| b.exec_count > 0).collect();
    let num_executed_blocks = executed_blocks.len();

    println!("\n{}", "=".repeat(60));
    println!("  PGO Analysis: {isa_name}");
    println!("{}\n", "=".repeat(60));

    println!("=== Overview ===");
    println!("  Static basic blocks:       {total_static_blocks:>10}");
    println!("  Static instructions:       {total_static_instructions:>10}");
    println!("  Executed basic blocks:     {num_executed_blocks:>10}");
    println!("  Total PCs executed:        {total_dynamic_instructions:>10}");
    println!("  Dynamic insns (from BBs):  {total_dynamic_instructions_from_blocks:>10}");
    println!();

    // === Block size distribution (all static blocks) ===
    println!("=== Block Size Distribution (all static blocks) ===");
    print_size_histogram(&block_stats);

    // === Block size distribution (executed blocks only) ===
    println!("=== Block Size Distribution (executed blocks only) ===");
    let executed_stats: Vec<BlockStat> = block_stats.iter().filter(|b| b.exec_count > 0).cloned().collect();
    print_size_histogram(&executed_stats);

    // === Top blocks by execution count ===
    block_stats.sort_by(|a, b| b.exec_count.cmp(&a.exec_count));
    println!("=== Top 50 Blocks by Execution Count ===");
    println!("  {:>12} {:>8} {:>12} {:>15}", "start_pc", "len", "exec_count", "dynamic_insns");
    for b in block_stats.iter().take(50) {
        let dynamic = b.exec_count as u64 * b.num_instructions as u64;
        println!("  {:>12} {:>8} {:>12} {:>15}", b.start_pc, b.num_instructions, b.exec_count, dynamic);
    }
    println!();

    // === Top blocks by dynamic instruction contribution ===
    block_stats.sort_by(|a, b| {
        let da = a.exec_count as u64 * a.num_instructions as u64;
        let db = b.exec_count as u64 * b.num_instructions as u64;
        db.cmp(&da)
    });
    println!("=== Top 50 Blocks by Dynamic Instruction Contribution ===");
    println!("  {:>12} {:>8} {:>12} {:>15} {:>8}", "start_pc", "len", "exec_count", "dynamic_insns", "% total");
    let mut cumulative_pct = 0.0f64;
    for b in block_stats.iter().take(50) {
        let dynamic = b.exec_count as u64 * b.num_instructions as u64;
        let pct = if total_dynamic_instructions_from_blocks > 0 {
            dynamic as f64 / total_dynamic_instructions_from_blocks as f64 * 100.0
        } else {
            0.0
        };
        cumulative_pct += pct;
        println!("  {:>12} {:>8} {:>12} {:>15} {:>7.2}%  (cum: {:.1}%)", b.start_pc, b.num_instructions, b.exec_count, dynamic, pct, cumulative_pct);
    }
    println!();

    // === Top blocks by instruction count (the "widest" blocks) ===
    block_stats.sort_by(|a, b| b.num_instructions.cmp(&a.num_instructions));
    println!("=== Top 50 Widest Blocks (by instruction count) ===");
    println!("  {:>12} {:>8} {:>12} {:>15}", "start_pc", "len", "exec_count", "dynamic_insns");
    for b in block_stats.iter().take(50) {
        let dynamic = b.exec_count as u64 * b.num_instructions as u64;
        println!("  {:>12} {:>8} {:>12} {:>15}", b.start_pc, b.num_instructions, b.exec_count, dynamic);
    }
    println!();

    // === Summary statistics for executed blocks ===
    if !executed_stats.is_empty() {
        let sizes: Vec<usize> = executed_stats.iter().map(|b| b.num_instructions).collect();
        let counts: Vec<u32> = executed_stats.iter().map(|b| b.exec_count).collect();

        let min_size = *sizes.iter().min().unwrap();
        let max_size = *sizes.iter().max().unwrap();
        let avg_size = sizes.iter().sum::<usize>() as f64 / sizes.len() as f64;
        let median_size = {
            let mut sorted = sizes.clone();
            sorted.sort();
            sorted[sorted.len() / 2]
        };

        let min_count = *counts.iter().min().unwrap();
        let max_count = *counts.iter().max().unwrap();
        let avg_count = counts.iter().map(|c| *c as f64).sum::<f64>() / counts.len() as f64;
        let median_count = {
            let mut sorted = counts.clone();
            sorted.sort();
            sorted[sorted.len() / 2]
        };

        // Weighted average block size (weighted by execution count)
        let weighted_avg_size = if total_dynamic_instructions_from_blocks > 0 {
            executed_stats.iter()
                .map(|b| b.num_instructions as f64 * b.exec_count as f64)
                .sum::<f64>()
                / executed_stats.iter().map(|b| b.exec_count as f64).sum::<f64>()
        } else {
            0.0
        };

        println!("=== Summary Statistics (executed blocks) ===");
        println!("  Block size:   min={min_size}, max={max_size}, avg={avg_size:.1}, median={median_size}, weighted_avg={weighted_avg_size:.1}");
        println!("  Exec count:   min={min_count}, max={max_count}, avg={avg_count:.1}, median={median_count}");
        println!();
    }
}

fn print_size_histogram(stats: &[BlockStat]) {
    let buckets = [
        (1, 1, "     1"),
        (2, 5, "   2-5"),
        (6, 10, "  6-10"),
        (11, 50, " 11-50"),
        (51, 100, "51-100"),
        (101, 500, "101-500"),
        (501, 1000, "501-1k"),
        (1001, 5000, " 1k-5k"),
        (5001, usize::MAX, "  5k+"),
    ];
    println!("  {:>10} {:>8} {:>8}", "range", "count", "% total");
    let total = stats.len();
    for (lo, hi, label) in &buckets {
        let count = stats.iter().filter(|b| b.num_instructions >= *lo && b.num_instructions <= *hi).count();
        let pct = if total > 0 { count as f64 / total as f64 * 100.0 } else { 0.0 };
        println!("  {:>10} {:>8} {:>7.1}%", label, count, pct);
    }
    println!();
}

#[derive(Clone)]
struct BlockStat {
    start_pc: u64,
    num_instructions: usize,
    exec_count: u32,
}
