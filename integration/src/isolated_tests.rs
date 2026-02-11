//! Isolated stage testing infrastructure for WOMIR instructions.
//!
//! This module provides a framework for testing WOMIR instructions
//! through isolated execution stages:
//! - Raw execution (InterpretedInstance::execute_from_state)
//! - Metered execution (InterpretedInstance::execute_metered_from_state)
//! - Preflight (VirtualMachine::execute_preflight)
//! - Proof generation (VirtualMachine::prove)

use openvm_circuit::{
    arch::{VirtualMachine, VmExecutor, VmState, execution_mode::Segment},
    system::memory::online::GuestMemory,
};
use openvm_instructions::{
    exe::VmExe, instruction::Instruction, program::Program, riscv::RV32_REGISTER_AS,
};
use openvm_sdk::{StdIn, config::DEFAULT_APP_LOG_BLOWUP};
use openvm_stark_sdk::{
    config::{FriParameters, baby_bear_poseidon2::BabyBearPoseidon2Engine},
    engine::StarkFriEngine,
};
use womir_circuit::{
    WomirConfig, WomirCpuBuilder, adapters::RV32_REGISTER_NUM_LIMBS, memory_config::FpMemory,
};

use crate::instruction_builder as wom;

type F = openvm_stark_sdk::p3_baby_bear::BabyBear;

/// Memory address space for RAM (heap memory).
const RV32_MEMORY_AS: u32 = openvm_instructions::riscv::RV32_MEMORY_AS;

/// Specification for an isolated instruction test.
/// Defines the start state, program, and expected end state.
#[derive(Clone, Default)]
pub struct IsolatedTestSpec {
    /// The program to execute (should end with halt instruction).
    pub program: Vec<Instruction<F>>,

    /// Initial PC (default: 0).
    pub start_pc: Option<u32>,
    /// Initial FP (default: 0).
    pub start_fp: Option<u32>,
    /// Initial register values: (register_index, value).
    pub start_registers: Vec<(usize, u32)>,
    /// Initial RAM values: (address, value).
    pub start_ram: Vec<(u32, u32)>,

    /// Expected PC after execution (if None, not checked).
    pub expected_pc: Option<u32>,
    /// Expected FP after execution (if None, not checked).
    pub expected_fp: Option<u32>,
    /// Expected register values after execution: (register_index, value).
    pub expected_registers: Vec<(usize, u32)>,
    /// Expected RAM values after execution: (address, value).
    pub expected_ram: Vec<(u32, u32)>,
}

/// Read a register value from memory.
/// Register address = reg_index * RV32_REGISTER_NUM_LIMBS
fn read_register(memory: &GuestMemory, reg: usize) -> u32 {
    let addr = (reg * RV32_REGISTER_NUM_LIMBS) as u32;
    let bytes: [u8; 4] = unsafe { memory.read(RV32_REGISTER_AS, addr) };
    u32::from_le_bytes(bytes)
}

/// Read a 32-bit value from RAM.
fn read_ram(memory: &GuestMemory, addr: u32) -> u32 {
    let bytes: [u8; 4] = unsafe { memory.read(RV32_MEMORY_AS, addr) };
    u32::from_le_bytes(bytes)
}

/// Read FP from memory.
fn read_fp(memory: &GuestMemory) -> u32 {
    memory.fp()
}

/// Build a VmExe from a test specification (program and PC only).
fn build_exe(spec: &IsolatedTestSpec) -> VmExe<F> {
    let program = Program::from_instructions(&spec.program);
    let mut exe = VmExe::new(program);

    // Set start PC if specified
    if let Some(pc) = spec.start_pc {
        exe = exe.with_pc_start(pc);
    }

    exe
}

/// Create initial VmState from spec, exe, and config.
/// Sets up memory with initial register values, RAM values, and FP from the spec.
fn build_initial_state(
    spec: &IsolatedTestSpec,
    exe: &VmExe<F>,
    vm_config: &WomirConfig,
) -> VmState<F> {
    let mut state = VmState::initial(
        &vm_config.system,
        &exe.init_memory,
        exe.pc_start,
        StdIn::default(),
    );

    // Set initial registers
    for &(reg, value) in &spec.start_registers {
        let addr = (reg * RV32_REGISTER_NUM_LIMBS) as u32;
        unsafe {
            state
                .memory
                .write::<u8, 4>(RV32_REGISTER_AS, addr, value.to_le_bytes());
        }
    }

    // Set initial RAM
    for &(addr, value) in &spec.start_ram {
        unsafe {
            state
                .memory
                .write::<u8, 4>(RV32_MEMORY_AS, addr, value.to_le_bytes());
        }
    }

    // Set initial FP
    if let Some(fp) = spec.start_fp {
        state.memory.set_fp(fp);
    }

    state
}

/// Verify the final state matches expected values.
fn verify_state(
    spec: &IsolatedTestSpec,
    final_state: &VmState<F>,
    stage_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Verify expected registers
    for &(reg, expected_value) in &spec.expected_registers {
        let actual_value = read_register(&final_state.memory, reg);
        if actual_value != expected_value {
            return Err(format!(
                "{stage_name}: reg[{reg}] expected {expected_value}, got {actual_value}"
            )
            .into());
        }
    }

    // Verify expected RAM
    for &(addr, expected_value) in &spec.expected_ram {
        let actual_value = read_ram(&final_state.memory, addr);
        if actual_value != expected_value {
            return Err(format!(
                "{stage_name}: RAM[{addr}] expected {expected_value}, got {actual_value}"
            )
            .into());
        }
    }

    // Verify expected FP
    if let Some(expected_fp) = spec.expected_fp {
        let actual_fp = read_fp(&final_state.memory);
        if actual_fp != expected_fp {
            return Err(format!("{stage_name}: FP expected {expected_fp}, got {actual_fp}").into());
        }
    }

    // Verify expected PC
    if let Some(expected_pc) = spec.expected_pc {
        let actual_pc = final_state.pc();
        if actual_pc != expected_pc {
            return Err(format!("{stage_name}: PC expected {expected_pc}, got {actual_pc}").into());
        }
    }

    Ok(())
}

fn default_engine() -> BabyBearPoseidon2Engine {
    let fri_params =
        FriParameters::standard_with_100_bits_conjectured_security(DEFAULT_APP_LOG_BLOWUP);
    BabyBearPoseidon2Engine::new(fri_params)
}

/// Test Stage 1: Raw Execution using InterpretedInstance::execute_from_state
pub fn test_execution(spec: &IsolatedTestSpec) -> Result<(), Box<dyn std::error::Error>> {
    let exe = build_exe(spec);
    let vm_config = WomirConfig::default();
    let vm = VmExecutor::new(vm_config.clone()).unwrap();
    let instance = vm.instance(&exe).unwrap();

    let from_state = build_initial_state(spec, &exe, &vm_config);
    let final_state = instance.execute_from_state(from_state, None)?;

    verify_state(spec, &final_state, "test_execution")
}

/// Test Stage 2: Metered Execution using InterpretedInstance::execute_metered_from_state
pub fn test_metered_execution(spec: &IsolatedTestSpec) -> Result<(), Box<dyn std::error::Error>> {
    let exe = build_exe(spec);
    let vm_config = WomirConfig::default();
    let (vm, _pk) = VirtualMachine::<_, WomirCpuBuilder>::new_with_keygen(
        default_engine(),
        WomirCpuBuilder,
        vm_config.clone(),
    )?;

    let metered_ctx = vm.build_metered_ctx(&exe);
    let metered_instance = vm.metered_interpreter(&exe)?;
    let from_state = build_initial_state(spec, &exe, &vm_config);
    let (segments, final_state) =
        metered_instance.execute_metered_from_state(from_state, metered_ctx)?;

    assert_eq!(segments.len(), 1, "expected a single segment");
    verify_state(spec, &final_state, "test_metered_execution")
}

/// Test Stage 3: Preflight using VirtualMachine::execute_preflight
pub fn test_preflight(spec: &IsolatedTestSpec) -> Result<(), Box<dyn std::error::Error>> {
    let exe = build_exe(spec);
    let vm_config = WomirConfig::default();
    let (vm, _pk) = VirtualMachine::<_, WomirCpuBuilder>::new_with_keygen(
        default_engine(),
        WomirCpuBuilder,
        vm_config.clone(),
    )?;

    // Run metered execution to get segment info
    let metered_ctx = vm.build_metered_ctx(&exe);
    let metered_instance = vm.metered_interpreter(&exe)?;
    let from_state = build_initial_state(spec, &exe, &vm_config);
    let (segments, _) = metered_instance.execute_metered_from_state(from_state, metered_ctx)?;
    assert_eq!(segments.len(), 1, "Expected a single segment.");
    let Segment {
        num_insns,
        trace_heights,
        ..
    } = &segments[0];

    // Run preflight
    let mut preflight_interpreter = vm.preflight_interpreter(&exe)?;
    let preflight_from_state = build_initial_state(spec, &exe, &vm_config);
    let preflight_output = vm.execute_preflight(
        &mut preflight_interpreter,
        preflight_from_state,
        Some(*num_insns),
        trace_heights,
    )?;

    verify_state(spec, &preflight_output.to_state, "test_preflight")
}

/// Test Stage 4: Proof Generation using VirtualMachine::prove
pub fn test_proof(spec: &IsolatedTestSpec) -> Result<(), Box<dyn std::error::Error>> {
    let exe = build_exe(spec);
    let vm_config = WomirConfig::default();
    let (mut vm, _pk) = VirtualMachine::<_, WomirCpuBuilder>::new_with_keygen(
        default_engine(),
        WomirCpuBuilder,
        vm_config.clone(),
    )?;

    // Run metered execution to get segment info
    let metered_ctx = vm.build_metered_ctx(&exe);
    let metered_instance = vm.metered_interpreter(&exe)?;
    let from_state = build_initial_state(spec, &exe, &vm_config);
    let (segments, _) = metered_instance.execute_metered_from_state(from_state, metered_ctx)?;
    assert_eq!(segments.len(), 1, "Expected a single segment.");
    let Segment {
        num_insns,
        trace_heights,
        ..
    } = &segments[0];

    // Load program trace (required before prove)
    let cached_program_trace = vm.commit_program_on_device(&exe.program);
    vm.load_program(cached_program_trace);

    // Generate proof
    let mut preflight_interpreter = vm.preflight_interpreter(&exe)?;
    let proof_from_state = build_initial_state(spec, &exe, &vm_config);
    let (_proof, _final_memory) = vm.prove(
        &mut preflight_interpreter,
        proof_from_state,
        Some(*num_insns),
        trace_heights,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::setup_tracing_with_log_level;
    use tracing::Level;

    /// Test a single instruction through all isolated stages.
    #[test]
    fn test_add_imm_isolated_stages() -> Result<(), Box<dyn std::error::Error>> {
        setup_tracing_with_log_level(Level::WARN);

        // Test specification:
        // - Start state: all memory zero (fp=0)
        // - Program: [add_imm reg[8], reg[0], 100; halt]
        // - Expected: reg[8] = 100
        let spec = IsolatedTestSpec {
            program: vec![wom::add_imm::<F>(8, 0, 100_i16.into()), wom::halt()],
            expected_registers: vec![(8, 100)],
            ..Default::default()
        };

        test_execution(&spec)?;
        test_metered_execution(&spec)?;
        test_preflight(&spec)?;
        test_proof(&spec)?;

        Ok(())
    }
}
