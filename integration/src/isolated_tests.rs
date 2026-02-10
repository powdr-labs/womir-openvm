//! Isolated stage testing infrastructure for WOMIR instructions.
//!
//! This module provides a framework for testing WOMIR instructions
//! through isolated execution stages:
//! - Raw execution (InterpretedInstance::execute_from_state)
//! - Metered execution (InterpretedInstance::execute_metered_from_state)
//! - Preflight (VirtualMachine::execute_preflight)
//! - Proof generation (VirtualMachine::prove)

use openvm_circuit::arch::{VirtualMachine, VmExecutor, VmState, execution_mode::Segment};
use openvm_instructions::{
    exe::VmExe, instruction::Instruction, program::Program, riscv::RV32_REGISTER_AS,
};
use openvm_sdk::{StdIn, config::DEFAULT_APP_LOG_BLOWUP};
use openvm_stark_sdk::{
    config::{FriParameters, baby_bear_poseidon2::BabyBearPoseidon2Engine},
    engine::StarkFriEngine,
};
use womir_circuit::{
    WomirConfig, WomirCpuBuilder,
    adapters::RV32_REGISTER_NUM_LIMBS,
    memory_config::{FP_AS, FpMemory},
};

use crate::instruction_builder as wom;

type F = openvm_stark_sdk::p3_baby_bear::BabyBear;

/// Memory address space for RAM (heap memory).
const RV32_MEMORY_AS: u32 = openvm_instructions::riscv::RV32_MEMORY_AS;

/// Specification for an isolated instruction test.
/// Defines the start state, program, and expected end state.
#[derive(Clone)]
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

impl Default for IsolatedTestSpec {
    fn default() -> Self {
        Self {
            program: vec![wom::halt()],
            start_pc: None,
            start_fp: None,
            start_registers: vec![],
            start_ram: vec![],
            expected_pc: None,
            expected_fp: None,
            expected_registers: vec![],
            expected_ram: vec![],
        }
    }
}

/// Read a register value from memory.
/// Register address = reg_index * RV32_REGISTER_NUM_LIMBS
fn read_register(memory: &openvm_circuit::system::memory::online::GuestMemory, reg: usize) -> u32 {
    let addr = (reg * RV32_REGISTER_NUM_LIMBS) as u32;
    let bytes: [u8; 4] = unsafe { memory.read(RV32_REGISTER_AS, addr) };
    u32::from_le_bytes(bytes)
}

/// Read a 32-bit value from RAM.
fn read_ram(memory: &openvm_circuit::system::memory::online::GuestMemory, addr: u32) -> u32 {
    let bytes: [u8; 4] = unsafe { memory.read(RV32_MEMORY_AS, addr) };
    u32::from_le_bytes(bytes)
}

/// Read FP from memory.
fn read_fp(memory: &openvm_circuit::system::memory::online::GuestMemory) -> u32 {
    memory.fp()
}

/// Build a VmExe from a test specification.
fn build_exe(spec: &IsolatedTestSpec) -> VmExe<F> {
    let program = Program::from_instructions(&spec.program);
    let mut exe = VmExe::new(program);

    // Set start PC if specified
    if let Some(pc) = spec.start_pc {
        exe = exe.with_pc_start(pc);
    }

    // Set initial memory state (registers, RAM, FP)
    for &(reg, value) in &spec.start_registers {
        let addr = (reg * RV32_REGISTER_NUM_LIMBS) as u32;
        for (i, byte) in value.to_le_bytes().iter().enumerate() {
            if *byte != 0 {
                exe.init_memory
                    .insert((RV32_REGISTER_AS, addr + i as u32), *byte);
            }
        }
    }

    for &(addr, value) in &spec.start_ram {
        for (i, byte) in value.to_le_bytes().iter().enumerate() {
            if *byte != 0 {
                exe.init_memory
                    .insert((RV32_MEMORY_AS, addr + i as u32), *byte);
            }
        }
    }

    // Set initial FP if specified
    if let Some(fp) = spec.start_fp {
        for (i, byte) in fp.to_le_bytes().iter().enumerate() {
            if *byte != 0 {
                exe.init_memory.insert((FP_AS, i as u32), *byte);
            }
        }
    }

    exe
}

/// Create initial VmState from spec and exe.
fn build_initial_state(
    _spec: &IsolatedTestSpec,
    exe: &VmExe<F>,
    vm_config: &WomirConfig,
) -> VmState<F> {
    VmState::initial(
        &vm_config.system,
        &exe.init_memory,
        exe.pc_start,
        StdIn::default(),
    )
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

    let app_fri_params =
        FriParameters::standard_with_100_bits_conjectured_security(DEFAULT_APP_LOG_BLOWUP);
    let engine = BabyBearPoseidon2Engine::new(app_fri_params);

    let (vm, _pk) = VirtualMachine::<_, WomirCpuBuilder>::new_with_keygen(
        engine,
        WomirCpuBuilder,
        vm_config.clone(),
    )?;

    let metered_ctx = vm.build_metered_ctx(&exe);
    let metered_instance = vm.metered_interpreter(&exe)?;

    let from_state = build_initial_state(spec, &exe, &vm_config);
    let (segments, final_state) =
        metered_instance.execute_metered_from_state(from_state, metered_ctx)?;

    // Verify we got at least one segment
    assert!(
        !segments.is_empty(),
        "test_metered_execution: expected at least one segment"
    );

    verify_state(spec, &final_state, "test_metered_execution")
}

/// Test Stage 3: Preflight using VirtualMachine::execute_preflight
pub fn test_preflight(spec: &IsolatedTestSpec) -> Result<(), Box<dyn std::error::Error>> {
    let exe = build_exe(spec);
    let vm_config = WomirConfig::default();

    let app_fri_params =
        FriParameters::standard_with_100_bits_conjectured_security(DEFAULT_APP_LOG_BLOWUP);
    let engine = BabyBearPoseidon2Engine::new(app_fri_params);

    let (vm, _pk) = VirtualMachine::<_, WomirCpuBuilder>::new_with_keygen(
        engine,
        WomirCpuBuilder,
        vm_config.clone(),
    )?;

    // First run metered execution to get segment info
    let metered_ctx = vm.build_metered_ctx(&exe);
    let metered_instance = vm.metered_interpreter(&exe)?;
    let from_state = build_initial_state(spec, &exe, &vm_config);
    let (segments, _) =
        metered_instance.execute_metered_from_state(from_state.clone(), metered_ctx)?;

    // Now run preflight
    let mut preflight_interpreter = vm.preflight_interpreter(&exe)?;
    let Segment {
        num_insns,
        trace_heights,
        ..
    } = &segments[0];

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

    let app_fri_params =
        FriParameters::standard_with_100_bits_conjectured_security(DEFAULT_APP_LOG_BLOWUP);
    let engine = BabyBearPoseidon2Engine::new(app_fri_params);

    let (mut vm, _pk) = VirtualMachine::<_, WomirCpuBuilder>::new_with_keygen(
        engine,
        WomirCpuBuilder,
        vm_config.clone(),
    )?;

    // First run metered execution to get segment info (trace heights)
    let metered_ctx = vm.build_metered_ctx(&exe);
    let metered_instance = vm.metered_interpreter(&exe)?;
    let from_state = build_initial_state(spec, &exe, &vm_config);
    let (segments, _) = metered_instance.execute_metered_from_state(from_state, metered_ctx)?;

    // Load the program trace (required before calling prove)
    let cached_program_trace = vm.commit_program_on_device(&exe.program);
    vm.load_program(cached_program_trace);

    // Get the preflight interpreter and trace heights from first segment
    let mut preflight_interpreter = vm.preflight_interpreter(&exe)?;
    let Segment {
        num_insns,
        trace_heights,
        ..
    } = &segments[0];

    // Generate proof from initial state using VirtualMachine::prove
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
