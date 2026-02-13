//! Isolated stage testing infrastructure for WOMIR instructions.
//!
//! This module provides a framework for testing WOMIR instructions
//! through isolated execution stages:
//! - Raw execution (InterpretedInstance::execute_from_state)
//! - Metered execution (InterpretedInstance::execute_metered_from_state)
//! - Preflight (VirtualMachine::execute_preflight)
//! - Proof generation (VirtualMachine::prove)

use openvm_circuit::{
    arch::{VirtualMachine, VmExecutor, VmState, debug_proving_ctx, execution_mode::Segment},
    system::memory::online::GuestMemory,
};
use openvm_instructions::{
    exe::VmExe,
    instruction::Instruction,
    program::{DEFAULT_PC_STEP, Program},
    riscv::RV32_REGISTER_AS,
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

/// Specification for an test case.
/// Defines the start state, program, and expected end state.
#[derive(Clone, Default)]
pub struct TestSpec {
    /// The program to execute (should end with halt instruction).
    pub program: Vec<Instruction<F>>,

    /// Initial PC (default: 0).
    pub start_pc: u32,
    /// Initial FP (default: 0).
    pub start_fp: u32,
    /// Initial register values: (register_index, value).
    pub start_registers: Vec<(usize, u32)>,
    /// Initial RAM values: (address, value).
    pub start_ram: Vec<(u32, u32)>,

    /// Expected PC after execution (if None, start_pc + num_instructions * DEFAULT_PC_STEP).
    pub expected_pc: Option<u32>,
    /// Expected FP after execution (if None, start_fp).
    pub expected_fp: Option<u32>,
    /// Expected register values after execution: (register_index, value).
    pub expected_registers: Vec<(usize, u32)>,
    /// Expected RAM values after execution: (address, value).
    pub expected_ram: Vec<(u32, u32)>,
}

/// Read a register value from memory.
fn read_register(memory: &GuestMemory, reg: usize) -> u32 {
    // The instruction builder multiplies all registers by RV32_REGISTER_NUM_LIMBS, so this is the
    // address that actually hits the key/value store.
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
    memory.fp::<F>()
}

/// Build a VmExe from a test specification (program and PC only).
fn build_exe(spec: &TestSpec) -> VmExe<F> {
    let program = Program::from_instructions(&spec.program);
    VmExe::new(program).with_pc_start(spec.start_pc)
}

/// Create initial VmState from spec, exe, and config.
/// Sets up memory with initial register values, RAM values, and FP from the spec.
fn build_initial_state(spec: &TestSpec, exe: &VmExe<F>, vm_config: &WomirConfig) -> VmState<F> {
    let mut state = VmState::initial(
        &vm_config.system,
        &exe.init_memory,
        exe.pc_start,
        StdIn::default(),
    );

    // Set initial registers
    for &(reg, value) in &spec.start_registers {
        // The instruction builder multiplies all registers by RV32_REGISTER_NUM_LIMBS, so this is the
        // address that actually hits the key/value store.
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
    // Again, the raw value stored in the state is the FP multiplied by RV32_REGISTER_NUM_LIMBS.
    state
        .memory
        .set_fp::<F>(spec.start_fp * RV32_REGISTER_NUM_LIMBS as u32);

    state
}

/// Verify the final state matches expected values.
fn verify_state(
    spec: &TestSpec,
    final_state: &VmState<F>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Verify expected registers
    for &(reg, expected_value) in &spec.expected_registers {
        let actual_value = read_register(&final_state.memory, reg);
        if actual_value != expected_value {
            return Err(format!("reg[{reg}] expected {expected_value}, got {actual_value}").into());
        }
    }

    // Verify expected RAM
    for &(addr, expected_value) in &spec.expected_ram {
        let actual_value = read_ram(&final_state.memory, addr);
        if actual_value != expected_value {
            return Err(
                format!("RAM[{addr}] expected {expected_value}, got {actual_value}").into(),
            );
        }
    }

    // Verify expected FP
    // The raw value stored in the state is the FP multiplied by RV32_REGISTER_NUM_LIMBS, so we need to divide it to get the actual FP.
    let actual_fp = read_fp(&final_state.memory) / RV32_REGISTER_NUM_LIMBS as u32;
    let expected_fp = spec.expected_fp.unwrap_or(spec.start_fp);
    if actual_fp != expected_fp {
        return Err(format!("FP expected {expected_fp}, got {actual_fp}").into());
    }

    // Verify expected PC
    let actual_pc = final_state.pc();
    let expected_pc = spec
        .expected_pc
        // The added HALT instruction does not advance the PC!
        .unwrap_or(spec.start_pc + ((spec.program.len() - 1) as u32 * DEFAULT_PC_STEP));
    if actual_pc != expected_pc {
        return Err(format!("PC expected {expected_pc}, got {actual_pc}").into());
    }

    Ok(())
}

fn default_engine() -> BabyBearPoseidon2Engine {
    let fri_params =
        FriParameters::standard_with_100_bits_conjectured_security(DEFAULT_APP_LOG_BLOWUP);
    BabyBearPoseidon2Engine::new(fri_params)
}

/// Test Stage 1: Raw Execution using InterpretedInstance::execute_from_state
pub fn test_execution(spec: &TestSpec) -> Result<(), Box<dyn std::error::Error>> {
    let exe = build_exe(spec);
    let vm_config = WomirConfig::default();
    let vm = VmExecutor::new(vm_config.clone()).unwrap();
    let instance = vm.instance(&exe).unwrap();

    let from_state = build_initial_state(spec, &exe, &vm_config);
    let final_state = instance.execute_from_state(from_state, None)?;

    verify_state(spec, &final_state)
}

/// Test Stage 2: Metered Execution using InterpretedInstance::execute_metered_from_state
pub fn test_metered_execution(spec: &TestSpec) -> Result<(), Box<dyn std::error::Error>> {
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
    verify_state(spec, &final_state)
}

/// Test Stage 3: Preflight using VirtualMachine::execute_preflight
pub fn test_preflight(spec: &TestSpec) -> Result<(), Box<dyn std::error::Error>> {
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

    verify_state(spec, &preflight_output.to_state)
}

/// Test Stage 4: Mock proving with constraint verification using debug_proving_ctx.
/// This generates traces and verifies all constraints are satisfied without
/// generating actual cryptographic proofs.
pub fn test_prove(spec: &TestSpec) -> Result<(), Box<dyn std::error::Error>> {
    let exe = build_exe(spec);
    let vm_config = WomirConfig::default();
    let (mut vm, pk) = VirtualMachine::<_, WomirCpuBuilder>::new_with_keygen(
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

    // Load program trace (required before preflight)
    let cached_program_trace = vm.commit_program_on_device(&exe.program);
    vm.load_program(cached_program_trace);

    // Run preflight to generate traces
    let mut preflight_interpreter = vm.preflight_interpreter(&exe)?;
    let preflight_from_state = build_initial_state(spec, &exe, &vm_config);
    vm.transport_init_memory_to_device(&preflight_from_state.memory);
    let preflight_output = vm.execute_preflight(
        &mut preflight_interpreter,
        preflight_from_state,
        Some(*num_insns),
        trace_heights,
    )?;

    // Generate proving context with traces
    let ctx = vm.generate_proving_ctx(
        preflight_output.system_records,
        preflight_output.record_arenas,
    )?;

    // Verify all constraints using mock prover
    debug_proving_ctx(&vm, &pk, &ctx);

    Ok(())
}

pub fn test_spec(mut spec: TestSpec) {
    // Append halt instruction:
    spec.program.push(wom::halt());

    // Test all stages and collect errors
    let mut is_error = false;
    if let Err(e) = test_execution(&spec) {
        println!("test_execution: {e}");
        is_error = true;
    }
    if let Err(e) = test_metered_execution(&spec) {
        println!("test_metered_execution: {e}");
        is_error = true;
    }
    if let Err(e) = test_preflight(&spec) {
        println!("test_preflight: {e}");
        is_error = true;
    }
    if let Err(e) = test_prove(&spec) {
        println!("test_prove: {e}");
        is_error = true;
    }

    assert!(!is_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::setup_tracing_with_log_level;
    use tracing::Level;

    const REG_BASE_FITS_1_BYTE: u32 = 200;
    const REG_BASE_FITS_2_BYTES: u32 = 30_000;
    const REG_BASE_FITS_3_BYTES: u32 = 3_000_000;

    fn rebase_spec(spec: &TestSpec, new_base: u32) -> TestSpec {
        let mut rebased = spec.clone();
        let old_base = spec.start_fp;
        let delta = new_base as i64 - old_base as i64;

        let shift_register = |register_index: usize| -> usize {
            let shifted = register_index as i64 + delta;
            assert!(
                shifted >= 0,
                "register index underflow after rebasing: {register_index} -> {shifted}"
            );
            shifted as usize
        };

        rebased.start_fp = new_base;
        rebased.start_registers = spec
            .start_registers
            .iter()
            .map(|(register_index, value)| (shift_register(*register_index), *value))
            .collect();
        rebased.expected_registers = spec
            .expected_registers
            .iter()
            .map(|(register_index, value)| (shift_register(*register_index), *value))
            .collect();

        rebased
    }

    fn test_spec_for_all_register_bases(spec: TestSpec) {
        test_spec(rebase_spec(&spec, REG_BASE_FITS_1_BYTE));
        test_spec(rebase_spec(&spec, REG_BASE_FITS_2_BYTES));
        test_spec(rebase_spec(&spec, REG_BASE_FITS_3_BYTES));
    }

    // ==================== BaseAlu Tests ====================

    #[test]
    fn test_add_imm() {
        setup_tracing_with_log_level(Level::WARN);

        let spec = TestSpec {
            program: vec![wom::add_imm::<F>(1, 0, 100_i16.into())],
            start_fp: 124,
            expected_registers: vec![(125, 100)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_add() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+2] = reg[fp+0] + reg[fp+1]
        // 30 + 12 = 42
        let spec = TestSpec {
            program: vec![wom::add::<F>(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 30), (11, 12)],
            expected_registers: vec![(12, 42)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_add_byte_carry() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+2] = reg[fp+0] + reg[fp+1]
        // 0xFF + 1 = 0x100 (carry into second byte)
        let spec = TestSpec {
            program: vec![wom::add::<F>(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 0xFF), (11, 1)],
            expected_registers: vec![(12, 0x100)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_add_u32_overflow() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+2] = reg[fp+0] + reg[fp+1]
        // 0xFFFFFFFF + 1 = 0 (wrapping overflow)
        let spec = TestSpec {
            program: vec![wom::add::<F>(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 0xFFFFFFFF), (11, 1)],
            expected_registers: vec![(12, 0)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_sub() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+2] = reg[fp+0] - reg[fp+1]
        // 100 - 42 = 58
        let spec = TestSpec {
            program: vec![wom::sub::<F>(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 100), (11, 42)],
            expected_registers: vec![(12, 58)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_sub_negative_result() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+2] = reg[fp+0] - reg[fp+1]
        // 10 - 20 = -10 (wraps to 0xFFFFFFF6)
        let spec = TestSpec {
            program: vec![wom::sub::<F>(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 10), (11, 20)],
            expected_registers: vec![(12, 0xFFFFFFF6)], // -10 as u32
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_sub_underflow() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+2] = reg[fp+0] - reg[fp+1]
        // 0 - 1 = 0xFFFFFFFF (wrapping underflow)
        let spec = TestSpec {
            program: vec![wom::sub::<F>(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 0), (11, 1)],
            expected_registers: vec![(12, 0xFFFFFFFF)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_xor() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+2] = reg[fp+0] ^ reg[fp+1]
        // 0b1010 ^ 0b1100 = 0b0110
        let spec = TestSpec {
            program: vec![wom::xor::<F>(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 0b1010), (11, 0b1100)],
            expected_registers: vec![(12, 0b0110)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_or() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+2] = reg[fp+0] | reg[fp+1]
        // 0b1010 | 0b1100 = 0b1110
        let spec = TestSpec {
            program: vec![wom::or::<F>(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 0b1010), (11, 0b1100)],
            expected_registers: vec![(12, 0b1110)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_and() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+2] = reg[fp+0] & reg[fp+1]
        // 0b1010 & 0b1100 = 0b1000
        let spec = TestSpec {
            program: vec![wom::and::<F>(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 0b1010), (11, 0b1100)],
            expected_registers: vec![(12, 0b1000)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_and_imm() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+1] = reg[fp+0] & 0xFF
        // 0x1234 & 0xFF = 0x34
        let spec = TestSpec {
            program: vec![wom::and_imm::<F>(1, 0, 0xFF_i16.into())],
            start_fp: 10,
            start_registers: vec![(10, 0x1234)],
            expected_registers: vec![(11, 0x34)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    // ==================== LoadStore Tests ====================

    #[test]
    fn test_loadw() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+1] = MEM[reg[fp+0] + 0]
        // Load 0xDEADBEEF from address 100
        let spec = TestSpec {
            program: vec![wom::loadw::<F>(1, 0, 0)],
            start_fp: 10,
            start_registers: vec![(10, 100)], // base address = 100
            start_ram: vec![(100, 0xDEADBEEF)],
            expected_registers: vec![(11, 0xDEADBEEF)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_loadw_with_offset() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+1] = MEM[reg[fp+0] + 8]
        // Load from address 100 + 8 = 108
        let spec = TestSpec {
            program: vec![wom::loadw::<F>(1, 0, 8)],
            start_fp: 10,
            start_registers: vec![(10, 100)],
            start_ram: vec![(108, 0x12345678)],
            expected_registers: vec![(11, 0x12345678)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_storew() {
        setup_tracing_with_log_level(Level::WARN);

        // MEM[reg[fp+1] + 0] = reg[fp+0]
        // Store 0xCAFEBABE at address 200
        let spec = TestSpec {
            program: vec![wom::storew::<F>(0, 1, 0)],
            start_fp: 10,
            start_registers: vec![(10, 0xCAFEBABE), (11, 200)],
            expected_ram: vec![(200, 0xCAFEBABE)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_storew_with_offset() {
        setup_tracing_with_log_level(Level::WARN);

        // MEM[reg[fp+1] + 4] = reg[fp+0]
        // Store at address 200 + 4 = 204
        let spec = TestSpec {
            program: vec![wom::storew::<F>(0, 1, 4)],
            start_fp: 10,
            start_registers: vec![(10, 0x11223344), (11, 200)],
            expected_ram: vec![(204, 0x11223344)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_loadbu() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+1] = MEM[reg[fp+0] + 0] (zero-extended byte)
        // Load byte 0xAB, should remain 0x000000AB
        let spec = TestSpec {
            program: vec![wom::loadbu::<F>(1, 0, 0)],
            start_fp: 10,
            start_registers: vec![(10, 100)],
            start_ram: vec![(100, 0xFFFFFFAB)], // Only lowest byte matters
            expected_registers: vec![(11, 0xAB)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_loadhu() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+1] = MEM[reg[fp+0] + 0] (zero-extended halfword)
        // Load halfword 0xABCD, should remain 0x0000ABCD
        let spec = TestSpec {
            program: vec![wom::loadhu::<F>(1, 0, 0)],
            start_fp: 10,
            start_registers: vec![(10, 100)],
            start_ram: vec![(100, 0xFFFFABCD)], // Only lowest 2 bytes matter
            expected_registers: vec![(11, 0xABCD)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_storeb() {
        setup_tracing_with_log_level(Level::WARN);

        // MEM[reg[fp+1] + 0] = reg[fp+0] (lowest byte only)
        // Store byte 0x42 at address 200
        let spec = TestSpec {
            program: vec![wom::storeb::<F>(0, 1, 0)],
            start_fp: 10,
            start_registers: vec![(10, 0x12345642), (11, 200)],
            start_ram: vec![(200, 0xFFFFFFFF)], // Prefill with ones
            expected_ram: vec![(200, 0xFFFFFF42)], // Only lowest byte changed
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_storeh() {
        setup_tracing_with_log_level(Level::WARN);

        // MEM[reg[fp+1] + 0] = reg[fp+0] (lowest halfword only)
        // Store halfword 0xBEEF at address 200
        let spec = TestSpec {
            program: vec![wom::storeh::<F>(0, 1, 0)],
            start_fp: 10,
            start_registers: vec![(10, 0x1234BEEF), (11, 200)],
            start_ram: vec![(200, 0xFFFFFFFF)], // Prefill with ones
            expected_ram: vec![(200, 0xFFFFBEEF)], // Only lowest 2 bytes changed
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    // ==================== LoadSignExtend Tests ====================

    #[test]
    fn test_loadb_positive() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+1] = MEM[reg[fp+0] + 0] (sign-extended byte)
        // Load byte 0x7F (positive), should remain 0x0000007F
        let spec = TestSpec {
            program: vec![wom::loadb::<F>(1, 0, 0)],
            start_fp: 10,
            start_registers: vec![(10, 100)],
            start_ram: vec![(100, 0x0000007F)],
            expected_registers: vec![(11, 0x0000007F)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_loadb_negative() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+1] = MEM[reg[fp+0] + 0] (sign-extended byte)
        // Load byte 0x80 (negative), should become 0xFFFFFF80
        let spec = TestSpec {
            program: vec![wom::loadb::<F>(1, 0, 0)],
            start_fp: 10,
            start_registers: vec![(10, 100)],
            start_ram: vec![(100, 0x00000080)],
            expected_registers: vec![(11, 0xFFFFFF80)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_loadh_positive() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+1] = MEM[reg[fp+0] + 0] (sign-extended halfword)
        // Load halfword 0x7FFF (positive), should remain 0x00007FFF
        let spec = TestSpec {
            program: vec![wom::loadh::<F>(1, 0, 0)],
            start_fp: 10,
            start_registers: vec![(10, 100)],
            start_ram: vec![(100, 0x00007FFF)],
            expected_registers: vec![(11, 0x00007FFF)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_loadh_negative() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+1] = MEM[reg[fp+0] + 0] (sign-extended halfword)
        // Load halfword 0x8000 (negative), should become 0xFFFF8000
        let spec = TestSpec {
            program: vec![wom::loadh::<F>(1, 0, 0)],
            start_fp: 10,
            start_registers: vec![(10, 100)],
            start_ram: vec![(100, 0x00008000)],
            expected_registers: vec![(11, 0xFFFF8000)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    // ==================== LessThan Tests ====================

    #[test]
    fn test_lt_u_true() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+2] = (reg[fp+0] < reg[fp+1]) unsigned
        // 10 < 20 = 1
        let spec = TestSpec {
            program: vec![wom::lt_u::<F>(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 10), (11, 20)],
            expected_registers: vec![(12, 1)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_lt_u_false() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+2] = (reg[fp+0] < reg[fp+1]) unsigned
        // 20 < 10 = 0
        let spec = TestSpec {
            program: vec![wom::lt_u::<F>(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 20), (11, 10)],
            expected_registers: vec![(12, 0)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_lt_u_equal() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+2] = (reg[fp+0] < reg[fp+1]) unsigned
        // 42 < 42 = 0
        let spec = TestSpec {
            program: vec![wom::lt_u::<F>(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 42), (11, 42)],
            expected_registers: vec![(12, 0)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_lt_s_negative() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+2] = (reg[fp+0] < reg[fp+1]) signed
        // -1 (0xFFFFFFFF) < 1 = 1 (signed)
        let spec = TestSpec {
            program: vec![wom::lt_s::<F>(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 0xFFFFFFFF), (11, 1)],
            expected_registers: vec![(12, 1)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_lt_u_imm() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+1] = (reg[fp+0] < 100) unsigned
        // 50 < 100 = 1
        let spec = TestSpec {
            program: vec![wom::lt_u_imm::<F>(1, 0, 100_i16.into())],
            start_fp: 10,
            start_registers: vec![(10, 50)],
            expected_registers: vec![(11, 1)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    // ==================== LessThan64 Tests ====================

    #[test]
    fn test_lt_u_64_true() {
        setup_tracing_with_log_level(Level::WARN);

        // 0x0000_0001_0000_0000 < 0x0000_0002_0000_0000 = 1 (unsigned)
        let spec = TestSpec {
            program: vec![wom::lt_u_64::<F>(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 0),
                (125, 1), // reg 0 = 0x0000_0001_0000_0000
                (126, 0),
                (127, 2), // reg 2 = 0x0000_0002_0000_0000
            ],
            expected_registers: vec![(128, 1), (129, 0)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_lt_u_64_false() {
        setup_tracing_with_log_level(Level::WARN);

        // 0x0000_0002_0000_0000 < 0x0000_0001_0000_0000 = 0 (unsigned)
        let spec = TestSpec {
            program: vec![wom::lt_u_64::<F>(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 0),
                (125, 2), // reg 0 = 0x0000_0002_0000_0000
                (126, 0),
                (127, 1), // reg 2 = 0x0000_0001_0000_0000
            ],
            expected_registers: vec![(128, 0), (129, 0)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_lt_s_64_negative() {
        setup_tracing_with_log_level(Level::WARN);

        // -1 (0xFFFF_FFFF_FFFF_FFFF) < 1 (0x0000_0000_0000_0001) = 1 (signed)
        let spec = TestSpec {
            program: vec![wom::lt_s_64::<F>(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 0xFFFF_FFFF),
                (125, 0xFFFF_FFFF), // reg 0 = -1
                (126, 1),
                (127, 0), // reg 2 = 1
            ],
            expected_registers: vec![(128, 1), (129, 0)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    // ==================== add_64 ====================

    #[test]
    fn test_add_imm_64() {
        setup_tracing_with_log_level(Level::WARN);

        // 0x0000_ffff_0000_0001 + 0x80 = 0x0000_ffff_0000_0081
        let spec = TestSpec {
            program: vec![wom::add_imm_64::<F>(2, 0, 0x80_i16.into())],
            start_fp: 124,
            start_registers: vec![(124, 1), (125, 0xffff)],
            expected_registers: vec![(126, 0x81), (127, 0xffff)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_add_imm_64_low_overflow() {
        setup_tracing_with_log_level(Level::WARN);

        // 0x0000_0001_FFFF_FF00 + 0x0100 = 0x0000_0002_0000_0000
        // Low limb overflows, carry propagates to high limb
        let spec = TestSpec {
            program: vec![wom::add_imm_64::<F>(2, 0, 0x0100_i16.into())],
            start_fp: 124,
            start_registers: vec![(124, 0xFFFF_FF00), (125, 0x0000_0001)],
            expected_registers: vec![(126, 0x0000_0000), (127, 0x0000_0002)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_add_imm_64_full_overflow() {
        setup_tracing_with_log_level(Level::WARN);

        // 0xFFFF_FFFF_FFFF_FFFF + 1 = 0 (wraps around)
        let spec = TestSpec {
            program: vec![wom::add_imm_64::<F>(2, 0, 1_i16.into())],
            start_fp: 124,
            start_registers: vec![(124, 0xFFFF_FFFF), (125, 0xFFFF_FFFF)],
            expected_registers: vec![(126, 0), (127, 0)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_add_64_reg() {
        setup_tracing_with_log_level(Level::WARN);

        // 0x0000_0001_0000_0003 + 0x0000_0002_0000_0004 = 0x0000_0003_0000_0007
        let spec = TestSpec {
            program: vec![wom::add_64::<F>(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 3),
                (125, 1), // reg 0 = 0x0000_0001_0000_0003
                (126, 4),
                (127, 2), // reg 2 = 0x0000_0002_0000_0004
            ],
            expected_registers: vec![(128, 7), (129, 3)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_add_64_reg_low_overflow() {
        setup_tracing_with_log_level(Level::WARN);

        // 0x0000_0001_8000_0000 + 0x0000_0001_8000_0000 = 0x0000_0003_0000_0000
        let spec = TestSpec {
            program: vec![wom::add_64::<F>(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 0x8000_0000),
                (125, 1), // reg 0
                (126, 0x8000_0000),
                (127, 1), // reg 2
            ],
            expected_registers: vec![(128, 0x0000_0000), (129, 3)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_add_64_reg_full_overflow() {
        setup_tracing_with_log_level(Level::WARN);

        // 0xFFFF_FFFF_FFFF_FFFE + 0x0000_0000_0000_0003 = 0x0000_0000_0000_0001
        let spec = TestSpec {
            program: vec![wom::add_64::<F>(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 0xFFFF_FFFE),
                (125, 0xFFFF_FFFF), // reg 0
                (126, 3),
                (127, 0), // reg 2
            ],
            expected_registers: vec![(128, 1), (129, 0)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    // ==================== sub_64 ====================

    #[test]
    fn test_sub_64_reg() {
        setup_tracing_with_log_level(Level::WARN);

        // 0x0000_0003_0000_0007 - 0x0000_0001_0000_0003 = 0x0000_0002_0000_0004
        let spec = TestSpec {
            program: vec![wom::sub_64::<F>(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 7),
                (125, 3), // reg 0
                (126, 3),
                (127, 1), // reg 2
            ],
            expected_registers: vec![(128, 4), (129, 2)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_sub_64_reg_low_borrow() {
        setup_tracing_with_log_level(Level::WARN);

        // 0x0000_0003_0000_0000 - 0x0000_0001_0000_0001 = 0x0000_0001_FFFF_FFFF
        // Low limb borrows from high limb
        let spec = TestSpec {
            program: vec![wom::sub_64::<F>(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 0x0000_0000),
                (125, 3), // reg 0
                (126, 1),
                (127, 1), // reg 2
            ],
            expected_registers: vec![(128, 0xFFFF_FFFF), (129, 1)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_sub_64_reg_full_underflow() {
        setup_tracing_with_log_level(Level::WARN);

        // 0x0000_0000_0000_0001 - 0x0000_0000_0000_0003 = 0xFFFF_FFFF_FFFF_FFFE (wraps)
        let spec = TestSpec {
            program: vec![wom::sub_64::<F>(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 1),
                (125, 0), // reg 0
                (126, 3),
                (127, 0), // reg 2
            ],
            expected_registers: vec![(128, 0xFFFF_FFFE), (129, 0xFFFF_FFFF)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_sub_imm_64() {
        setup_tracing_with_log_level(Level::WARN);

        // 0x0000_0002_0000_0000 - 1 = 0x0000_0001_FFFF_FFFF
        // Low borrow propagates to high limb
        let spec = TestSpec {
            program: vec![wom::sub_imm_64::<F>(2, 0, 1_i16.into())],
            start_fp: 124,
            start_registers: vec![(124, 0x0000_0000), (125, 2)],
            expected_registers: vec![(126, 0xFFFF_FFFF), (127, 1)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    // ==================== xor_64 ====================

    #[test]
    fn test_xor_64_reg() {
        setup_tracing_with_log_level(Level::WARN);

        // 0xDEAD_BEEF_CAFE_BABE ^ 0xFFFF_FFFF_0000_0000 = 0x2152_4110_CAFE_BABE
        let spec = TestSpec {
            program: vec![wom::xor_64::<F>(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 0xCAFE_BABE),
                (125, 0xDEAD_BEEF), // reg 0
                (126, 0x0000_0000),
                (127, 0xFFFF_FFFF), // reg 2
            ],
            expected_registers: vec![(128, 0xCAFE_BABE), (129, 0x2152_4110)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_xor_imm_64() {
        setup_tracing_with_log_level(Level::WARN);

        // 0x0000_0001_0000_00FF ^ 0xFF (sign-extended to 0x0000_0000_0000_00FF) = 0x0000_0001_0000_0000
        let spec = TestSpec {
            program: vec![wom::xor_imm_64::<F>(2, 0, 0xFF_i16.into())],
            start_fp: 124,
            start_registers: vec![(124, 0x0000_00FF), (125, 1)],
            expected_registers: vec![(126, 0x0000_0000), (127, 1)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    // ==================== or_64 ====================

    #[test]
    fn test_or_64_reg() {
        setup_tracing_with_log_level(Level::WARN);

        // 0x00FF_00FF_00FF_00FF | 0xFF00_FF00_FF00_FF00 = 0xFFFF_FFFF_FFFF_FFFF
        let spec = TestSpec {
            program: vec![wom::or_64::<F>(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 0x00FF_00FF),
                (125, 0x00FF_00FF), // reg 0
                (126, 0xFF00_FF00),
                (127, 0xFF00_FF00), // reg 2
            ],
            expected_registers: vec![(128, 0xFFFF_FFFF), (129, 0xFFFF_FFFF)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_or_imm_64() {
        setup_tracing_with_log_level(Level::WARN);

        // 0x0000_0001_0000_0000 | 0x0F (sign-extended to 0x0000_0000_0000_000F) = 0x0000_0001_0000_000F
        let spec = TestSpec {
            program: vec![wom::or_imm_64::<F>(2, 0, 0x0F_i16.into())],
            start_fp: 124,
            start_registers: vec![(124, 0x0000_0000), (125, 1)],
            expected_registers: vec![(126, 0x0000_000F), (127, 1)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    // ==================== and_64 ====================

    #[test]
    fn test_and_64_reg() {
        setup_tracing_with_log_level(Level::WARN);

        // 0xFFFF_0000_FFFF_0000 & 0x0F0F_0F0F_0F0F_0F0F = 0x0F0F_0000_0F0F_0000
        let spec = TestSpec {
            program: vec![wom::and_64::<F>(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 0xFFFF_0000),
                (125, 0xFFFF_0000), // reg 0
                (126, 0x0F0F_0F0F),
                (127, 0x0F0F_0F0F), // reg 2
            ],
            expected_registers: vec![(128, 0x0F0F_0000), (129, 0x0F0F_0000)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_and_imm_64() {
        setup_tracing_with_log_level(Level::WARN);

        // 0xDEAD_BEEF_CAFE_BABE & 0xFF (sign-extended to 0x0000_0000_0000_00FF) = 0x0000_0000_0000_00BE
        let spec = TestSpec {
            program: vec![wom::and_imm_64::<F>(2, 0, 0xFF_i16.into())],
            start_fp: 124,
            start_registers: vec![(124, 0xCAFE_BABE), (125, 0xDEAD_BEEF)],
            expected_registers: vec![(126, 0x0000_00BE), (127, 0x0000_0000)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    // ==================== Jump Tests ====================

    #[test]
    fn test_jump() {
        setup_tracing_with_log_level(Level::WARN);

        let spec = TestSpec {
            program: vec![
                wom::jump::<F>(8),
                wom::halt(), // Should be skipped!
            ],
            expected_pc: Some(8),
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_jump_if_true() {
        setup_tracing_with_log_level(Level::WARN);

        let spec = TestSpec {
            program: vec![
                wom::jump_if::<F>(2, 8),
                wom::halt(), // Should be skipped!
            ],
            start_fp: 10,
            start_registers: vec![(12, 5)], // Should jump
            expected_pc: Some(8),
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_jump_if_false() {
        setup_tracing_with_log_level(Level::WARN);

        let spec = TestSpec {
            program: vec![wom::jump_if::<F>(2, 8)],
            start_fp: 10,
            start_registers: vec![(12, 0)], // Should not jump
            expected_pc: Some(4),
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_jump_if_zero_true() {
        setup_tracing_with_log_level(Level::WARN);

        let spec = TestSpec {
            program: vec![
                wom::jump_if_zero::<F>(2, 8),
                wom::halt(), // Should be skipped!
            ],
            start_fp: 10,
            start_registers: vec![(12, 0)], // Should jump
            expected_pc: Some(8),
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_jump_if_zero_false() {
        setup_tracing_with_log_level(Level::WARN);

        let spec = TestSpec {
            program: vec![wom::jump_if_zero::<F>(2, 2)],
            start_fp: 10,
            start_registers: vec![(12, 5)], // Should not jump
            expected_pc: Some(4),
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_skip() {
        setup_tracing_with_log_level(Level::WARN);

        let spec = TestSpec {
            program: vec![
                wom::skip::<F>(2),
                wom::halt(), // Should be skipped!
            ],
            start_fp: 10,
            start_registers: vec![(12, 2)],
            expected_pc: Some(8),
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    // ==================== Const32 Tests ====================

    #[test]
    fn test_const32_small() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+1] = 42
        let spec = TestSpec {
            program: vec![wom::const_32_imm::<F>(1, 42, 0)],
            start_fp: 10,
            expected_registers: vec![(11, 42)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_const32_large() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+1] = 0xDEADBEEF
        // imm_lo = 0xBEEF, imm_hi = 0xDEAD
        let spec = TestSpec {
            program: vec![wom::const_32_imm::<F>(1, 0xBEEF, 0xDEAD)],
            start_fp: 10,
            expected_registers: vec![(11, 0xDEADBEEF)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_const32_zero() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+1] = 0
        let spec = TestSpec {
            program: vec![wom::const_32_imm::<F>(1, 0, 0)],
            start_fp: 10,
            start_registers: vec![(11, 0x12345678)], // Should be overwritten
            expected_registers: vec![(11, 0)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_const32_max() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+1] = 0xFFFFFFFF
        let spec = TestSpec {
            program: vec![wom::const_32_imm::<F>(1, 0xFFFF, 0xFFFF)],
            start_fp: 10,
            expected_registers: vec![(11, 0xFFFFFFFF)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    // ==================== Cross-width tests ====================

    #[test]
    fn test_cross_width_32_to_64() {
        setup_tracing_with_log_level(Level::WARN);

        // 32-bit writes 0x42 to reg fp+0, then 64-bit reads reg pair fp+0:fp+1
        let spec = TestSpec {
            program: vec![
                wom::add_imm::<F>(0, 0, 0x42_i16.into()),
                wom::add_imm_64::<F>(2, 0, 0_i16.into()),
            ],
            start_fp: 10,
            expected_registers: vec![(10, 0x42), (12, 0x42), (13, 0)],
            ..Default::default()
        };
        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_cross_width_64_to_32() {
        setup_tracing_with_log_level(Level::WARN);

        // 64-bit writes 0x42 to reg pair fp+0:fp+1, then 32-bit reads reg fp+0
        let spec = TestSpec {
            program: vec![
                wom::add_imm_64::<F>(0, 0, 0x42_i16.into()),
                wom::add_imm::<F>(2, 0, 0_i16.into()),
            ],
            start_fp: 10,
            expected_registers: vec![(10, 0x42), (11, 0), (12, 0x42)],
            ..Default::default()
        };
        test_spec_for_all_register_bases(spec)
    }

    // ==================== JAAF Tests ====================
    //
    // JAAF instructions change the frame pointer (FP) and jump to a new PC.
    // Register accesses are FP-relative: register N at FP=F is at absolute
    // address (N * 4 + F * 4), which is absolute register index (N + F).
    //
    // Memory layout for these tests:
    //   start_fp: 0 (raw FP = 0)
    //   Register 9 holds the new FP raw value (e.g., 200)
    //   After JAAF, FP changes and register addressing shifts accordingly.

    #[test]
    fn test_jaaf() {
        setup_tracing_with_log_level(Level::WARN);

        // JAAF: Jump to target PC and set FP from register.
        // Program: at PC=0, jump to PC=8 with new FP from reg 9.
        //
        // start_fp = 0, reg[9] = 200 (new raw FP).
        // JAAF jumps to PC=8 and sets FP to 200.
        // At PC=8 (halt), we verify FP changed.
        //
        // The program needs a padding instruction at PC=4 (skipped by jump)
        // and halt at PC=8.
        let spec = TestSpec {
            program: vec![
                wom::jaaf::<F>(8, 9), // PC=0: jump to PC=8, FP = reg[9]
                wom::halt(),          // PC=4: skipped
                                      // PC=8: halt (appended by test_spec)
            ],
            start_fp: 0,
            start_registers: vec![(9, 200)],
            expected_pc: Some(8),
            expected_fp: Some(50), // 200 / 4
            ..Default::default()
        };
        test_spec(spec)
    }

    #[test]
    fn test_jaaf_save() {
        setup_tracing_with_log_level(Level::WARN);

        // JAAF_SAVE: Jump, set FP, and save old FP to register in new frame.
        //
        // start_fp = 0, reg[9] = 200 (new raw FP).
        // jaaf_save(save_fp=11, to_pc=8, to_fp_reg=9)
        // Saves old FP (=0) to register 11 in the NEW frame (address 11*4 + 200 = 244 = abs index 61).
        // Jumps to PC=8, sets FP=200.
        let spec = TestSpec {
            program: vec![
                wom::jaaf_save::<F>(11, 8, 9), // PC=0: jump to PC=8, FP=reg[9], save old FP to new_frame[11]
                wom::halt(),                   // PC=4: skipped
                                               // PC=8: halt (appended by test_spec)
            ],
            start_fp: 0,
            start_registers: vec![(9, 200)],
            expected_pc: Some(8),
            expected_fp: Some(50), // 200 / 4
            expected_registers: vec![
                (61, 0), // abs index 61 = (11*4 + 200)/4 = old FP saved in new frame
            ],
            ..Default::default()
        };
        test_spec(spec)
    }

    #[test]
    fn test_ret() {
        setup_tracing_with_log_level(Level::WARN);

        // RET: Restore PC and FP from registers.
        //
        // start_fp = 50 (raw FP = 200).
        // reg[10] at abs index (50+10)=60 holds target PC = 8.
        // reg[11] at abs index (50+11)=61 holds target FP = 0 (raw).
        // RET reads both, jumps to PC=8, sets FP=0.
        let spec = TestSpec {
            program: vec![
                wom::ret::<F>(10, 11), // PC=0: return to PC=reg[10], FP=reg[11]
                wom::halt(),           // PC=4: skipped
                                       // PC=8: halt (appended by test_spec)
            ],
            start_fp: 50,
            start_registers: vec![
                (60, 8), // reg[10] at fp=50: target PC
                (61, 0), // reg[11] at fp=50: target FP (raw)
            ],
            expected_pc: Some(8),
            expected_fp: Some(0), // 0 / 4 = 0
            ..Default::default()
        };
        test_spec(spec)
    }

    #[test]
    fn test_call() {
        setup_tracing_with_log_level(Level::WARN);

        // CALL: Save PC and FP, then jump to immediate PC with new FP.
        //
        // start_fp = 0, reg[9] = 200 (new raw FP).
        // call(save_pc=10, save_fp=11, to_pc=8, to_fp_reg=9)
        //
        // Saves:
        //   return PC (= 0 + 4 = 4) to new_frame[10] at address (10*4 + 200 = 240) = abs index 60
        //   old FP (= 0) to new_frame[11] at address (11*4 + 200 = 244) = abs index 61
        // Jumps to PC=8, sets FP=200.
        let spec = TestSpec {
            program: vec![
                wom::call::<F>(10, 11, 8, 9), // PC=0: call to PC=8, FP=reg[9], save PC to new[10], FP to new[11]
                wom::halt(),                  // PC=4: skipped
                                              // PC=8: halt (appended by test_spec)
            ],
            start_fp: 0,
            start_registers: vec![(9, 200)],
            expected_pc: Some(8),
            expected_fp: Some(50), // 200 / 4
            expected_registers: vec![
                (60, 4), // abs index 60 = (10*4 + 200)/4 = return PC saved
                (61, 0), // abs index 61 = (11*4 + 200)/4 = old FP saved
            ],
            ..Default::default()
        };
        test_spec(spec)
    }

    #[test]
    fn test_call_indirect() {
        setup_tracing_with_log_level(Level::WARN);

        // CALL_INDIRECT: Save PC and FP, jump to register PC with new FP.
        //
        // start_fp = 0, reg[9] = 200 (new raw FP), reg[12] = 8 (target PC).
        // call_indirect(save_pc=10, save_fp=11, to_pc_reg=12, to_fp_reg=9)
        //
        // Saves:
        //   return PC (= 0 + 4 = 4) to new_frame[10] at abs index 60
        //   old FP (= 0) to new_frame[11] at abs index 61
        // Jumps to PC=reg[12]=8, sets FP=200.
        let spec = TestSpec {
            program: vec![
                wom::call_indirect::<F>(10, 11, 12, 9), // PC=0: call indirect
                wom::halt(),                            // PC=4: skipped
                                                        // PC=8: halt (appended by test_spec)
            ],
            start_fp: 0,
            start_registers: vec![
                (9, 200), // new FP (raw)
                (12, 8),  // target PC
            ],
            expected_pc: Some(8),
            expected_fp: Some(50),
            expected_registers: vec![
                (60, 4), // return PC saved at new_frame[10]
                (61, 0), // old FP saved at new_frame[11]
            ],
            ..Default::default()
        };
        test_spec(spec)
    }

    #[test]
    fn test_call_and_return() {
        setup_tracing_with_log_level(Level::WARN);

        // Complete call + return sequence using only JAAF instructions.
        //
        // Frame layout:
        //   start_fp = 0, reg[9] = 200 (new raw FP for callee frame).
        //
        // PC=0: CALL(save_pc=10, save_fp=11, to_pc=12, to_fp_reg=9)
        //   - saves return PC=4 to new_frame[10] (abs 60)
        //   - saves old FP=0 to new_frame[11] (abs 61)
        //   - jumps to PC=12, FP=200
        //
        // PC=4: (skipped on forward, executed on return)
        //   add_imm reg[0] = reg[0] + 0 (nop to land on)
        //   then halt at PC=8
        //
        // PC=12: (callee) RET(to_pc_reg=10, to_fp_reg=11)
        //   - reads new_frame[10] (abs 60) = 4 (return PC)
        //   - reads new_frame[11] (abs 61) = 0 (caller FP raw)
        //   - jumps to PC=4, FP=0
        //
        // PC=4: add_imm nop
        // PC=8: halt
        //
        // After return: FP=0, PC=8
        let spec = TestSpec {
            program: vec![
                wom::call::<F>(10, 11, 12, 9),         // PC=0: call to PC=12
                wom::add_imm::<F>(0, 0, 0_i16.into()), // PC=4: nop (return lands here)
                wom::halt(),                           // PC=8: halt after return
                wom::ret::<F>(10, 11),                 // PC=12: return to caller
            ],
            start_fp: 0,
            start_registers: vec![(9, 200)],
            expected_pc: Some(8),
            expected_fp: Some(0), // returned to original FP
            ..Default::default()
        };
        test_spec(spec)
    }
}
