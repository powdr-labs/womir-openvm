use std::collections::{BTreeMap, BTreeSet, HashSet};

use openvm_circuit::arch::{
    AirInventory, AirInventoryError, ChipInventoryError, VmBuilder, VmCircuitConfig,
    VmCircuitExtension, VmProverExtension,
};
use openvm_circuit::system::SystemCpuBuilder;
use openvm_instructions::{LocalOpcode, SystemOpcode, VmOpcode, instruction::Instruction, riscv};
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::BabyBearPoseidon2Engine, p3_baby_bear::BabyBear,
};
use openvm_womir_transpiler::{
    BaseAlu64Opcode, BaseAluOpcode, CallOpcode, ConstOpcodes, DivRem64Opcode, DivRemOpcode,
    Eq64Opcode, EqOpcode, HintStoreOpcode, JumpOpcode, LessThan64Opcode, LessThanOpcode,
    LoadStoreOpcode, Mul64Opcode, MulOpcode, Phantom, Shift64Opcode, ShiftOpcode,
};
use powdr_openvm_common::BabyBearSC;
use powdr_openvm_common::program::OriginalCompiledProgram;
use powdr_openvm_common::{
    isa::{OpenVmISA, OriginalCpuChipComplex},
    trace_generator::cpu::periphery::{SharedPeripheryChipsCpu, SharedPeripheryChipsCpuProverExt},
};
use strum::IntoEnumIterator;
use womir_circuit::{WomirConfig, WomirConfigExecutor, WomirCpuBuilder, WomirCpuProverExt};
use womir_translation::LinkedProgram;

#[derive(Clone, Default)]
pub struct WomirISA;

fn vm_opcode_set() -> HashSet<VmOpcode> {
    let mut set = HashSet::new();
    set.extend(BaseAluOpcode::iter().map(|x| x.global_opcode()));
    set.extend(BaseAlu64Opcode::iter().map(|x| x.global_opcode()));
    set.extend(MulOpcode::iter().map(|x| x.global_opcode()));
    set.extend(Mul64Opcode::iter().map(|x| x.global_opcode()));
    set.extend(LessThanOpcode::iter().map(|x| x.global_opcode()));
    set.extend(LessThan64Opcode::iter().map(|x| x.global_opcode()));
    set.extend(DivRemOpcode::iter().map(|x| x.global_opcode()));
    set.extend(DivRem64Opcode::iter().map(|x| x.global_opcode()));
    set.extend(EqOpcode::iter().map(|x| x.global_opcode()));
    set.extend(Eq64Opcode::iter().map(|x| x.global_opcode()));
    set.extend(ShiftOpcode::iter().map(|x| x.global_opcode()));
    set.extend(Shift64Opcode::iter().map(|x| x.global_opcode()));
    set.extend(
        LoadStoreOpcode::iter()
            .take(LoadStoreOpcode::STOREB as usize + 1)
            .map(|x| x.global_opcode()),
    );
    set.extend([LoadStoreOpcode::LOADB, LoadStoreOpcode::LOADH].map(|x| x.global_opcode()));
    set.extend(JumpOpcode::iter().map(|x| x.global_opcode()));
    set.extend(CallOpcode::iter().map(|x| x.global_opcode()));
    set.extend(ConstOpcodes::iter().map(|x| x.global_opcode()));
    set
}

fn branch_opcodes() -> HashSet<VmOpcode> {
    let mut set = HashSet::new();
    set.extend(JumpOpcode::iter().map(|x| x.global_opcode()));
    set.extend(CallOpcode::iter().map(|x| x.global_opcode()));
    set
}

fn opcode_is_in<I>(opcode: VmOpcode) -> bool
where
    I: IntoEnumIterator + LocalOpcode,
{
    I::iter().any(|op| op.global_opcode() == opcode)
}

fn is_alu_like(opcode: VmOpcode) -> bool {
    opcode_is_in::<BaseAluOpcode>(opcode)
        || opcode_is_in::<BaseAlu64Opcode>(opcode)
        || opcode_is_in::<MulOpcode>(opcode)
        || opcode_is_in::<Mul64Opcode>(opcode)
        || opcode_is_in::<LessThanOpcode>(opcode)
        || opcode_is_in::<LessThan64Opcode>(opcode)
        || opcode_is_in::<DivRemOpcode>(opcode)
        || opcode_is_in::<DivRem64Opcode>(opcode)
        || opcode_is_in::<EqOpcode>(opcode)
        || opcode_is_in::<Eq64Opcode>(opcode)
        || opcode_is_in::<ShiftOpcode>(opcode)
        || opcode_is_in::<Shift64Opcode>(opcode)
}

fn reg_name(ptr: u32) -> String {
    format!("r{}", ptr / riscv::RV32_REGISTER_NUM_LIMBS as u32)
}

fn decode_alu_imm(c: u32) -> i32 {
    let imm24 = c & 0x00ff_ffff;
    if (imm24 & 0x0080_0000) != 0 {
        (imm24 | 0xff00_0000) as i32
    } else {
        imm24 as i32
    }
}

fn opcode_name(opcode: VmOpcode) -> String {
    let opcode_usize = opcode.as_usize();

    for op in BaseAluOpcode::iter() {
        if op.global_opcode().as_usize() == opcode_usize {
            return format!("{op:?}");
        }
    }
    for op in BaseAlu64Opcode::iter() {
        if op.global_opcode().as_usize() == opcode_usize {
            return format!("{op:?}_64");
        }
    }
    for op in MulOpcode::iter() {
        if op.global_opcode().as_usize() == opcode_usize {
            return format!("{op:?}");
        }
    }
    for op in Mul64Opcode::iter() {
        if op.global_opcode().as_usize() == opcode_usize {
            return format!("{op:?}_64");
        }
    }
    for op in LessThanOpcode::iter() {
        if op.global_opcode().as_usize() == opcode_usize {
            return format!("{op:?}");
        }
    }
    for op in LessThan64Opcode::iter() {
        if op.global_opcode().as_usize() == opcode_usize {
            return format!("{op:?}_64");
        }
    }
    for op in DivRemOpcode::iter() {
        if op.global_opcode().as_usize() == opcode_usize {
            return format!("{op:?}");
        }
    }
    for op in DivRem64Opcode::iter() {
        if op.global_opcode().as_usize() == opcode_usize {
            return format!("{op:?}_64");
        }
    }
    for op in EqOpcode::iter() {
        if op.global_opcode().as_usize() == opcode_usize {
            return format!("{op:?}");
        }
    }
    for op in Eq64Opcode::iter() {
        if op.global_opcode().as_usize() == opcode_usize {
            return format!("{op:?}_64");
        }
    }
    for op in ShiftOpcode::iter() {
        if op.global_opcode().as_usize() == opcode_usize {
            return format!("{op:?}");
        }
    }
    for op in Shift64Opcode::iter() {
        if op.global_opcode().as_usize() == opcode_usize {
            return format!("{op:?}_64");
        }
    }
    for op in LoadStoreOpcode::iter() {
        if op.global_opcode().as_usize() == opcode_usize {
            return format!("{op:?}");
        }
    }
    for op in JumpOpcode::iter() {
        if op.global_opcode().as_usize() == opcode_usize {
            return format!("{op:?}");
        }
    }
    for op in CallOpcode::iter() {
        if op.global_opcode().as_usize() == opcode_usize {
            return format!("{op:?}");
        }
    }
    for op in HintStoreOpcode::iter() {
        if op.global_opcode().as_usize() == opcode_usize {
            return format!("{op:?}");
        }
    }
    for op in ConstOpcodes::iter() {
        if op.global_opcode().as_usize() == opcode_usize {
            return format!("{op:?}");
        }
    }

    format!("<opcode {}>", opcode.as_usize())
}

fn womir_instruction_formatter<F: PrimeField32>(instruction: &Instruction<F>) -> String {
    let Instruction {
        opcode,
        a,
        b,
        c,
        d,
        e,
        f,
        g,
    } = instruction;

    let name = opcode_name(*opcode);
    let a = a.as_canonical_u32();
    let b = b.as_canonical_u32();
    let c = c.as_canonical_u32();
    let d = d.as_canonical_u32();
    let e = e.as_canonical_u32();
    let f = f.as_canonical_u32();
    let g = g.as_canonical_u32();

    if is_alu_like(*opcode) {
        let rd = reg_name(a);
        let rs1 = reg_name(b);
        if e == 1 {
            return format!("{name} {rd}, {rs1}, {}", reg_name(c));
        }
        return format!("{name} {rd}, {rs1}, {}", decode_alu_imm(c));
    }

    if *opcode == ConstOpcodes::CONST32.global_opcode() {
        let imm = ((c & 0xffff) << 16) | (b & 0xffff);
        return format!("{name} {}, 0x{imm:08x}", reg_name(a));
    }

    if opcode_is_in::<LoadStoreOpcode>(*opcode) {
        let imm = ((g & 0xffff) << 16) | (c & 0xffff);
        if matches!(
            *opcode,
            x if x == LoadStoreOpcode::LOADW.global_opcode()
                || x == LoadStoreOpcode::LOADB.global_opcode()
                || x == LoadStoreOpcode::LOADBU.global_opcode()
                || x == LoadStoreOpcode::LOADH.global_opcode()
                || x == LoadStoreOpcode::LOADHU.global_opcode()
        ) {
            return format!("{name} {}, [{} + {:#x}]", reg_name(a), reg_name(b), imm);
        }
        return format!("{name} [{} + {:#x}], {}", reg_name(b), imm, reg_name(a));
    }

    if *opcode == CallOpcode::RET.global_opcode() {
        return format!("{name} to_pc={}, to_fp={}", reg_name(c), reg_name(d));
    }
    if *opcode == CallOpcode::CALL.global_opcode() {
        return format!(
            "{name} save_pc={}, save_fp={}, target_pc={}, fp_offset={}",
            reg_name(a),
            reg_name(b),
            c,
            d
        );
    }
    if *opcode == CallOpcode::CALL_INDIRECT.global_opcode() {
        return format!(
            "{name} save_pc={}, save_fp={}, target_pc={}, fp_offset={}",
            reg_name(a),
            reg_name(b),
            reg_name(c),
            d
        );
    }

    if *opcode == JumpOpcode::JUMP.global_opcode() {
        return format!("{name} target_pc={a}");
    }
    if *opcode == JumpOpcode::SKIP.global_opcode() {
        return format!("{name} offset_reg={}", reg_name(b));
    }
    if *opcode == JumpOpcode::JUMP_IF.global_opcode()
        || *opcode == JumpOpcode::JUMP_IF_ZERO.global_opcode()
    {
        return format!("{name} cond={}, target_pc={a}", reg_name(b));
    }

    if *opcode == HintStoreOpcode::HINT_STOREW.global_opcode() {
        return format!("{name} mem_ptr={}", reg_name(b));
    }
    if *opcode == HintStoreOpcode::HINT_BUFFER.global_opcode() {
        return format!("{name} words={}, mem_ptr={}", reg_name(a), reg_name(b));
    }

    if *opcode == SystemOpcode::TERMINATE.global_opcode() {
        return format!("{name} exit_code={c}");
    }
    if *opcode == SystemOpcode::PHANTOM.global_opcode() {
        let phantom = (c & 0xffff) as u16;
        let phantom_name = match Phantom::from_repr(phantom) {
            Some(v) => format!("{v:?}"),
            None => format!("Unknown({phantom})"),
        };
        let extra = c >> 16;
        return format!(
            "{name} phantom={phantom_name}, a={a}, b={b}, extra={extra}, d={d}, e={e}, f={f}, g={g}"
        );
    }

    format!("{name} {a} {b} {c} {d} {e} {f} {g}")
}

impl OpenVmISA for WomirISA {
    type Program<'a> = LinkedProgram<'a, BabyBear>;
    type RegisterAddress = ();
    type OriginalExecutor<F: PrimeField32> = WomirConfigExecutor<F>;
    type OriginalConfig = WomirConfig;
    type OriginalBuilderCpu = WomirCpuBuilder;

    fn create_original_chip_complex(
        config: &Self::OriginalConfig,
        airs: AirInventory<BabyBearSC>,
    ) -> Result<OriginalCpuChipComplex, ChipInventoryError> {
        <WomirCpuBuilder as VmBuilder<BabyBearPoseidon2Engine>>::create_chip_complex(
            &WomirCpuBuilder,
            config,
            airs,
        )
    }

    fn is_branching(opcode: VmOpcode) -> bool {
        branch_opcodes().contains(&opcode)
    }

    fn instruction_allowlist() -> HashSet<VmOpcode> {
        vm_opcode_set()
    }

    fn get_register_value(_: &Self::RegisterAddress) -> u32 {
        unimplemented!("execution constraints are currently unused")
    }

    fn value_limb(_: u32, _: usize) -> u32 {
        unimplemented!("execution constraints are currently unused")
    }

    fn format<F: PrimeField32>(instruction: &Instruction<F>) -> String {
        womir_instruction_formatter(instruction)
    }

    fn get_labels_debug<'a>(program: &Self::Program<'a>) -> BTreeMap<u64, Vec<String>> {
        program.labels()
    }

    fn get_jump_destinations(original_program: &OriginalCompiledProgram<Self>) -> BTreeSet<u64> {
        original_program.elf.labels().into_keys().collect()
    }

    fn create_dummy_airs<E: VmCircuitExtension<BabyBearSC>>(
        config: &Self::OriginalConfig,
        shared_chips: E,
    ) -> Result<AirInventory<BabyBearSC>, AirInventoryError> {
        let mut inventory = config.system.create_airs()?;
        inventory.start_new_extension();
        VmCircuitExtension::extend_circuit(&shared_chips, &mut inventory)?;
        VmCircuitExtension::extend_circuit(&config.base, &mut inventory)?;
        Ok(inventory)
    }

    fn create_dummy_chip_complex_cpu(
        config: &Self::OriginalConfig,
        circuit: AirInventory<BabyBearSC>,
        shared_chips: SharedPeripheryChipsCpu<Self>,
    ) -> Result<OriginalCpuChipComplex, ChipInventoryError> {
        let mut chip_complex = VmBuilder::<BabyBearPoseidon2Engine>::create_chip_complex(
            &SystemCpuBuilder,
            &config.system,
            circuit,
        )?;
        let inventory = &mut chip_complex.inventory;

        VmProverExtension::<BabyBearPoseidon2Engine, _, _>::extend_prover(
            &SharedPeripheryChipsCpuProverExt,
            &shared_chips,
            inventory,
        )?;

        VmProverExtension::<BabyBearPoseidon2Engine, _, _>::extend_prover(
            &WomirCpuProverExt,
            &config.base,
            inventory,
        )?;

        Ok(chip_complex)
    }

    #[cfg(feature = "cuda")]
    fn create_dummy_chip_complex_gpu(
        config: &Self::OriginalConfig,
        circuit: AirInventory<BabyBearSC>,
        shared_chips: SharedPeripheryChipsGpu<Self>,
    ) -> Result<OriginalGpuChipComplex, ChipInventoryError> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use powdr_openvm_common::extraction_utils::OriginalVmConfig;

    use super::*;

    #[test]
    fn machine_extraction() {
        let powdr_config = powdr_openvm_common::default_powdr_openvm_config(0, 0);
        let original_config = OriginalVmConfig::<WomirISA>::new(WomirConfig::default());
        let _ = original_config.airs(powdr_config.degree_bound).unwrap();
    }
}
