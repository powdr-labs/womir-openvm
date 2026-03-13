use openvm_instructions::{LocalOpcode, SystemOpcode, VmOpcode, instruction::Instruction, riscv};
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_wasm_transpiler::*;
use strum::IntoEnumIterator;

pub fn crush_instruction_formatter<F: PrimeField32>(instruction: &Instruction<F>) -> String {
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
