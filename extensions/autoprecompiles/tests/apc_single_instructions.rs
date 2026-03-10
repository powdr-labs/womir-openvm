mod common;

use openvm_instructions::instruction::Instruction;
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use powdr_autoprecompiles::blocks::BasicBlock;
use womir_translation::instruction_builder::{self as ib, AluImm};

fn assert_machine_output(program: Vec<Instruction<BabyBear>>, test_name: &str) {
    let bb = BasicBlock {
        start_pc: 0,
        instructions: program,
    };
    common::assert_machine_output(bb.into(), "single_instructions", test_name);
}

// ==================== BaseAlu 32-bit ====================

#[test]
fn single_add() {
    assert_machine_output(vec![ib::add(2, 0, 1)], "add");
}

#[test]
fn single_add_imm() {
    assert_machine_output(vec![ib::add_imm(1, 0, 100_i16)], "add_imm");
}

#[test]
fn single_sub() {
    assert_machine_output(vec![ib::sub(2, 0, 1)], "sub");
}

#[test]
fn single_xor() {
    assert_machine_output(vec![ib::xor(2, 0, 1)], "xor");
}

#[test]
fn single_or() {
    assert_machine_output(vec![ib::or(2, 0, 1)], "or");
}

#[test]
fn single_and() {
    assert_machine_output(vec![ib::and(2, 0, 1)], "and");
}

#[test]
fn single_and_imm() {
    assert_machine_output(vec![ib::and_imm(1, 0, 0xFF_i16)], "and_imm");
}

// ==================== BaseAlu 64-bit ====================

#[test]
fn single_add_64() {
    assert_machine_output(vec![ib::add_64(4, 0, 2)], "add_64");
}

#[test]
fn single_add_imm_64() {
    assert_machine_output(vec![ib::add_imm_64(2, 0, 42_i16)], "add_imm_64");
}

#[test]
fn single_sub_64() {
    assert_machine_output(vec![ib::sub_64(4, 0, 2)], "sub_64");
}

#[test]
fn single_sub_imm_64() {
    assert_machine_output(vec![ib::sub_imm_64(2, 0, 42_i16)], "sub_imm_64");
}

#[test]
fn single_xor_64() {
    assert_machine_output(vec![ib::xor_64(4, 0, 2)], "xor_64");
}

#[test]
fn single_xor_imm_64() {
    assert_machine_output(vec![ib::xor_imm_64(2, 0, 0xFF_i16)], "xor_imm_64");
}

#[test]
fn single_or_64() {
    assert_machine_output(vec![ib::or_64(4, 0, 2)], "or_64");
}

#[test]
fn single_or_imm_64() {
    assert_machine_output(vec![ib::or_imm_64(2, 0, 0xFF_i16)], "or_imm_64");
}

#[test]
fn single_and_64() {
    assert_machine_output(vec![ib::and_64(4, 0, 2)], "and_64");
}

#[test]
fn single_and_imm_64() {
    assert_machine_output(vec![ib::and_imm_64(2, 0, 0xFF_i16)], "and_imm_64");
}

// ==================== Mul ====================

#[test]
fn single_mul() {
    assert_machine_output(vec![ib::mul(2, 0, 1)], "mul");
}

#[test]
fn single_mul_imm() {
    assert_machine_output(vec![ib::mul_imm(1, 0, 5_i16)], "mul_imm");
}

#[test]
fn single_mul_64() {
    assert_machine_output(vec![ib::mul_64(4, 0, 2)], "mul_64");
}

#[test]
fn single_mul_imm_64() {
    assert_machine_output(
        vec![ib::mul_imm_64(2, 0, AluImm::from(5_i16))],
        "mul_imm_64",
    );
}

// ==================== LessThan 32-bit ====================

#[test]
fn single_lt_u() {
    assert_machine_output(vec![ib::lt_u(2, 0, 1)], "lt_u");
}

#[test]
fn single_lt_u_imm() {
    assert_machine_output(vec![ib::lt_u_imm(1, 0, 100_i16)], "lt_u_imm");
}

#[test]
fn single_lt_s() {
    assert_machine_output(vec![ib::lt_s(2, 0, 1)], "lt_s");
}

// ==================== LessThan 64-bit ====================

#[test]
fn single_lt_u_64() {
    assert_machine_output(vec![ib::lt_u_64(4, 0, 2)], "lt_u_64");
}

#[test]
fn single_lt_s_64() {
    assert_machine_output(vec![ib::lt_s_64(4, 0, 2)], "lt_s_64");
}

// ==================== Eq 32-bit ====================

#[test]
fn single_eq() {
    assert_machine_output(vec![ib::eq(2, 0, 1)], "eq");
}

#[test]
fn single_eq_imm() {
    assert_machine_output(vec![ib::eq_imm(1, 0, 42_i16)], "eq_imm");
}

#[test]
fn single_neq() {
    assert_machine_output(vec![ib::neq(2, 0, 1)], "neq");
}

#[test]
fn single_neq_imm() {
    assert_machine_output(vec![ib::neq_imm(1, 0, 42_i16)], "neq_imm");
}

// ==================== Eq 64-bit ====================

#[test]
fn single_eq_64() {
    assert_machine_output(vec![ib::eq_64(4, 0, 2)], "eq_64");
}

#[test]
fn single_eq_imm_64() {
    assert_machine_output(vec![ib::eq_imm_64(2, 0, 42_i16)], "eq_imm_64");
}

#[test]
fn single_neq_64() {
    assert_machine_output(vec![ib::neq_64(4, 0, 2)], "neq_64");
}

#[test]
fn single_neq_imm_64() {
    assert_machine_output(vec![ib::neq_imm_64(2, 0, 42_i16)], "neq_imm_64");
}

// ==================== Shift 32-bit ====================

#[test]
fn single_shl() {
    assert_machine_output(vec![ib::shl(2, 0, 1)], "shl");
}

#[test]
fn single_shl_imm() {
    assert_machine_output(vec![ib::shl_imm(2, 0, 3_i16)], "shl_imm");
}

#[test]
fn single_shr_u() {
    assert_machine_output(vec![ib::shr_u(2, 0, 1)], "shr_u");
}

#[test]
fn single_shr_u_imm() {
    assert_machine_output(vec![ib::shr_u_imm(2, 0, 3_i16)], "shr_u_imm");
}

#[test]
fn single_shr_s_imm() {
    assert_machine_output(vec![ib::shr_s_imm(2, 0, 3_i16)], "shr_s_imm");
}

// ==================== Shift 64-bit ====================

#[test]
fn single_shl_64() {
    assert_machine_output(vec![ib::shl_64(4, 0, 2)], "shl_64");
}

#[test]
fn single_shl_imm_64() {
    assert_machine_output(vec![ib::shl_imm_64(2, 0, 3_i16)], "shl_imm_64");
}

#[test]
fn single_shr_u_64() {
    assert_machine_output(vec![ib::shr_u_64(4, 0, 2)], "shr_u_64");
}

#[test]
fn single_shr_u_imm_64() {
    assert_machine_output(vec![ib::shr_u_imm_64(2, 0, 3_i16)], "shr_u_imm_64");
}

#[test]
fn single_shr_s_imm_64() {
    assert_machine_output(vec![ib::shr_s_imm_64(2, 0, 3_i16)], "shr_s_imm_64");
}

// ==================== DivRem 32-bit ====================

#[test]
fn single_div() {
    assert_machine_output(vec![ib::div(2, 0, 1)], "div");
}

#[test]
fn single_divu() {
    assert_machine_output(vec![ib::divu(2, 0, 1)], "divu");
}

#[test]
fn single_rems() {
    assert_machine_output(vec![ib::rems(2, 0, 1)], "rems");
}

#[test]
fn single_remu() {
    assert_machine_output(vec![ib::remu(2, 0, 1)], "remu");
}

#[test]
fn single_div_imm() {
    assert_machine_output(vec![ib::div_imm(1, 0, 7_i16)], "div_imm");
}

#[test]
fn single_divu_imm() {
    assert_machine_output(vec![ib::divu_imm(1, 0, 7_i16)], "divu_imm");
}

#[test]
fn single_rems_imm() {
    assert_machine_output(vec![ib::rems_imm(1, 0, 7_i16)], "rems_imm");
}

#[test]
fn single_remu_imm() {
    assert_machine_output(vec![ib::remu_imm(1, 0, 7_i16)], "remu_imm");
}

// ==================== DivRem 64-bit ====================

#[test]
fn single_div_64() {
    assert_machine_output(vec![ib::div_64(4, 0, 2)], "div_64");
}

#[test]
fn single_divu_64() {
    assert_machine_output(vec![ib::divu_64(4, 0, 2)], "divu_64");
}

#[test]
fn single_rems_64() {
    assert_machine_output(vec![ib::rems_64(4, 0, 2)], "rems_64");
}

#[test]
fn single_remu_64() {
    assert_machine_output(vec![ib::remu_64(4, 0, 2)], "remu_64");
}

#[test]
fn single_div_imm_64() {
    assert_machine_output(vec![ib::div_imm_64(2, 0, 7_i16)], "div_imm_64");
}

#[test]
fn single_divu_imm_64() {
    assert_machine_output(vec![ib::divu_imm_64(2, 0, 7_i16)], "divu_imm_64");
}

#[test]
fn single_rems_imm_64() {
    assert_machine_output(vec![ib::rems_imm_64(2, 0, 7_i16)], "rems_imm_64");
}

#[test]
fn single_remu_imm_64() {
    assert_machine_output(vec![ib::remu_imm_64(2, 0, 7_i16)], "remu_imm_64");
}

// ==================== LoadStore ====================

#[test]
fn single_loadw() {
    assert_machine_output(vec![ib::loadw(1, 0, 0)], "loadw");
}

#[test]
fn single_loadw_with_offset() {
    assert_machine_output(vec![ib::loadw(1, 0, 8)], "loadw_with_offset");
}

#[test]
fn single_storew() {
    assert_machine_output(vec![ib::storew(0, 1, 0)], "storew");
}

#[test]
fn single_storew_with_offset() {
    assert_machine_output(vec![ib::storew(0, 1, 4)], "storew_with_offset");
}

#[test]
fn single_loadbu() {
    assert_machine_output(vec![ib::loadbu(1, 0, 0)], "loadbu");
}

#[test]
fn single_loadhu() {
    assert_machine_output(vec![ib::loadhu(1, 0, 0)], "loadhu");
}

#[test]
fn single_storeb() {
    assert_machine_output(vec![ib::storeb(0, 1, 0)], "storeb");
}

#[test]
fn single_storeh() {
    assert_machine_output(vec![ib::storeh(0, 1, 0)], "storeh");
}

#[test]
fn single_loadb() {
    assert_machine_output(vec![ib::loadb(1, 0, 0)], "loadb");
}

#[test]
fn single_loadh() {
    assert_machine_output(vec![ib::loadh(1, 0, 0)], "loadh");
}

// ==================== Jump ====================

#[test]
fn single_jump() {
    assert_machine_output(vec![ib::jump(8)], "jump");
}

#[test]
fn single_jump_if() {
    assert_machine_output(vec![ib::jump_if(2, 8)], "jump_if");
}

#[test]
fn single_jump_if_zero() {
    assert_machine_output(vec![ib::jump_if_zero(2, 8)], "jump_if_zero");
}

#[test]
fn single_skip() {
    assert_machine_output(vec![ib::skip(2)], "skip");
}

// ==================== Call / Ret ====================

#[test]
fn single_call() {
    assert_machine_output(vec![ib::call(10, 11, 12, 120)], "call");
}

#[test]
fn single_call_indirect() {
    assert_machine_output(vec![ib::call_indirect(10, 11, 12, 120)], "call_indirect");
}

#[test]
fn single_ret() {
    assert_machine_output(vec![ib::ret(10, 11)], "ret");
}

// ==================== Const ====================

#[test]
fn single_const_32() {
    assert_machine_output(vec![ib::const_32_imm(1, 42, 0)], "const_32");
}
