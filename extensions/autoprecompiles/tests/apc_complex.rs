mod common;

use openvm_instructions::instruction::Instruction;
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use powdr_autoprecompiles::blocks::BasicBlock;
use womir_translation::instruction_builder as ib;

fn assert_machine_output(program: Vec<Instruction<BabyBear>>, test_name: &str) {
    let bb = BasicBlock {
        start_pc: 0,
        instructions: program,
    };
    common::assert_machine_output(bb.into(), "complex", test_name);
}

// ==================== LoadStore roundtrips ====================

#[test]
fn storeb_roundtrip() {
    assert_machine_output(
        vec![
            ib::storeb(0, 1, 0),
            ib::storeb(0, 1, 1),
            ib::loadbu(2, 1, 0),
            ib::loadbu(3, 1, 1),
            ib::add(4, 2, 3),
        ],
        "storeb_roundtrip",
    );
}

#[test]
fn storeh_roundtrip() {
    assert_machine_output(
        vec![
            ib::storeh(0, 2, 0),
            ib::storeh(1, 2, 2),
            ib::loadhu(3, 2, 0),
            ib::loadhu(4, 2, 2),
            ib::add(5, 3, 4),
        ],
        "storeh_roundtrip",
    );
}

#[test]
fn storeb_loadb_roundtrip() {
    assert_machine_output(
        vec![ib::storeb(0, 1, 0x2_0004), ib::loadb(2, 1, 0x2_0004)],
        "storeb_loadb_roundtrip",
    );
}

// ==================== Comparison sequences ====================

#[test]
fn comparison_chain() {
    assert_machine_output(
        vec![ib::lt_u(3, 0, 1), ib::lt_u(4, 1, 2)],
        "comparison_chain",
    );
}

#[test]
fn mixed_comparisons() {
    assert_machine_output(
        vec![ib::gt_u(2, 0, 1), ib::gt_s(5, 3, 4)],
        "mixed_comparisons",
    );
}

#[test]
fn comparison_equivalence() {
    assert_machine_output(
        vec![ib::gt_u(2, 0, 1), ib::lt_u(3, 1, 0), ib::xor(4, 2, 3)],
        "comparison_equivalence",
    );
}

#[test]
fn mixed_signed_unsigned() {
    assert_machine_output(
        vec![ib::gt_u(2, 0, 1), ib::gt_s(3, 0, 1), ib::sub(4, 2, 3)],
        "mixed_signed_unsigned",
    );
}

// ==================== Mul sequences ====================

#[test]
fn mul_commutative() {
    assert_machine_output(
        vec![ib::mul(2, 0, 1), ib::mul(3, 1, 0), ib::sub(4, 2, 3)],
        "mul_commutative",
    );
}

#[test]
fn mul_chain() {
    assert_machine_output(vec![ib::mul(3, 0, 1), ib::mul(4, 3, 2)], "mul_chain");
}

// ==================== Div sequences ====================

#[test]
fn div_chain() {
    assert_machine_output(vec![ib::div(3, 0, 1), ib::div(4, 3, 2)], "div_chain");
}

#[test]
fn div_and_mul_inverse() {
    assert_machine_output(
        vec![ib::div(2, 0, 1), ib::mul(3, 2, 1)],
        "div_and_mul_inverse",
    );
}

// ==================== Cross-width ====================

#[test]
fn cross_width_32_to_64() {
    assert_machine_output(
        vec![ib::add_imm(0, 0, 0x42_i16), ib::add_imm_64(2, 0, 0_i16)],
        "cross_width_32_to_64",
    );
}

#[test]
fn cross_width_64_to_32() {
    assert_machine_output(
        vec![ib::add_imm_64(0, 0, 0x42_i16), ib::add_imm(2, 0, 0_i16)],
        "cross_width_64_to_32",
    );
}

// ==================== I64Load patterns ====================

#[test]
fn i64load_output_plus_1_eq_base_addr() {
    assert_machine_output(
        vec![ib::loadw(0, 1, 0), ib::loadw(1, 1, 4)],
        "i64load_output_plus_1_eq_base_addr",
    );
}

#[test]
fn i64load_output_eq_base_addr() {
    assert_machine_output(
        vec![ib::loadw(1, 0, 4), ib::loadw(0, 0, 0)],
        "i64load_output_eq_base_addr",
    );
}

#[test]
fn i64load32u() {
    assert_machine_output(
        vec![ib::loadw(0, 1, 0), ib::const_32_imm(1, 0, 0)],
        "i64load32u",
    );
}
