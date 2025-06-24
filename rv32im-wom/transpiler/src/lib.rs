use std::marker::PhantomData;

use openvm_instructions::{
    instruction::Instruction, riscv::RV32_REGISTER_NUM_LIMBS, LocalOpcode, PhantomDiscriminant,
    SystemOpcode,
};
use openvm_stark_backend::p3_field::PrimeField32;

mod instructions;
pub use instructions::*;
