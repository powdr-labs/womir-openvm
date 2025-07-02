// =================================================================================================
// RV32IM support opcodes.
// Enum types that do not start with Rv32 can be used for generic big integers, but the default
// offset is reserved for RV32IM.
//
// Create a new wrapper struct U256BaseAluOpcode(pub BaseAluOpcode) with the LocalOpcode macro to
// specify a different offset.
// =================================================================================================

use openvm_instructions::LocalOpcode;
use openvm_instructions_derive::LocalOpcode;
use serde::{Deserialize, Serialize};
use strum::{EnumCount, EnumIter, FromRepr};

#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    EnumCount,
    EnumIter,
    FromRepr,
    LocalOpcode,
    Serialize,
    Deserialize,
)]
#[opcode_offset = 0x1200]
#[repr(usize)]
pub enum BaseAluOpcode {
    ADD, // i32.add
    SUB, // i32.sub
    XOR, // i32.xor
    OR,  // i32.or
    AND, // i32.and
}

#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    EnumCount,
    EnumIter,
    FromRepr,
    LocalOpcode,
    Serialize,
    Deserialize,
)]
#[opcode_offset = 0x1205]
#[repr(usize)]
pub enum ShiftOpcode {
    SLL, // i32.shl
    SRL, // i32.shr_u
    SRA, // i32.shr_s
}

#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    EnumCount,
    EnumIter,
    FromRepr,
    LocalOpcode,
    Serialize,
    Deserialize,
)]
#[opcode_offset = 0x1208]
#[repr(usize)]
pub enum LessThanOpcode {
    SLT,  // i32.lt_s, i32.gt_s (inverted)
    SLTU, // i32.lt_u, i32.gt_u (inverted)
}

#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    EnumCount,
    EnumIter,
    FromRepr,
    LocalOpcode,
    Serialize,
    Deserialize,
)]
#[opcode_offset = 0x1210]
#[repr(usize)]
pub enum Rv32LoadStoreOpcode {
    LOADW, // i32.load
    /// LOADBU, LOADHU are unsigned extend opcodes, implemented in the same chip with LOADW
    LOADBU, // i32.load8_u
    LOADHU, // i32.load16_u
    STOREW, // i32.store
    STOREH, // i32.store16
    STOREB, // i32.store8
    /// The following are signed extend opcodes
    LOADB, // i32.load8_s
    LOADH, // i32.load16_s
}

#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    EnumCount,
    EnumIter,
    FromRepr,
    LocalOpcode,
    Serialize,
    Deserialize,
)]
#[opcode_offset = 0x1220]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum BranchEqualOpcode {
    BEQ,
    BNE,
}

#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    EnumCount,
    EnumIter,
    FromRepr,
    LocalOpcode,
    Serialize,
    Deserialize,
)]
#[opcode_offset = 0x1225]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum BranchLessThanOpcode {
    BLT,
    BLTU,
    BGE,
    BGEU,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, LocalOpcode,
)]
#[opcode_offset = 0x1230]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum Rv32JalLuiOpcode {
    JAL,
    LUI,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, LocalOpcode,
)]
#[opcode_offset = 0x1235]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum Rv32JalrOpcode {
    JALR,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, LocalOpcode,
)]
#[opcode_offset = 0x1240]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum Rv32AuipcOpcode {
    AUIPC,
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, LocalOpcode,
)]
#[opcode_offset = 0x1250]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum MulOpcode {
    MUL, // i32.mul
}

#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    EnumCount,
    EnumIter,
    FromRepr,
    LocalOpcode,
    Serialize,
    Deserialize,
)]
#[opcode_offset = 0x1251]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum MulHOpcode {
    MULH,
    MULHSU,
    MULHU,
}

#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    EnumCount,
    EnumIter,
    FromRepr,
    LocalOpcode,
    Serialize,
    Deserialize,
)]
#[opcode_offset = 0x1254]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum DivRemOpcode {
    // TODO: WASM and RISC-V are diffent in case of division by zero.
    DIV,  // i32.div_s
    DIVU, // i32.div_u
    REM,  // i32.rem_s
    REMU, // i32.rem_u
}

// =================================================================================================
// Rv32HintStore Instruction
// =================================================================================================

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, LocalOpcode,
)]
#[opcode_offset = 0x1260]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum Rv32HintStoreOpcode {
    HINT_STOREW,
    HINT_BUFFER,
}

// =================================================================================================
// Phantom opcodes
// =================================================================================================

#[derive(Copy, Clone, Debug, PartialEq, Eq, FromRepr)]
#[repr(u16)]
pub enum Rv32Phantom {
    /// Prepare the next input vector for hinting, but prepend it with a 4-byte decomposition of
    /// its length instead of one field element.
    HintInput = 0x120,
    /// Peek string from memory and print it to stdout.
    PrintStr,
    /// Prepare given amount of random numbers for hinting.
    HintRandom,
    /// Hint the VM to load values from the stream KV store into input streams.
    HintLoadByKey,
}
