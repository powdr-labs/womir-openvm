use openvm_instructions::LocalOpcode;
use openvm_instructions_derive::LocalOpcode;
use serde::{Deserialize, Serialize};
use strum::{EnumCount, EnumIter, FromRepr};

pub use openvm_rv32im_transpiler::{
    BaseAluOpcode, DivRemOpcode, LessThanOpcode, MulOpcode, Rv32LoadStoreOpcode as LoadStoreOpcode,
    ShiftOpcode,
};

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
#[opcode_offset = 0x2200]
#[repr(usize)]
// Note: these need to be exactly the same and in the exact same order as OpenVM's BaseAluOpcode
// in order to be able to re-use the original Alu core chip.
// We do re-use the `BaseAluOpcode` type for 32-bit operations, but need a new opcode/enum for
// 64-bit ops.
pub enum BaseAlu64Opcode {
    ADD,
    SUB,
    XOR,
    OR,
    AND,
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
#[opcode_offset = 0x2205]
#[repr(usize)]
// Note: these need to be exactly the same and in the exact same order as OpenVM's ShiftOpcode
// in order to be able to re-use the original Shift core chip.
// We do re-use the `BaseAluOpcode` type for 32-bit operations, but need a new opcode/enum for
// 64-bit ops.
pub enum Shift64Opcode {
    SLL,
    SRL,
    SRA,
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
#[opcode_offset = 0x2208]
#[repr(usize)]
// Note: these need to be exactly the same and in the exact same order as OpenVM's LessThanOpcode
// in order to be able to re-use the original LessThan core chip.
// We do re-use the `BaseAluOpcode` type for 32-bit operations, but need a new opcode/enum for
// 64-bit ops.
pub enum LessThan64Opcode {
    SLT,
    SLTU,
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
#[opcode_offset = 0x120c]
#[repr(usize)]
pub enum EqOpcode {
    EQ,
    NEQ,
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
#[opcode_offset = 0x220c]
#[repr(usize)]
pub enum Eq64Opcode {
    EQ,
    NEQ,
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
#[opcode_offset = 0x1236]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum JaafOpcode {
    JAAF,          // jump and activate frame
    JAAF_SAVE,     // jump and activate frame, save fp
    RET,           // return to pc and restore frame
    CALL,          // call function, save pc and fp
    CALL_INDIRECT, // call function indirect, save pc and fp
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
#[opcode_offset = 0x123B]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum JumpOpcode {
    JUMP,         // unconditional jump to immediate PC
    SKIP,         // unconditional jump to current PC + offset
    JUMP_IF,      // conditional jump to immediate PC if condition register != 0
    JUMP_IF_ZERO, // conditional jump to immediate PC if condition register == 0
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, LocalOpcode,
)]
#[opcode_offset = 0x2250]
#[repr(usize)]
#[allow(non_camel_case_types)]
// Note: these need to be exactly the same and in the exact same order as OpenVM's MulOpcode
// in order to be able to re-use the original Mul core chip.
// We do re-use the `BaseAluOpcode` type for 32-bit operations, but need a new opcode/enum for
// 64-bit ops.
pub enum Mul64Opcode {
    MUL,
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
#[opcode_offset = 0x2254]
#[repr(usize)]
#[allow(non_camel_case_types)]
// Note: these need to be exactly the same and in the exact same order as OpenVM's DivRemOpcode
// in order to be able to re-use the original DivRem core chip.
// We do re-use the `BaseAluOpcode` type for 32-bit operations, but need a new opcode/enum for
// 64-bit ops.
pub enum DivRem64Opcode {
    DIV,
    DIVU,
    REM,
    REMU,
}

// =================================================================================================
// HintStore Instruction
// =================================================================================================

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, LocalOpcode,
)]
#[opcode_offset = 0x1260]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum HintStoreOpcode {
    HINT_STOREW,
    HINT_BUFFER,
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
#[opcode_offset = 0x1265]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum AllocateFrameOpcode {
    ALLOCATE_FRAME, // allocate frame and return pointer
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
#[opcode_offset = 0x126A]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum CopyIntoFrameOpcode {
    COPY_INTO_FRAME, // copy value into frame-relative address
    COPY_FROM_FRAME, // copy value from another frame into the current frame
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
#[opcode_offset = 0x127A]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum RegWriteOpcode {
    /// stores an immediate into a register
    CONST32,
    /// stores an immediate into the least significant field element of a register.
    /// Used for storing an immediate as a PC/FP value.
    CONST_FIELD,
    /// Copy value from one register to another.
    COPY_REG,
}

// =================================================================================================
// Phantom opcodes
// =================================================================================================

#[derive(Copy, Clone, Debug, PartialEq, Eq, FromRepr)]
#[repr(u16)]
pub enum Phantom {
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
