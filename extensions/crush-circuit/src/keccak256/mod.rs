/// The AIR that handles interactions with the VM ExecutionBus and MemoryBus for handling of the
/// keccakf opcode.
pub mod keccakf_op;
/// Wrapper around the Plonky3 keccakf permutation AIR with a direct lookup bus for interaction with
/// `KeccakfOpAir`.
pub(crate) mod keccakf_perm;
/// AIR that handles the `xorin` opcode.
pub mod xorin;

pub(crate) mod constants;
pub use constants::*;
