/// Creates a newtype wrapper around an OpenVM executor type.
///
/// This macro generates:
/// - A newtype struct wrapping the inner executor
/// - A `new()` constructor that forwards to the inner type
/// - A `Deref` implementation for easy access to inner methods
/// - A `PreflightExecutor` implementation that delegates to the inner type
///
/// # Example
///
/// ```ignore
/// executor_newtype! {
///     /// Newtype wrapper to satisfy orphan rules.
///     #[derive(Clone, Copy)]
///     pub struct BaseAluExecutor<A, const NUM_LIMBS: usize, const LIMB_BITS: usize>(
///         pub BaseAluExecutorInner<A, NUM_LIMBS, LIMB_BITS>
///     );
///     new(adapter, offset: usize) => BaseAluExecutorInner::new(adapter, offset)
/// }
/// ```
#[macro_export]
macro_rules! executor_newtype {
    // Two const generics, extra constructor args
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident<A, const $const1:ident: $const1_ty:ty, const $const2:ident: $const2_ty:ty>(
            $inner_vis:vis $inner:ident<A, $const1_use:ident, $const2_use:ident>
        );
        new($adapter:ident $(, $arg:ident: $arg_ty:ty)*) => $inner_new:expr
    ) => {
        $(#[$meta])*
        $vis struct $name<A, const $const1: $const1_ty, const $const2: $const2_ty>(
            $inner_vis $inner<A, $const1, $const2>,
        );

        impl<A, const $const1: $const1_ty, const $const2: $const2_ty> $name<A, $const1, $const2> {
            pub fn new($adapter: A $(, $arg: $arg_ty)*) -> Self {
                Self($inner_new)
            }
        }

        impl<A, const $const1: $const1_ty, const $const2: $const2_ty> std::ops::Deref
            for $name<A, $const1, $const2>
        {
            type Target = $inner<A, $const1, $const2>;

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl<F, A, RA, const $const1: $const1_ty, const $const2: $const2_ty>
            $crate::macros::PreflightExecutor<F, RA> for $name<A, $const1, $const2>
        where
            F: $crate::macros::PrimeField32,
            $inner<A, $const1, $const2>: $crate::macros::PreflightExecutor<F, RA>,
        {
            fn get_opcode_name(&self, opcode: usize) -> String {
                self.0.get_opcode_name(opcode)
            }

            fn execute(
                &self,
                state: $crate::macros::VmStateMut<F, $crate::macros::TracingMemory, RA>,
                instruction: &$crate::macros::Instruction<F>,
            ) -> Result<(), $crate::macros::ExecutionError> {
                self.0.execute(state, instruction)
            }
        }
    };

    // One const generic, extra constructor args
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident<A, const $const1:ident: $const1_ty:ty>(
            $inner_vis:vis $inner:ident<A, $const1_use:ident>
        );
        new($adapter:ident $(, $arg:ident: $arg_ty:ty)*) => $inner_new:expr
    ) => {
        $(#[$meta])*
        $vis struct $name<A, const $const1: $const1_ty>(
            $inner_vis $inner<A, $const1>,
        );

        impl<A, const $const1: $const1_ty> $name<A, $const1> {
            pub fn new($adapter: A $(, $arg: $arg_ty)*) -> Self {
                Self($inner_new)
            }
        }

        impl<A, const $const1: $const1_ty> std::ops::Deref for $name<A, $const1> {
            type Target = $inner<A, $const1>;

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl<F, A, RA, const $const1: $const1_ty> $crate::macros::PreflightExecutor<F, RA>
            for $name<A, $const1>
        where
            F: $crate::macros::PrimeField32,
            $inner<A, $const1>: $crate::macros::PreflightExecutor<F, RA>,
        {
            fn get_opcode_name(&self, opcode: usize) -> String {
                self.0.get_opcode_name(opcode)
            }

            fn execute(
                &self,
                state: $crate::macros::VmStateMut<F, $crate::macros::TracingMemory, RA>,
                instruction: &$crate::macros::Instruction<F>,
            ) -> Result<(), $crate::macros::ExecutionError> {
                self.0.execute(state, instruction)
            }
        }
    };
}

// Re-export types needed by the macro
pub use openvm_circuit::arch::{ExecutionError, PreflightExecutor, VmStateMut};
pub use openvm_circuit::system::memory::online::TracingMemory;
pub use openvm_instructions::instruction::Instruction;
pub use openvm_stark_backend::p3_field::PrimeField32;
