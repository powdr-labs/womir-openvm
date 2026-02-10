/// Creates a newtype wrapper around an OpenVM executor type.
///
/// This macro generates:
/// - A newtype struct (with `Clone`, `Copy`) wrapping the inner executor
/// - A `new()` constructor that forwards to the inner type
/// - A `Deref` implementation for easy access to inner methods
/// - A `PreflightExecutor` implementation that delegates to the inner type
///
/// # Example
///
/// ```ignore
/// executor_newtype! {
///     pub struct BaseAluExecutor(pub BaseAluExecutorInner)
///         <A, const NUM_LIMBS: usize, const LIMB_BITS: usize>;
///     new(adapter, offset: usize) => BaseAluExecutorInner::new(adapter, offset)
/// }
/// ```
#[macro_export]
macro_rules! executor_newtype {
    (
        $vis:vis struct $name:ident($inner_vis:vis $inner:ident)
            <A $(, const $const_name:ident: $const_ty:ty)*>;
        new($adapter:ident $(, $arg:ident: $arg_ty:ty)*) => $inner_new:expr
    ) => {
        #[derive(Clone, Copy)]
        $vis struct $name<A $(, const $const_name: $const_ty)*>(
            $inner_vis $inner<A $(, $const_name)*>,
        );

        impl<A $(, const $const_name: $const_ty)*> $name<A $(, $const_name)*> {
            pub fn new($adapter: A $(, $arg: $arg_ty)*) -> Self {
                Self($inner_new)
            }
        }

        impl<A $(, const $const_name: $const_ty)*> std::ops::Deref
            for $name<A $(, $const_name)*>
        {
            type Target = $inner<A $(, $const_name)*>;

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl<F, A, RA $(, const $const_name: $const_ty)*>
            $crate::macros::PreflightExecutor<F, RA> for $name<A $(, $const_name)*>
        where
            F: $crate::macros::PrimeField32,
            $inner<A $(, $const_name)*>: $crate::macros::PreflightExecutor<F, RA>,
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
