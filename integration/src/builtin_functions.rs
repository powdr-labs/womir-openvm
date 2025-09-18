use std::{collections::HashMap, ops::RangeFrom};

use itertools::Itertools;
use openvm_stark_backend::p3_field::PrimeField32;
use wasmparser::{Operator, ValType};
use womir::loader::{
    blockless_dag::{BlocklessDag, Operation},
    flattening::WriteOnceAsm,
    Module,
};

use crate::womir_translation::{Directive, OpenVMSettings};

struct BuiltinDefinition {
    wasm_bytes: &'static [u8],
    params: &'static [ValType],
    results: &'static [ValType],
}

// TODO: add more built-in functions as needed.

const MEMORY_COPY_WASM: BuiltinDefinition = BuiltinDefinition {
    wasm_bytes: include_bytes!(env!("MEMORY_COPY_WASM_PATH")),
    params: &[ValType::I32, ValType::I32, ValType::I32],
    results: &[],
};

const MEMORY_FILL_WASM: BuiltinDefinition = BuiltinDefinition {
    wasm_bytes: include_bytes!(env!("MEMORY_FILL_WASM_PATH")),
    params: &[ValType::I32, ValType::I32, ValType::I32],
    results: &[],
};

const I32_POPCNT_WASM: BuiltinDefinition = BuiltinDefinition {
    wasm_bytes: include_bytes!(env!("I32_POPCNT_WASM_PATH")),
    params: &[ValType::I32],
    results: &[ValType::I32],
};

const I64_POPCNT_WASM: BuiltinDefinition = BuiltinDefinition {
    wasm_bytes: include_bytes!(env!("I64_POPCNT_WASM_PATH")),
    params: &[ValType::I64],
    results: &[ValType::I64],
};

const NUM_BUILTINS: usize = 4;

struct BuiltinFunction<F: PrimeField32> {
    function_index: u32,
    function_definition: WriteOnceAsm<Directive<F>>,
}

/// Tracks which built-in functions are used and provides functions to replace unsupported
/// instructions with calls to these built-in functions.
pub struct Tracker<F: PrimeField32> {
    used_builtins: HashMap<*const u8, BuiltinFunction<F>>,
}

impl<F: PrimeField32> Tracker<F> {
    pub fn new() -> Self {
        Self {
            used_builtins: HashMap::new(),
        }
    }

    /// If the given instruction is not supported natively by the VM, replace it with a call to
    /// a built-in function.
    pub fn replace_with_builtins(
        &mut self,
        module: &mut Module,
        label_gen: &mut RangeFrom<u32>,
        dag: &mut BlocklessDag,
    ) {
        for node in &mut dag.nodes {
            match &mut node.operation {
                Operation::WASMOp(op) => {
                    let builtin = match op {
                        Operator::MemoryCopy { dst_mem, src_mem } => {
                            // We don't support multi-memory modules.
                            assert_eq!(*dst_mem, 0);
                            assert_eq!(*src_mem, 0);
                            &MEMORY_COPY_WASM
                        }
                        Operator::MemoryFill { mem } => {
                            // We don't support multi-memory modules.
                            assert_eq!(*mem, 0);
                            &MEMORY_FILL_WASM
                        }
                        Operator::I32Popcnt => &I32_POPCNT_WASM,
                        Operator::I64Popcnt => &I64_POPCNT_WASM,
                        // TODO: Add more built-in functions here as needed.
                        _ => continue,
                    };

                    // Get the index of the built-in function, adding it to the module if needed.
                    let function_index = self
                        .get_builtin_function(module, label_gen, builtin)
                        .function_index;
                    // Replace the instruction with a call to the built-in function.
                    *op = Operator::Call { function_index };
                }
                Operation::Loop { sub_dag, .. } => {
                    self.replace_with_builtins(module, label_gen, sub_dag);
                }
                _ => (),
            }
        }
    }

    /// Returns an iterator over the used built-in functions to be added to the final program.
    /// Sorted by function index.
    pub fn into_used_builtins(self) -> impl Iterator<Item = WriteOnceAsm<Directive<F>>> {
        // There can't be more used built-ins than we have defined. Otherwise some built-in
        // function would have been added more than once.
        assert!(self.used_builtins.len() <= NUM_BUILTINS);

        self.used_builtins
            .into_values()
            .sorted_unstable_by_key(|bf| bf.function_index)
            .map(|bf| bf.function_definition)
    }

    fn get_builtin_function(
        &mut self,
        module: &mut Module,
        label_gen: &mut RangeFrom<u32>,
        builtin: &BuiltinDefinition,
    ) -> &BuiltinFunction<F> {
        let key = builtin.wasm_bytes.as_ptr();
        self.used_builtins.entry(key).or_insert_with(|| {
            // Add the function to the module and get an index for it.
            let function_index = module.append_function(builtin.params, builtin.results);

            // Load the built-in function module.
            let program =
                womir::loader::load_wasm(OpenVMSettings::<F>::new(), builtin.wasm_bytes).unwrap();

            // Compile the built-in function.
            let function = program.functions.into_iter().exactly_one().unwrap();
            let function_definition = function
                .advance_all_stages(
                    &OpenVMSettings::<F>::new(),
                    module,
                    function_index,
                    label_gen,
                    None,
                )
                .unwrap();

            BuiltinFunction {
                function_index,
                function_definition,
            }
        })
    }
}
