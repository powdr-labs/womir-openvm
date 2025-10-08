use std::{
    collections::{HashMap, hash_map::Entry},
    sync::{Mutex, atomic::AtomicU32},
};

use itertools::Itertools;
use openvm_stark_backend::p3_field::PrimeField32;
use wasmparser::{Operator, ValType};
use womir::loader::{
    blockless_dag::{BlocklessDag, Operation},
    flattening::WriteOnceAsm,
};

use crate::womir_translation::{Directive, OpenVMSettings};

pub struct BuiltinDefinition {
    wasm_bytes: &'static [u8],
    pub params: &'static [ValType],
    pub results: &'static [ValType],
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

const I32_CTZ_WASM: BuiltinDefinition = BuiltinDefinition {
    wasm_bytes: include_bytes!(env!("I32_CTZ_WASM_PATH")),
    params: &[ValType::I32],
    results: &[ValType::I32],
};

const I64_CTZ_WASM: BuiltinDefinition = BuiltinDefinition {
    wasm_bytes: include_bytes!(env!("I64_CTZ_WASM_PATH")),
    params: &[ValType::I64],
    results: &[ValType::I64],
};

const I32_CLZ_WASM: BuiltinDefinition = BuiltinDefinition {
    wasm_bytes: include_bytes!(env!("I32_CLZ_WASM_PATH")),
    params: &[ValType::I32],
    results: &[ValType::I32],
};

const I64_CLZ_WASM: BuiltinDefinition = BuiltinDefinition {
    wasm_bytes: include_bytes!(env!("I64_CLZ_WASM_PATH")),
    params: &[ValType::I64],
    results: &[ValType::I64],
};

const NUM_BUILTINS: usize = 8;

pub struct BuiltinFunction<F: PrimeField32> {
    pub function_index: u32,
    pub definition: &'static BuiltinDefinition,
    pub function_definition: WriteOnceAsm<Directive<F>>,
}

/// Tracks which built-in functions are used and provides functions to replace unsupported
/// instructions with calls to these built-in functions.
pub struct Tracker<F: PrimeField32> {
    next_available_function_index: AtomicU32,
    used_builtins: Mutex<HashMap<usize, u32>>,
    builtin_definitions: Mutex<Vec<BuiltinFunction<F>>>,
}

impl<F: PrimeField32> Tracker<F> {
    pub fn new(next_available_function_index: u32) -> Self {
        Self {
            next_available_function_index: AtomicU32::new(next_available_function_index),
            used_builtins: Mutex::new(HashMap::new()),
            builtin_definitions: Mutex::new(Vec::new()),
        }
    }

    /// If the given instruction is not supported natively by the VM, replace it with a call to
    /// a built-in function.
    pub fn replace_with_builtins(&self, label_gen: &AtomicU32, dag: &mut BlocklessDag) {
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
                        Operator::I32Ctz => &I32_CTZ_WASM,
                        Operator::I64Ctz => &I64_CTZ_WASM,
                        Operator::I32Clz => &I32_CLZ_WASM,
                        Operator::I64Clz => &I64_CLZ_WASM,
                        // TODO: Add more built-in functions here as needed.
                        _ => continue,
                    };

                    // Get the index of the built-in function, adding it to the module if needed.
                    let function_index = self.get_index_for_builtin(label_gen, builtin);
                    // Replace the instruction with a call to the built-in function.
                    *op = Operator::Call { function_index };
                }
                Operation::Loop { sub_dag, .. } => {
                    self.replace_with_builtins(label_gen, sub_dag);
                }
                _ => (),
            }
        }
    }

    /// Returns an iterator over the used built-in functions to be added to the final program.
    /// Sorted by function index.
    pub fn into_used_builtins(self) -> Vec<BuiltinFunction<F>> {
        // There can't be more used built-ins than we have defined. Otherwise some built-in
        // function would have been added more than once.
        assert!(self.used_builtins.into_inner().unwrap().len() <= NUM_BUILTINS);

        let mut result = self.builtin_definitions.into_inner().unwrap();
        result.sort_unstable_by_key(|bf| bf.function_index);
        result
    }

    fn get_index_for_builtin(
        &self,
        label_gen: &AtomicU32,
        builtin: &'static BuiltinDefinition,
    ) -> u32 {
        let key = builtin as *const _ as usize;

        let function_index = match self.used_builtins.lock().unwrap().entry(key) {
            Entry::Occupied(entry) => {
                // This built-in function was already loaded, we can return early.
                return *entry.into_mut();
            }
            Entry::Vacant(entry) => *entry.insert(
                self.next_available_function_index
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            ),
        };

        // Load the built-in function module.
        let program =
            womir::loader::load_wasm(OpenVMSettings::<F>::new(), builtin.wasm_bytes).unwrap();

        // Compile the built-in function.
        let function = program.functions.into_iter().exactly_one().unwrap();
        let mut function_definition = function
            .advance_all_stages(
                &OpenVMSettings::<F>::new(),
                // It shouldn't make any difference which module we pass here,
                // because the built-in function should be self-contained.
                &program.m,
                0,
                label_gen,
                None,
            )
            .unwrap();
        // Fix-up the function index for the module it will be added to.
        function_definition.func_idx = function_index;

        self.builtin_definitions
            .lock()
            .unwrap()
            .push(BuiltinFunction {
                function_index,
                definition: builtin,
                function_definition,
            });

        function_index
    }
}
