use std::{
    collections::HashMap,
    sync::{Mutex, atomic::AtomicU32},
};

use itertools::Itertools;
use openvm_stark_backend::p3_field::PrimeField32;
use wasmparser::{Operator, ValType};
use womir::loader::{
    FunctionAsm, FunctionProcessingStage, Module,
    passes::blockless_dag::{BlocklessDag, Operation},
    rwm::RWMStages,
};

use crate::womir_translation::{Directive, OpenVMSettings};

pub struct BuiltinDefinition {
    wasm_bytes: &'static [u8],
    pub params: &'static [ValType],
    pub results: &'static [ValType],
    /// For multi-function WASM modules, the export name to look up.
    /// None means the module has exactly one function (legacy behavior).
    export_name: Option<&'static str>,
}

// ---- Existing builtins (single-function modules) ----

const MEMORY_COPY_WASM: BuiltinDefinition = BuiltinDefinition {
    wasm_bytes: include_bytes!(env!("MEMORY_COPY_WASM_PATH")),
    params: &[ValType::I32, ValType::I32, ValType::I32],
    results: &[],
    export_name: None,
};

const MEMORY_FILL_WASM: BuiltinDefinition = BuiltinDefinition {
    wasm_bytes: include_bytes!(env!("MEMORY_FILL_WASM_PATH")),
    params: &[ValType::I32, ValType::I32, ValType::I32],
    results: &[],
    export_name: None,
};

const I32_POPCNT_WASM: BuiltinDefinition = BuiltinDefinition {
    wasm_bytes: include_bytes!(env!("I32_POPCNT_WASM_PATH")),
    params: &[ValType::I32],
    results: &[ValType::I32],
    export_name: None,
};

const I64_POPCNT_WASM: BuiltinDefinition = BuiltinDefinition {
    wasm_bytes: include_bytes!(env!("I64_POPCNT_WASM_PATH")),
    params: &[ValType::I64],
    results: &[ValType::I64],
    export_name: None,
};

const I32_CTZ_WASM: BuiltinDefinition = BuiltinDefinition {
    wasm_bytes: include_bytes!(env!("I32_CTZ_WASM_PATH")),
    params: &[ValType::I32],
    results: &[ValType::I32],
    export_name: None,
};

const I64_CTZ_WASM: BuiltinDefinition = BuiltinDefinition {
    wasm_bytes: include_bytes!(env!("I64_CTZ_WASM_PATH")),
    params: &[ValType::I64],
    results: &[ValType::I64],
    export_name: None,
};

const I32_CLZ_WASM: BuiltinDefinition = BuiltinDefinition {
    wasm_bytes: include_bytes!(env!("I32_CLZ_WASM_PATH")),
    params: &[ValType::I32],
    results: &[ValType::I32],
    export_name: None,
};

const I64_CLZ_WASM: BuiltinDefinition = BuiltinDefinition {
    wasm_bytes: include_bytes!(env!("I64_CLZ_WASM_PATH")),
    params: &[ValType::I64],
    results: &[ValType::I64],
    export_name: None,
};

// ---- Float builtin modules (multi-function) ----

const F32_OPS_WASM: &[u8] = include_bytes!(env!("F32_OPS_WASM_PATH"));
const F64_OPS_WASM: &[u8] = include_bytes!(env!("F64_OPS_WASM_PATH"));
const FLOAT_CONV_WASM: &[u8] = include_bytes!(env!("FLOAT_CONV_WASM_PATH"));

/// Helper macro to define a float builtin from a multi-function module.
macro_rules! float_builtin {
    ($name:ident, $wasm:expr, $export:expr, [$($p:expr),*], [$($r:expr),*]) => {
        const $name: BuiltinDefinition = BuiltinDefinition {
            wasm_bytes: $wasm,
            params: &[$($p),*],
            results: &[$($r),*],
            export_name: Some($export),
        };
    };
}

// f32 arithmetic: (i32, i32) -> i32  (f32 values represented as i32 bit patterns)
float_builtin!(
    F32_ADD_WASM,
    F32_OPS_WASM,
    "f32_add",
    [ValType::I32, ValType::I32],
    [ValType::I32]
);
float_builtin!(
    F32_SUB_WASM,
    F32_OPS_WASM,
    "f32_sub",
    [ValType::I32, ValType::I32],
    [ValType::I32]
);
float_builtin!(
    F32_MUL_WASM,
    F32_OPS_WASM,
    "f32_mul",
    [ValType::I32, ValType::I32],
    [ValType::I32]
);
float_builtin!(
    F32_DIV_WASM,
    F32_OPS_WASM,
    "f32_div",
    [ValType::I32, ValType::I32],
    [ValType::I32]
);

// f32 comparison: (i32, i32) -> i32
float_builtin!(
    F32_EQ_WASM,
    F32_OPS_WASM,
    "f32_eq",
    [ValType::I32, ValType::I32],
    [ValType::I32]
);
float_builtin!(
    F32_NE_WASM,
    F32_OPS_WASM,
    "f32_ne",
    [ValType::I32, ValType::I32],
    [ValType::I32]
);
float_builtin!(
    F32_LT_WASM,
    F32_OPS_WASM,
    "f32_lt",
    [ValType::I32, ValType::I32],
    [ValType::I32]
);
float_builtin!(
    F32_LE_WASM,
    F32_OPS_WASM,
    "f32_le",
    [ValType::I32, ValType::I32],
    [ValType::I32]
);
float_builtin!(
    F32_GT_WASM,
    F32_OPS_WASM,
    "f32_gt",
    [ValType::I32, ValType::I32],
    [ValType::I32]
);
float_builtin!(
    F32_GE_WASM,
    F32_OPS_WASM,
    "f32_ge",
    [ValType::I32, ValType::I32],
    [ValType::I32]
);

// f32 unary: (i32) -> i32
float_builtin!(
    F32_ABS_WASM,
    F32_OPS_WASM,
    "f32_abs",
    [ValType::I32],
    [ValType::I32]
);
float_builtin!(
    F32_NEG_WASM,
    F32_OPS_WASM,
    "f32_neg",
    [ValType::I32],
    [ValType::I32]
);
float_builtin!(
    F32_CEIL_WASM,
    F32_OPS_WASM,
    "f32_ceil",
    [ValType::I32],
    [ValType::I32]
);
float_builtin!(
    F32_FLOOR_WASM,
    F32_OPS_WASM,
    "f32_floor",
    [ValType::I32],
    [ValType::I32]
);
float_builtin!(
    F32_TRUNC_WASM,
    F32_OPS_WASM,
    "f32_trunc",
    [ValType::I32],
    [ValType::I32]
);
float_builtin!(
    F32_NEAREST_WASM,
    F32_OPS_WASM,
    "f32_nearest",
    [ValType::I32],
    [ValType::I32]
);
float_builtin!(
    F32_SQRT_WASM,
    F32_OPS_WASM,
    "f32_sqrt",
    [ValType::I32],
    [ValType::I32]
);

// f32 binary: (i32, i32) -> i32
float_builtin!(
    F32_COPYSIGN_WASM,
    F32_OPS_WASM,
    "f32_copysign",
    [ValType::I32, ValType::I32],
    [ValType::I32]
);
float_builtin!(
    F32_MIN_WASM,
    F32_OPS_WASM,
    "f32_min",
    [ValType::I32, ValType::I32],
    [ValType::I32]
);
float_builtin!(
    F32_MAX_WASM,
    F32_OPS_WASM,
    "f32_max",
    [ValType::I32, ValType::I32],
    [ValType::I32]
);

// f64 arithmetic: (i64, i64) -> i64  (f64 values represented as i64 bit patterns)
float_builtin!(
    F64_ADD_WASM,
    F64_OPS_WASM,
    "f64_add",
    [ValType::I64, ValType::I64],
    [ValType::I64]
);
float_builtin!(
    F64_SUB_WASM,
    F64_OPS_WASM,
    "f64_sub",
    [ValType::I64, ValType::I64],
    [ValType::I64]
);
float_builtin!(
    F64_MUL_WASM,
    F64_OPS_WASM,
    "f64_mul",
    [ValType::I64, ValType::I64],
    [ValType::I64]
);
float_builtin!(
    F64_DIV_WASM,
    F64_OPS_WASM,
    "f64_div",
    [ValType::I64, ValType::I64],
    [ValType::I64]
);

// f64 comparison: (i64, i64) -> i32
float_builtin!(
    F64_EQ_WASM,
    F64_OPS_WASM,
    "f64_eq",
    [ValType::I64, ValType::I64],
    [ValType::I32]
);
float_builtin!(
    F64_NE_WASM,
    F64_OPS_WASM,
    "f64_ne",
    [ValType::I64, ValType::I64],
    [ValType::I32]
);
float_builtin!(
    F64_LT_WASM,
    F64_OPS_WASM,
    "f64_lt",
    [ValType::I64, ValType::I64],
    [ValType::I32]
);
float_builtin!(
    F64_LE_WASM,
    F64_OPS_WASM,
    "f64_le",
    [ValType::I64, ValType::I64],
    [ValType::I32]
);
float_builtin!(
    F64_GT_WASM,
    F64_OPS_WASM,
    "f64_gt",
    [ValType::I64, ValType::I64],
    [ValType::I32]
);
float_builtin!(
    F64_GE_WASM,
    F64_OPS_WASM,
    "f64_ge",
    [ValType::I64, ValType::I64],
    [ValType::I32]
);

// f64 unary: (i64) -> i64
float_builtin!(
    F64_ABS_WASM,
    F64_OPS_WASM,
    "f64_abs",
    [ValType::I64],
    [ValType::I64]
);
float_builtin!(
    F64_NEG_WASM,
    F64_OPS_WASM,
    "f64_neg",
    [ValType::I64],
    [ValType::I64]
);
float_builtin!(
    F64_CEIL_WASM,
    F64_OPS_WASM,
    "f64_ceil",
    [ValType::I64],
    [ValType::I64]
);
float_builtin!(
    F64_FLOOR_WASM,
    F64_OPS_WASM,
    "f64_floor",
    [ValType::I64],
    [ValType::I64]
);
float_builtin!(
    F64_TRUNC_WASM,
    F64_OPS_WASM,
    "f64_trunc",
    [ValType::I64],
    [ValType::I64]
);
float_builtin!(
    F64_NEAREST_WASM,
    F64_OPS_WASM,
    "f64_nearest",
    [ValType::I64],
    [ValType::I64]
);
float_builtin!(
    F64_SQRT_WASM,
    F64_OPS_WASM,
    "f64_sqrt",
    [ValType::I64],
    [ValType::I64]
);

// f64 binary: (i64, i64) -> i64
float_builtin!(
    F64_COPYSIGN_WASM,
    F64_OPS_WASM,
    "f64_copysign",
    [ValType::I64, ValType::I64],
    [ValType::I64]
);
float_builtin!(
    F64_MIN_WASM,
    F64_OPS_WASM,
    "f64_min",
    [ValType::I64, ValType::I64],
    [ValType::I64]
);
float_builtin!(
    F64_MAX_WASM,
    F64_OPS_WASM,
    "f64_max",
    [ValType::I64, ValType::I64],
    [ValType::I64]
);

// Conversions: i32 -> f32
float_builtin!(
    F32_CONVERT_I32_S_WASM,
    FLOAT_CONV_WASM,
    "f32_convert_i32_s",
    [ValType::I32],
    [ValType::I32]
);
float_builtin!(
    F32_CONVERT_I32_U_WASM,
    FLOAT_CONV_WASM,
    "f32_convert_i32_u",
    [ValType::I32],
    [ValType::I32]
);

// Conversions: i64 -> f32
float_builtin!(
    F32_CONVERT_I64_S_WASM,
    FLOAT_CONV_WASM,
    "f32_convert_i64_s",
    [ValType::I64],
    [ValType::I32]
);
float_builtin!(
    F32_CONVERT_I64_U_WASM,
    FLOAT_CONV_WASM,
    "f32_convert_i64_u",
    [ValType::I64],
    [ValType::I32]
);

// Conversions: i32 -> f64
float_builtin!(
    F64_CONVERT_I32_S_WASM,
    FLOAT_CONV_WASM,
    "f64_convert_i32_s",
    [ValType::I32],
    [ValType::I64]
);
float_builtin!(
    F64_CONVERT_I32_U_WASM,
    FLOAT_CONV_WASM,
    "f64_convert_i32_u",
    [ValType::I32],
    [ValType::I64]
);

// Conversions: i64 -> f64
float_builtin!(
    F64_CONVERT_I64_S_WASM,
    FLOAT_CONV_WASM,
    "f64_convert_i64_s",
    [ValType::I64],
    [ValType::I64]
);
float_builtin!(
    F64_CONVERT_I64_U_WASM,
    FLOAT_CONV_WASM,
    "f64_convert_i64_u",
    [ValType::I64],
    [ValType::I64]
);

// Conversions: f32 -> i32
float_builtin!(
    I32_TRUNC_F32_S_WASM,
    FLOAT_CONV_WASM,
    "i32_trunc_f32_s",
    [ValType::I32],
    [ValType::I32]
);
float_builtin!(
    I32_TRUNC_F32_U_WASM,
    FLOAT_CONV_WASM,
    "i32_trunc_f32_u",
    [ValType::I32],
    [ValType::I32]
);

// Conversions: f64 -> i32
float_builtin!(
    I32_TRUNC_F64_S_WASM,
    FLOAT_CONV_WASM,
    "i32_trunc_f64_s",
    [ValType::I64],
    [ValType::I32]
);
float_builtin!(
    I32_TRUNC_F64_U_WASM,
    FLOAT_CONV_WASM,
    "i32_trunc_f64_u",
    [ValType::I64],
    [ValType::I32]
);

// Conversions: f32 -> i64
float_builtin!(
    I64_TRUNC_F32_S_WASM,
    FLOAT_CONV_WASM,
    "i64_trunc_f32_s",
    [ValType::I32],
    [ValType::I64]
);
float_builtin!(
    I64_TRUNC_F32_U_WASM,
    FLOAT_CONV_WASM,
    "i64_trunc_f32_u",
    [ValType::I32],
    [ValType::I64]
);

// Conversions: f64 -> i64
float_builtin!(
    I64_TRUNC_F64_S_WASM,
    FLOAT_CONV_WASM,
    "i64_trunc_f64_s",
    [ValType::I64],
    [ValType::I64]
);
float_builtin!(
    I64_TRUNC_F64_U_WASM,
    FLOAT_CONV_WASM,
    "i64_trunc_f64_u",
    [ValType::I64],
    [ValType::I64]
);

// Conversions: f32 <-> f64
float_builtin!(
    F64_PROMOTE_F32_WASM,
    FLOAT_CONV_WASM,
    "f64_promote_f32",
    [ValType::I32],
    [ValType::I64]
);
float_builtin!(
    F32_DEMOTE_F64_WASM,
    FLOAT_CONV_WASM,
    "f32_demote_f64",
    [ValType::I64],
    [ValType::I32]
);

// Saturating truncations: f32 -> i32
float_builtin!(
    I32_TRUNC_SAT_F32_S_WASM,
    FLOAT_CONV_WASM,
    "i32_trunc_sat_f32_s",
    [ValType::I32],
    [ValType::I32]
);
float_builtin!(
    I32_TRUNC_SAT_F32_U_WASM,
    FLOAT_CONV_WASM,
    "i32_trunc_sat_f32_u",
    [ValType::I32],
    [ValType::I32]
);
// Saturating truncations: f64 -> i32
float_builtin!(
    I32_TRUNC_SAT_F64_S_WASM,
    FLOAT_CONV_WASM,
    "i32_trunc_sat_f64_s",
    [ValType::I64],
    [ValType::I32]
);
float_builtin!(
    I32_TRUNC_SAT_F64_U_WASM,
    FLOAT_CONV_WASM,
    "i32_trunc_sat_f64_u",
    [ValType::I64],
    [ValType::I32]
);
// Saturating truncations: f32 -> i64
float_builtin!(
    I64_TRUNC_SAT_F32_S_WASM,
    FLOAT_CONV_WASM,
    "i64_trunc_sat_f32_s",
    [ValType::I32],
    [ValType::I64]
);
float_builtin!(
    I64_TRUNC_SAT_F32_U_WASM,
    FLOAT_CONV_WASM,
    "i64_trunc_sat_f32_u",
    [ValType::I32],
    [ValType::I64]
);
// Saturating truncations: f64 -> i64
float_builtin!(
    I64_TRUNC_SAT_F64_S_WASM,
    FLOAT_CONV_WASM,
    "i64_trunc_sat_f64_s",
    [ValType::I64],
    [ValType::I64]
);
float_builtin!(
    I64_TRUNC_SAT_F64_U_WASM,
    FLOAT_CONV_WASM,
    "i64_trunc_sat_f64_u",
    [ValType::I64],
    [ValType::I64]
);

const NUM_BUILTINS: usize = 74; // 8 existing + 20 f32 + 20 f64 + 18 conversions + 8 trunc_sat

pub struct BuiltinFunction {
    pub function_index: u32,
    pub definition: &'static BuiltinDefinition,
}

impl BuiltinFunction {
    pub fn load<F: PrimeField32>(
        self,
        module: &Module,
        label_gen: &AtomicU32,
    ) -> FunctionAsm<Directive<F>> {
        // Load the built-in function module.
        let program =
            womir::loader::load_wasm(OpenVMSettings::<F>::new(), self.definition.wasm_bytes)
                .unwrap();

        // Extract the function, either by export name or as the sole function.
        let function = if let Some(export_name) = self.definition.export_name {
            // Multi-function module: find the function by export name.
            let func_idx = program
                .m
                .exported_functions
                .iter()
                .find_map(
                    |(&idx, &name)| {
                        if name == export_name { Some(idx) } else { None }
                    },
                )
                .unwrap_or_else(|| {
                    panic!("Export '{}' not found in builtin WASM module", export_name)
                });
            program
                .functions
                .into_iter()
                .nth(func_idx as usize)
                .unwrap()
        } else {
            // Single-function module: use exactly_one().
            program.functions.into_iter().exactly_one().unwrap()
        };

        // Compile the built-in function.
        let rwm_func: RWMStages<'_, OpenVMSettings<F>> = function.into();
        rwm_func
            .advance_all_stages(
                &OpenVMSettings::new(),
                // It shouldn't make any difference which module we pass here,
                // because the built-in function should be self-contained.
                module,
                self.function_index,
                label_gen,
                None,
            )
            .unwrap()
    }
}

/// Tracks which built-in functions are used and provides functions to replace unsupported
/// instructions with calls to these built-in functions.
pub struct Tracker {
    next_available_function_index: AtomicU32,
    used_builtins: Mutex<HashMap<usize, BuiltinFunction>>,
}

impl Tracker {
    pub fn new(next_available_function_index: u32) -> Self {
        Self {
            next_available_function_index: AtomicU32::new(next_available_function_index),
            used_builtins: Mutex::new(HashMap::new()),
        }
    }

    /// If the given instruction is not supported natively by the VM, replace it with a call to
    /// a built-in function.
    pub fn replace_with_builtins(&self, dag: &mut BlocklessDag) {
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

                        // f32 operations
                        Operator::F32Add => &F32_ADD_WASM,
                        Operator::F32Sub => &F32_SUB_WASM,
                        Operator::F32Mul => &F32_MUL_WASM,
                        Operator::F32Div => &F32_DIV_WASM,
                        Operator::F32Eq => &F32_EQ_WASM,
                        Operator::F32Ne => &F32_NE_WASM,
                        Operator::F32Lt => &F32_LT_WASM,
                        Operator::F32Le => &F32_LE_WASM,
                        Operator::F32Gt => &F32_GT_WASM,
                        Operator::F32Ge => &F32_GE_WASM,
                        Operator::F32Abs => &F32_ABS_WASM,
                        Operator::F32Neg => &F32_NEG_WASM,
                        Operator::F32Copysign => &F32_COPYSIGN_WASM,
                        Operator::F32Ceil => &F32_CEIL_WASM,
                        Operator::F32Floor => &F32_FLOOR_WASM,
                        Operator::F32Trunc => &F32_TRUNC_WASM,
                        Operator::F32Nearest => &F32_NEAREST_WASM,
                        Operator::F32Sqrt => &F32_SQRT_WASM,
                        Operator::F32Min => &F32_MIN_WASM,
                        Operator::F32Max => &F32_MAX_WASM,

                        // f64 operations
                        Operator::F64Add => &F64_ADD_WASM,
                        Operator::F64Sub => &F64_SUB_WASM,
                        Operator::F64Mul => &F64_MUL_WASM,
                        Operator::F64Div => &F64_DIV_WASM,
                        Operator::F64Eq => &F64_EQ_WASM,
                        Operator::F64Ne => &F64_NE_WASM,
                        Operator::F64Lt => &F64_LT_WASM,
                        Operator::F64Le => &F64_LE_WASM,
                        Operator::F64Gt => &F64_GT_WASM,
                        Operator::F64Ge => &F64_GE_WASM,
                        Operator::F64Abs => &F64_ABS_WASM,
                        Operator::F64Neg => &F64_NEG_WASM,
                        Operator::F64Copysign => &F64_COPYSIGN_WASM,
                        Operator::F64Ceil => &F64_CEIL_WASM,
                        Operator::F64Floor => &F64_FLOOR_WASM,
                        Operator::F64Trunc => &F64_TRUNC_WASM,
                        Operator::F64Nearest => &F64_NEAREST_WASM,
                        Operator::F64Sqrt => &F64_SQRT_WASM,
                        Operator::F64Min => &F64_MIN_WASM,
                        Operator::F64Max => &F64_MAX_WASM,

                        // Float conversions
                        Operator::F32ConvertI32S => &F32_CONVERT_I32_S_WASM,
                        Operator::F32ConvertI32U => &F32_CONVERT_I32_U_WASM,
                        Operator::F32ConvertI64S => &F32_CONVERT_I64_S_WASM,
                        Operator::F32ConvertI64U => &F32_CONVERT_I64_U_WASM,
                        Operator::F64ConvertI32S => &F64_CONVERT_I32_S_WASM,
                        Operator::F64ConvertI32U => &F64_CONVERT_I32_U_WASM,
                        Operator::F64ConvertI64S => &F64_CONVERT_I64_S_WASM,
                        Operator::F64ConvertI64U => &F64_CONVERT_I64_U_WASM,
                        Operator::I32TruncF32S => &I32_TRUNC_F32_S_WASM,
                        Operator::I32TruncF32U => &I32_TRUNC_F32_U_WASM,
                        Operator::I32TruncF64S => &I32_TRUNC_F64_S_WASM,
                        Operator::I32TruncF64U => &I32_TRUNC_F64_U_WASM,
                        Operator::I64TruncF32S => &I64_TRUNC_F32_S_WASM,
                        Operator::I64TruncF32U => &I64_TRUNC_F32_U_WASM,
                        Operator::I64TruncF64S => &I64_TRUNC_F64_S_WASM,
                        Operator::I64TruncF64U => &I64_TRUNC_F64_U_WASM,
                        Operator::F64PromoteF32 => &F64_PROMOTE_F32_WASM,
                        Operator::F32DemoteF64 => &F32_DEMOTE_F64_WASM,

                        // Saturating truncations
                        Operator::I32TruncSatF32S => &I32_TRUNC_SAT_F32_S_WASM,
                        Operator::I32TruncSatF32U => &I32_TRUNC_SAT_F32_U_WASM,
                        Operator::I32TruncSatF64S => &I32_TRUNC_SAT_F64_S_WASM,
                        Operator::I32TruncSatF64U => &I32_TRUNC_SAT_F64_U_WASM,
                        Operator::I64TruncSatF32S => &I64_TRUNC_SAT_F32_S_WASM,
                        Operator::I64TruncSatF32U => &I64_TRUNC_SAT_F32_U_WASM,
                        Operator::I64TruncSatF64S => &I64_TRUNC_SAT_F64_S_WASM,
                        Operator::I64TruncSatF64U => &I64_TRUNC_SAT_F64_U_WASM,

                        _ => continue,
                    };

                    // Get the index of the built-in function, adding it to the module if needed.
                    let function_index = self.get_index_for_builtin(builtin);
                    // Replace the instruction with a call to the built-in function.
                    *op = Operator::Call { function_index };
                }
                Operation::Loop { sub_dag, .. } => {
                    self.replace_with_builtins(sub_dag);
                }
                _ => (),
            }
        }
    }

    /// Returns an iterator over the used built-in functions to be added to the final program.
    /// Sorted by function index.
    pub fn into_used_builtins(self) -> impl Iterator<Item = BuiltinFunction> {
        let used_builtins = self.used_builtins.into_inner().unwrap();

        // There can't be more used built-ins than we have defined. Otherwise some built-in
        // function would have been added more than once.
        assert!(used_builtins.len() <= NUM_BUILTINS);

        used_builtins
            .into_values()
            .sorted_unstable_by_key(|bf| bf.function_index)
    }

    fn get_index_for_builtin(&self, builtin: &'static BuiltinDefinition) -> u32 {
        let key = builtin as *const _ as usize;

        self.used_builtins
            .lock()
            .unwrap()
            .entry(key)
            .or_insert_with(|| BuiltinFunction {
                definition: builtin,
                function_index: self
                    .next_available_function_index
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            })
            .function_index
    }
}
