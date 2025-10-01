use wasmparser::Operator;
use womir::loader::{dag::WasmValue, settings::MaybeConstant};

pub fn collapse_const_if_possible(op: &Operator, inputs: &[MaybeConstant]) {
    // Handle the cases where we can turn unsigned "reg > 0", "reg >= 1", "0 < reg" and "1 <= reg" into "reg != 0"
    if let (
        Operator::I32GtU,
        [
            _,
            MaybeConstant::ReferenceConstant {
                value: WasmValue::I32(0),
                must_collapse,
            },
        ],
    )
    | (
        Operator::I64GtU,
        [
            _,
            MaybeConstant::ReferenceConstant {
                value: WasmValue::I64(0),
                must_collapse,
            },
        ],
    )
    | (
        Operator::I32GeU,
        [
            _,
            MaybeConstant::ReferenceConstant {
                value: WasmValue::I32(1),
                must_collapse,
            },
        ],
    )
    | (
        Operator::I64GeU,
        [
            _,
            MaybeConstant::ReferenceConstant {
                value: WasmValue::I64(1),
                must_collapse,
            },
        ],
    )
    | (
        Operator::I32LtU,
        [
            MaybeConstant::ReferenceConstant {
                value: WasmValue::I32(0),
                must_collapse,
            },
            _,
        ],
    )
    | (
        Operator::I64LtU,
        [
            MaybeConstant::ReferenceConstant {
                value: WasmValue::I64(0),
                must_collapse,
            },
            _,
        ],
    )
    | (
        Operator::I32LeU,
        [
            MaybeConstant::ReferenceConstant {
                value: WasmValue::I32(1),
                must_collapse,
            },
            _,
        ],
    )
    | (
        Operator::I64LeU,
        [
            MaybeConstant::ReferenceConstant {
                value: WasmValue::I64(1),
                must_collapse,
            },
            _,
        ],
    ) = (op, inputs)
    {
        must_collapse.replace(true);
        return;
    }

    match op {
        // All the operations using the alu adapter, thus supporting immediate operands
        Operator::I32Add
        | Operator::I64Add
        | Operator::I32Sub
        | Operator::I64Sub
        | Operator::I32Mul
        | Operator::I64Mul
        | Operator::I32DivS
        | Operator::I64DivS
        | Operator::I32DivU
        | Operator::I64DivU
        | Operator::I32RemS
        | Operator::I64RemS
        | Operator::I32RemU
        | Operator::I64RemU
        | Operator::I32Xor
        | Operator::I64Xor
        | Operator::I32Or
        | Operator::I64Or
        | Operator::I32And
        | Operator::I64And
        | Operator::I32LtS
        | Operator::I64LtS
        | Operator::I32LtU
        | Operator::I64LtU
        | Operator::I32GeS
        | Operator::I64GeS
        | Operator::I32GeU
        | Operator::I64GeU
        | Operator::I32Eq
        | Operator::I64Eq
        | Operator::I32Ne
        | Operator::I64Ne => {
            if let [
                _,
                MaybeConstant::ReferenceConstant {
                    value,
                    must_collapse,
                },
            ] = inputs
                && can_be_i16(value)
            {
                // Right operand is constant and can be immediate
                must_collapse.replace(true);
            } else if let [
                MaybeConstant::ReferenceConstant {
                    value,
                    must_collapse,
                },
                MaybeConstant::NonConstant,
            ] = inputs
                && ((is_commutative(op) && can_be_i16(value))
                    || can_turn_to_lt(op, value).is_some())
            {
                // Left operand is constant and can be immediate
                // (either the operation is commutative, or it's
                // a GE that can be transformed to LT)
                must_collapse.replace(true);
            }
        }

        // Shift and rot operations are special because they can handle immediates
        // outside of i16 range, as the value is masked to the bitwidth of the type.
        Operator::I32Shl
        | Operator::I64Shl
        | Operator::I32ShrS
        | Operator::I64ShrS
        | Operator::I32ShrU
        | Operator::I64ShrU
        | Operator::I32Rotl
        | Operator::I64Rotl
        | Operator::I32Rotr
        | Operator::I64Rotr => {
            if let [_, MaybeConstant::ReferenceConstant { must_collapse, .. }] = inputs {
                must_collapse.replace(true);
            }
        }

        // GT is special because the left operand is the one that can be immediate
        // LE is implemented in terms of GT, so the same applies.
        Operator::I32GtS
        | Operator::I32GtU
        | Operator::I64GtS
        | Operator::I64GtU
        | Operator::I32LeS
        | Operator::I64LeS
        | Operator::I32LeU
        | Operator::I64LeU => {
            if let [
                MaybeConstant::ReferenceConstant {
                    value,
                    must_collapse,
                },
                _,
            ] = inputs
                && can_be_i16(value)
            {
                // Left operand is constant and can be immediate
                must_collapse.replace(true);
            } else if let [
                _,
                MaybeConstant::ReferenceConstant {
                    value,
                    must_collapse,
                },
            ] = inputs
                && can_turn_to_lt(op, value).is_some()
            {
                // The constant is on the right, but we can turn the LE into an LT,
                // so we can use the left operand as immediate.
                must_collapse.replace(true);
            }
        }

        // In Select, both value inputs can be immediates.
        // The condition can not, assuming an optimized wasm.
        Operator::Select | Operator::TypedSelect { .. } => {
            for input in &inputs[..2] {
                if let MaybeConstant::ReferenceConstant { must_collapse, .. } = input {
                    must_collapse.replace(true);
                }
            }
        }
        _ => {
            // Instruction doesn't support immediate operands.
        }
    }
}

fn is_commutative(op: &Operator) -> bool {
    matches!(
        op,
        Operator::I32Add
            | Operator::I64Add
            | Operator::I32Mul
            | Operator::I64Mul
            | Operator::I32Xor
            | Operator::I64Xor
            | Operator::I32Or
            | Operator::I64Or
            | Operator::I32And
            | Operator::I64And
            | Operator::I32Eq
            | Operator::I64Eq
            | Operator::I32Ne
            | Operator::I64Ne
    )
}

/// If op is a GE or LE, can we turn "c >= x" or "x <= c" into "x < c + 1", where c + 1 fits in i16?
pub fn can_turn_to_lt(op: &Operator, value: &WasmValue) -> Option<i16> {
    match op {
        // Signed case:
        Operator::I32GeS | Operator::I64GeS | Operator::I32LeS | Operator::I64LeS => match value {
            WasmValue::I32(v) => v.checked_add(1).and_then(|v| i16::try_from(v).ok()),
            WasmValue::I64(v) => v.checked_add(1).and_then(|v| i16::try_from(v).ok()),
            _ => None,
        },
        // Unsigned case:
        // This is weird, because we need to represent an unsigned value as a sign-extended i16.
        // So, we can represent (c+1) if it falls in the ranges:
        // [0..0x7FFF] or [0xFFFF8000..0xFFFFFFFF], for u32
        // [0..0x7FFF] or [0xFFFFFFFFFFFF8000..0xFFFFFFFFFFFFFFFF], for u64
        Operator::I32GeU | Operator::I64GeU | Operator::I32LeU | Operator::I64LeU => match value {
            WasmValue::I32(v) => {
                let uv = *v as u32;
                uv.checked_add(1).and_then(|v| i16::try_from(v as i32).ok())
            }
            WasmValue::I64(v) => {
                let uv = *v as u64;
                uv.checked_add(1).and_then(|v| i16::try_from(v as i64).ok())
            }
            _ => None,
        },
        _ => None,
    }
}

fn can_be_i16(value: &WasmValue) -> bool {
    match value {
        WasmValue::I32(v) => i16::try_from(*v).is_ok(),
        WasmValue::I64(v) => i16::try_from(*v).is_ok(),
        _ => false,
    }
}
