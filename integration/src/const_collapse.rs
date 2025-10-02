use wasmparser::Operator as Op;
use womir::loader::{dag::WasmValue as Val, settings};

pub fn collapse_const_if_possible(op: &Op, inputs: &[settings::MaybeConstant]) {
    use settings::MaybeConstant::{NonConstant, ReferenceConstant};

    if inputs.len() < 2 {
        // Not enough inputs to consider collapsing
        return;
    }

    // Handle the cases where unsigned "reg >= 1" and "reg > 0" can be turned into "reg != 0"
    if let (
        Op::I32GeU | Op::I64GeU,
        ReferenceConstant {
            value: Val::I32(1) | Val::I64(1),
            must_collapse,
        },
    )
    | (
        Op::I32GtU | Op::I64GtU,
        ReferenceConstant {
            value: Val::I32(0) | Val::I64(0),
            must_collapse,
        },
    ) = (&op, &inputs[1])
    {
        must_collapse.replace(true);
        return;
    }

    // Handle the cases where unsigned "0 < reg" and "1 <= reg" can be turned into "reg != 0"
    if let (
        Op::I32LeU | Op::I64LeU,
        ReferenceConstant {
            value: Val::I32(1) | Val::I64(1),
            must_collapse,
        },
    )
    | (
        Op::I32LtU | Op::I64LtU,
        ReferenceConstant {
            value: Val::I32(0) | Val::I64(0),
            must_collapse,
        },
    ) = (&op, &inputs[0])
    {
        must_collapse.replace(true);
        return;
    }

    match op {
        // All the operations using the alu adapter, thus supporting immediate operands
        Op::I32Add
        | Op::I64Add
        | Op::I32Sub
        | Op::I64Sub
        | Op::I32Mul
        | Op::I64Mul
        | Op::I32DivS
        | Op::I64DivS
        | Op::I32DivU
        | Op::I64DivU
        | Op::I32RemS
        | Op::I64RemS
        | Op::I32RemU
        | Op::I64RemU
        | Op::I32Xor
        | Op::I64Xor
        | Op::I32Or
        | Op::I64Or
        | Op::I32And
        | Op::I64And
        | Op::I32LtS
        | Op::I64LtS
        | Op::I32LtU
        | Op::I64LtU
        | Op::I32GeS
        | Op::I64GeS
        | Op::I32GeU
        | Op::I64GeU
        | Op::I32Eq
        | Op::I64Eq
        | Op::I32Ne
        | Op::I64Ne => {
            if let [
                _,
                ReferenceConstant {
                    value,
                    must_collapse,
                },
            ] = inputs
                && can_be_i16(value)
            {
                // Right operand is constant and can be immediate
                must_collapse.replace(true);
            } else if let [
                ReferenceConstant {
                    value,
                    must_collapse,
                },
                NonConstant,
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
        Op::I32Shl
        | Op::I64Shl
        | Op::I32ShrS
        | Op::I64ShrS
        | Op::I32ShrU
        | Op::I64ShrU
        | Op::I32Rotl
        | Op::I64Rotl
        | Op::I32Rotr
        | Op::I64Rotr => {
            if let [_, ReferenceConstant { must_collapse, .. }] = inputs {
                must_collapse.replace(true);
            }
        }

        // GT is special because the left operand is the one that can be immediate
        // LE is implemented in terms of GT, so the same applies.
        Op::I32GtS
        | Op::I32GtU
        | Op::I64GtS
        | Op::I64GtU
        | Op::I32LeS
        | Op::I64LeS
        | Op::I32LeU
        | Op::I64LeU => {
            if let [
                ReferenceConstant {
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
                ReferenceConstant {
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
        Op::Select | Op::TypedSelect { .. } => {
            for input in &inputs[..2] {
                if let ReferenceConstant { must_collapse, .. } = input {
                    must_collapse.replace(true);
                }
            }
        }
        _ => {
            // Instruction doesn't support immediate operands.
        }
    }
}

fn is_commutative(op: &Op) -> bool {
    matches!(
        op,
        Op::I32Add
            | Op::I64Add
            | Op::I32Mul
            | Op::I64Mul
            | Op::I32Xor
            | Op::I64Xor
            | Op::I32Or
            | Op::I64Or
            | Op::I32And
            | Op::I64And
            | Op::I32Eq
            | Op::I64Eq
            | Op::I32Ne
            | Op::I64Ne
    )
}

/// If op is a GE or LE, can we turn "c >= x" or "x <= c" into "x < c + 1", where c + 1 fits in i16?
pub fn can_turn_to_lt(op: &Op, value: &Val) -> Option<i16> {
    match op {
        // Signed case:
        Op::I32GeS | Op::I64GeS | Op::I32LeS | Op::I64LeS => match value {
            Val::I32(v) => v.checked_add(1).and_then(|v| i16::try_from(v).ok()),
            Val::I64(v) => v.checked_add(1).and_then(|v| i16::try_from(v).ok()),
            _ => None,
        },
        // Unsigned case:
        // This is weird, because we need to represent an unsigned value as a sign-extended i16.
        // So, we can represent (c+1) if it falls in the ranges:
        // [0..0x7FFF] or [0xFFFF8000..0xFFFFFFFF], for u32
        // [0..0x7FFF] or [0xFFFFFFFFFFFF8000..0xFFFFFFFFFFFFFFFF], for u64
        Op::I32GeU | Op::I64GeU | Op::I32LeU | Op::I64LeU => match value {
            Val::I32(v) => {
                let uv = *v as u32;
                uv.checked_add(1).and_then(|v| i16::try_from(v as i32).ok())
            }
            Val::I64(v) => {
                let uv = *v as u64;
                uv.checked_add(1).and_then(|v| i16::try_from(v as i64).ok())
            }
            _ => None,
        },
        _ => None,
    }
}

fn can_be_i16(value: &Val) -> bool {
    match value {
        Val::I32(v) => i16::try_from(*v).is_ok(),
        Val::I64(v) => i16::try_from(*v).is_ok(),
        _ => false,
    }
}
