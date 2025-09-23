use wasmparser::Operator;
use womir::loader::{dag::WasmValue, settings::MaybeConstant};

pub fn collapse_const_if_possible(op: &Operator, inputs: &[MaybeConstant]) {
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
        | Operator::I32Shl
        | Operator::I64Shl
        | Operator::I32ShrS
        | Operator::I64ShrS
        | Operator::I32ShrU
        | Operator::I64ShrU
        | Operator::I32Rotl
        | Operator::I64Rotl
        | Operator::I32Rotr
        | Operator::I64Rotr
        | Operator::I32LtS
        | Operator::I64LtS
        | Operator::I32LtU
        | Operator::I64LtU
        | Operator::I32Eq
        | Operator::I64Eq
        | Operator::I32Ne
        | Operator::I64Ne 
        // TODO: the operations below can have right-hand immediates, but need special handling
        // on code translation, so we leave them out for now.
        // | Operator::I32GeS
        // | Operator::I64GeS
        // | Operator::I32GeU
        // | Operator::I64GeU
        // | Operator::I32Rotl
        // | Operator::I64Rotl
        // | Operator::I32Rotr
        // | Operator::I64Rotr
        // TODO: Operator::Select could benefit from immediates, but also needs special handling
        => {
            if let [
                MaybeConstant::NonConstant,
                MaybeConstant::ReferenceConstant {
                    value,
                    must_collapse,
                },
            ] = inputs
                && can_be_i16(value)
            {
                // Right operand is constant and can be immediate
                must_collapse.replace(true);
            } else if is_commutative(op)
                && let [
                    MaybeConstant::ReferenceConstant {  
                        value,
                        must_collapse,
                    },
                    MaybeConstant::NonConstant,
                ] = inputs
                && can_be_i16(value)
            {
                // Left operand is constant and can be immediate (commutative operations only)
                must_collapse.replace(true);
            }
        }

        // GT is special because the left operand is the one that can be immediate
        Operator::I32GtS | Operator::I32GtU | Operator::I64GtS | Operator::I64GtU 
            // TODO: the operations below can have left-hand immediates, but need special handling
            // on code translation, so we leave them out for now.
            // | Operator::I32LeS
            // | Operator::I64LeS
            // | Operator::I32LeU
            // | Operator::I64LeU
        => {
            if let [
                MaybeConstant::ReferenceConstant {
                    value,
                    must_collapse,
                },
                MaybeConstant::NonConstant,
            ] = inputs
                && can_be_i16(value)
            {
                // Left operand is constant and can be immediate
                must_collapse.replace(true);
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

fn can_be_i16(value: &WasmValue) -> bool {
    match value {
        WasmValue::I32(v) => i16::try_from(*v).is_ok(),
        WasmValue::I64(v) => i16::try_from(*v).is_ok(),
        _ => false,
    }
}
