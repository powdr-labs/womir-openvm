use std::{collections::HashMap, vec};

use openvm_instructions::{exe::VmExe, instruction::Instruction, program::Program};
use openvm_stark_backend::p3_field::PrimeField32;
use womir::{
    generic_ir::GenericIrSetting,
    linker::LabelValue,
    loader::{flattening::WriteOnceASM, func_idx_to_label},
};

use crate::instruction_builder;

pub fn program_from_wasm<F: PrimeField32>(wasm_path: &str, entry_point: &str) -> VmExe<F> {
    let wasm_bytes = std::fs::read(wasm_path).expect("Failed to read WASM file");
    let ir_program = womir::loader::load_wasm(GenericIrSetting, &wasm_bytes).unwrap();

    let functions = ir_program
        .functions
        .into_iter()
        .map(|f| {
            let directives = f
                .directives
                .into_iter()
                .flat_map(|d| translate_directives::<F>(d))
                .collect();
            WriteOnceASM {
                directives,
                func_idx: f.func_idx,
                _frame_size: f._frame_size,
            }
        })
        .collect::<Vec<_>>();

    let (linked_program, label_map) = womir::linker::link(&functions, 1);
    drop(functions);
    let mut linked_program = linked_program
        .into_iter()
        .map(|d| {
            if let Some(i) = d.to_instruction(&label_map) {
                i
            } else {
                unreachable!("All remaining directives should be instructions")
            }
        })
        .collect::<Vec<_>>();

    // Now we need a little bit of startup code to call the entry point function.
    let start_offset = linked_program.len();
    linked_program.extend([/* TODO */]);

    // TODO: make womir read and carry debug info
    // Skip the first instruction, which is a nop inserted by the linker, and adjust pc_base accordingly.
    let program = Program::new_without_debug_infos(&linked_program[1..], 4, 4);
    drop(linked_program);

    let memory_image = ir_program
        .c
        .initial_memory
        .into_iter()
        .flat_map(|(addr, value)| {
            use womir::loader::MemoryEntry::*;
            let v = match value {
                Value(v) => v,
                FuncAddr(idx) => {
                    let label = func_idx_to_label(idx);
                    label_map[&label].pc * 4
                }
                FuncFrameSize(func_idx) => {
                    let label = func_idx_to_label(func_idx);
                    label_map[&label].frame_size.unwrap()
                }
            };

            [
                v & 0xff,
                (v >> 8) & 0xff,
                (v >> 16) & 0xff,
                (v >> 24) & 0xff,
            ]
            .into_iter()
            .enumerate()
            .filter_map(move |(i, byte)| {
                const ROM_ID: u32 = 2;
                if byte != 0 {
                    Some(((ROM_ID, addr + i as u32), F::from_canonical_u32(byte)))
                } else {
                    None
                }
            })
        })
        .collect();

    VmExe::new(program)
        .with_pc_start(start_offset as u32 * 4)
        .with_init_memory(memory_image)
}

// The instructions in this IR are 1-to-1 mapped to OpenVM instructions,
// and it is needed because we can only resolve the labels to PCs during linking.
#[derive(Clone)]
enum Directive<F: Clone> {
    Nop,
    Label {
        id: String,
        frame_size: Option<u32>,
    },
    Jump {
        target: String,
    },
    JumpIf {
        target: String,
        condition_reg: u32,
    },
    JumpIfZero {
        target: String,
        condition_reg: u32,
    },
    Jaaf {
        target: String,
        new_frame_ptr: u32,
    },
    JaafSave {
        target: String,
        new_frame_ptr: u32,
        saved_caller_fp: u32,
    },
    Call {
        target_pc: String,
        new_frame_ptr: u32,
        saved_ret_pc: u32,
        saved_caller_fp: u32,
    },
    Instruction(Instruction<F>),
}

impl<F: PrimeField32> Directive<F> {
    fn to_instruction(self, label_map: &HashMap<String, LabelValue>) -> Option<Instruction<F>> {
        match self {
            Directive::Nop | Directive::Label { .. } => None,
            Directive::Jump { target } => {
                let pc = label_map.get(&target)?.pc;
                Some(instruction_builder::jump(pc as usize))
            }
            Directive::JumpIf {
                target,
                condition_reg,
            } => {
                let pc = label_map.get(&target)?.pc;
                Some(instruction_builder::jump_if(
                    condition_reg as usize,
                    pc as usize,
                ))
            }
            Directive::JumpIfZero {
                target,
                condition_reg,
            } => {
                let pc = label_map.get(&target)?.pc;
                Some(instruction_builder::jump_if_zero(
                    condition_reg as usize,
                    pc as usize,
                ))
            }
            Directive::Jaaf {
                target,
                new_frame_ptr,
            } => {
                let pc = label_map.get(&target)?.pc;
                Some(instruction_builder::jaaf(
                    pc as usize,
                    new_frame_ptr as usize,
                ))
            }
            Directive::JaafSave {
                target,
                new_frame_ptr,
                saved_caller_fp,
            } => {
                let pc = label_map.get(&target)?.pc;
                Some(instruction_builder::jaaf_save(
                    saved_caller_fp as usize,
                    pc as usize,
                    new_frame_ptr as usize,
                ))
            }
            Directive::Call {
                target_pc,
                new_frame_ptr,
                saved_ret_pc,
                saved_caller_fp,
            } => {
                let pc = label_map.get(&target_pc)?.pc;
                Some(instruction_builder::call(
                    pc as usize,
                    new_frame_ptr as usize,
                    saved_ret_pc as usize,
                    saved_caller_fp as usize,
                ))
            }
            Directive::Instruction(i) => Some(i),
        }
    }
}

impl<F: Clone> womir::linker::Directive for Directive<F> {
    fn nop() -> Directive<F> {
        Directive::Nop
    }

    fn as_label(&self) -> Option<womir::linker::Label> {
        if let Directive::Label { id, frame_size } = self {
            Some(womir::linker::Label {
                id,
                frame_size: *frame_size,
            })
        } else {
            None
        }
    }
}

fn translate_directives<F: PrimeField32>(
    directive: womir::generic_ir::Directive,
) -> Vec<Directive<F>> {
    use womir::generic_ir::Directive as W;
    match directive {
        W::Label { id, frame_size } => {
            vec![Directive::Label { id, frame_size }]
        }
        W::AllocateFrameI {
            target_frame,
            result_ptr,
        } => todo!(),
        W::AllocateFrameV {
            frame_size,
            result_ptr,
        } => todo!(),
        W::Copy {
            src_word,
            dest_word,
        } => todo!(),
        W::CopyIntoFrame {
            src_word,
            dest_frame,
            dest_word,
        } => todo!(),
        W::Jump { target } => vec![Directive::Jump { target }],
        W::JumpOffset { offset } => todo!(),
        W::JumpIf { target, condition } => vec![Directive::JumpIf {
            target,
            condition_reg: condition,
        }],
        // W::JumpIfZero { target, condition } => {}, // TODO: implement jump if zero in Womir
        W::JumpAndActivateFrame {
            target,
            new_frame_ptr,
            saved_caller_fp,
        } => {
            vec![if let Some(fp) = saved_caller_fp {
                Directive::JaafSave {
                    target,
                    new_frame_ptr,
                    saved_caller_fp: fp,
                }
            } else {
                Directive::Jaaf {
                    target,
                    new_frame_ptr,
                }
            }]
        }
        W::Return { ret_pc, ret_fp } => vec![Directive::Instruction(instruction_builder::ret(
            ret_pc as usize,
            ret_fp as usize,
        ))],
        W::Call {
            target,
            new_frame_ptr,
            saved_ret_pc,
            saved_caller_fp,
        } => vec![Directive::Call {
            target_pc: target,
            new_frame_ptr,
            saved_ret_pc,
            saved_caller_fp,
        }],
        W::CallIndirect {
            target_pc,
            new_frame_ptr,
            saved_ret_pc,
            saved_caller_fp,
        } => vec![Directive::Instruction(instruction_builder::call(
            saved_ret_pc as usize,
            saved_caller_fp as usize,
            target_pc as usize,
            new_frame_ptr as usize,
        ))],
        W::ImportedCall {
            module,
            function,
            inputs,
            outputs,
        } => todo!(),
        W::Trap { reason } => todo!(),
        W::WASMOp { op, inputs, output } => {
            use instruction_builder as ib;
            use wasmparser::Operator as Op;

            let binary_op: Result<fn(usize, usize, usize) -> Instruction<F>, Op> = match op {
                // Integer instructions
                Op::I32Eqz => todo!(),
                Op::I32Eq => todo!(),
                Op::I32Ne => todo!(),
                Op::I32LtS => Ok(ib::lt_s),
                Op::I32LtU => Ok(ib::lt_u),
                Op::I32GtS => Ok(ib::gt_s),
                Op::I32GtU => Ok(ib::gt_u),
                Op::I32LeS => todo!(),
                Op::I32LeU => todo!(),
                Op::I32GeS => todo!(),
                Op::I32GeU => todo!(),
                Op::I64Eqz => todo!(),
                Op::I64Eq => todo!(),
                Op::I64Ne => todo!(),
                Op::I64LtS => todo!(),
                Op::I64LtU => todo!(),
                Op::I64GtS => todo!(),
                Op::I64GtU => todo!(),
                Op::I64LeS => todo!(),
                Op::I64LeU => todo!(),
                Op::I64GeS => todo!(),
                Op::I64GeU => todo!(),
                Op::I32Add => Ok(ib::add),
                Op::I32Sub => Ok(ib::sub),
                Op::I32Mul => todo!(),
                Op::I32DivS => todo!(),
                Op::I32DivU => todo!(),
                Op::I32RemS => todo!(),
                Op::I32RemU => todo!(),
                Op::I32And => Ok(ib::and),
                Op::I32Or => Ok(ib::or),
                Op::I32Xor => Ok(ib::xor),
                Op::I32Shl => Ok(ib::shl),
                Op::I32ShrS => Ok(ib::shr_s),
                Op::I32ShrU => Ok(ib::shr_u),
                Op::I32Rotl => todo!(),
                Op::I32Rotr => todo!(),
                Op::I64Add => todo!(),
                Op::I64Sub => todo!(),
                Op::I64Mul => todo!(),
                Op::I64DivS => todo!(),
                Op::I64DivU => todo!(),
                Op::I64RemS => todo!(),
                Op::I64RemU => todo!(),
                Op::I64And => todo!(),
                Op::I64Or => todo!(),
                Op::I64Xor => todo!(),
                Op::I64Shl => todo!(),
                Op::I64ShrS => todo!(),
                Op::I64ShrU => todo!(),
                Op::I64Rotl => todo!(),
                Op::I64Rotr => todo!(),

                // Float instructions
                Op::F32Eq => todo!(),
                Op::F32Ne => todo!(),
                Op::F32Lt => todo!(),
                Op::F32Gt => todo!(),
                Op::F32Le => todo!(),
                Op::F32Ge => todo!(),
                Op::F64Eq => todo!(),
                Op::F64Ne => todo!(),
                Op::F64Lt => todo!(),
                Op::F64Gt => todo!(),
                Op::F64Le => todo!(),
                Op::F64Ge => todo!(),
                Op::F32Add => todo!(),
                Op::F32Sub => todo!(),
                Op::F32Mul => todo!(),
                Op::F32Div => todo!(),
                Op::F32Min => todo!(),
                Op::F32Max => todo!(),
                Op::F32Copysign => todo!(),
                Op::F64Add => todo!(),
                Op::F64Sub => todo!(),
                Op::F64Mul => todo!(),
                Op::F64Div => todo!(),
                Op::F64Min => todo!(),
                Op::F64Max => todo!(),
                Op::F64Copysign => todo!(),

                // If not a binary operation, return the operator directly
                op => Err(op),
            };

            let op: Op<'_> = match binary_op {
                Ok(op_fn) => {
                    let input1 = inputs[0].start as usize;
                    let input2 = inputs[1].start as usize;
                    let output = output.unwrap().start as usize;
                    return vec![Directive::Instruction(op_fn(input1, input2, output))];
                }
                Err(op) => op,
            };

            match op {
                // Integer instructions
                Op::I32Const { value } => todo!(),
                Op::I64Const { value } => todo!(),
                Op::I32Clz => todo!(),
                Op::I32Ctz => todo!(),
                Op::I32Popcnt => todo!(),
                Op::I64Clz => todo!(),
                Op::I64Ctz => todo!(),
                Op::I64Popcnt => todo!(),
                Op::I32WrapI64 => todo!(),
                Op::I64ExtendI32S => todo!(),
                Op::I64ExtendI32U => todo!(),
                Op::I32Extend8S => todo!(),
                Op::I32Extend16S => todo!(),
                Op::I64Extend8S => todo!(),
                Op::I64Extend16S => todo!(),
                Op::I64Extend32S => todo!(),

                // Parametric instruction
                Op::Select => todo!(),

                // Global instructions
                Op::GlobalGet { global_index } => todo!(),
                Op::GlobalSet { global_index } => todo!(),

                // Memory instructions
                Op::I32Load { memarg } => todo!(),
                Op::I64Load { memarg } => todo!(),
                Op::I32Load8S { memarg } => todo!(),
                Op::I32Load8U { memarg } => todo!(),
                Op::I32Load16S { memarg } => todo!(),
                Op::I32Load16U { memarg } => todo!(),
                Op::I64Load8S { memarg } => todo!(),
                Op::I64Load8U { memarg } => todo!(),
                Op::I64Load16S { memarg } => todo!(),
                Op::I64Load16U { memarg } => todo!(),
                Op::I64Load32S { memarg } => todo!(),
                Op::I64Load32U { memarg } => todo!(),
                Op::I32Store { memarg } => todo!(),
                Op::I64Store { memarg } => todo!(),
                Op::I32Store8 { memarg } => todo!(),
                Op::I32Store16 { memarg } => todo!(),
                Op::I64Store8 { memarg } => todo!(),
                Op::I64Store16 { memarg } => todo!(),
                Op::I64Store32 { memarg } => todo!(),
                Op::MemorySize { mem } => todo!(),
                Op::MemoryGrow { mem } => todo!(),
                Op::MemoryInit { data_index, mem } => todo!(),
                Op::MemoryCopy { dst_mem, src_mem } => todo!(),
                Op::MemoryFill { mem } => todo!(),
                Op::DataDrop { data_index } => todo!(),

                // Table instructions
                Op::TableInit { elem_index, table } => todo!(),
                Op::TableCopy {
                    dst_table,
                    src_table,
                } => todo!(),
                Op::TableFill { table } => todo!(),
                Op::TableGet { table } => todo!(),
                Op::TableSet { table } => todo!(),
                Op::TableGrow { table } => todo!(),
                Op::TableSize { table } => todo!(),
                Op::ElemDrop { elem_index } => todo!(),

                // Reference instructions
                Op::RefNull { hty } => todo!(),
                Op::RefIsNull => todo!(),
                Op::RefFunc { function_index } => todo!(),

                // Float instructions
                Op::F32Load { memarg } => todo!(),
                Op::F64Load { memarg } => todo!(),
                Op::F32Store { memarg } => todo!(),
                Op::F64Store { memarg } => todo!(),
                Op::F32Const { value } => todo!(),
                Op::F64Const { value } => todo!(),
                Op::F32Abs => todo!(),
                Op::F32Neg => todo!(),
                Op::F32Ceil => todo!(),
                Op::F32Floor => todo!(),
                Op::F32Trunc => todo!(),
                Op::F32Nearest => todo!(),
                Op::F32Sqrt => todo!(),
                Op::F64Abs => todo!(),
                Op::F64Neg => todo!(),
                Op::F64Ceil => todo!(),
                Op::F64Floor => todo!(),
                Op::F64Trunc => todo!(),
                Op::F64Nearest => todo!(),
                Op::F64Sqrt => todo!(),
                Op::I32TruncF32S => todo!(),
                Op::I32TruncF32U => todo!(),
                Op::I32TruncF64S => todo!(),
                Op::I32TruncF64U => todo!(),
                Op::I64TruncF32S => todo!(),
                Op::I64TruncF32U => todo!(),
                Op::I64TruncF64S => todo!(),
                Op::I64TruncF64U => todo!(),
                Op::F32ConvertI32S => todo!(),
                Op::F32ConvertI32U => todo!(),
                Op::F32ConvertI64S => todo!(),
                Op::F32ConvertI64U => todo!(),
                Op::F32DemoteF64 => todo!(),
                Op::F64ConvertI32S => todo!(),
                Op::F64ConvertI32U => todo!(),
                Op::F64ConvertI64S => todo!(),
                Op::F64ConvertI64U => todo!(),
                Op::F64PromoteF32 => todo!(),
                Op::I32ReinterpretF32 => todo!(),
                Op::I64ReinterpretF64 => todo!(),
                Op::F32ReinterpretI32 => todo!(),
                Op::F64ReinterpretI64 => todo!(),
                _ => todo!(),
            }
        }
    }
}
