use std::{collections::HashMap, vec};

use openvm_instructions::{exe::VmExe, instruction::Instruction, program::Program};
use openvm_stark_backend::p3_field::{Field, PrimeField32};
use womir::{
    generic_ir::GenericIrSetting,
    linker::LabelValue,
    loader::{flattening::WriteOnceASM, func_idx_to_label, CommonProgram},
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
        W::WASMOp { op, inputs, output } => match op {
            // Integer instructions
            wasmparser::Operator::I32Const { value } => todo!(),
            wasmparser::Operator::I64Const { value } => todo!(),
            wasmparser::Operator::I32Eqz => todo!(),
            wasmparser::Operator::I32Eq => todo!(),
            wasmparser::Operator::I32Ne => todo!(),
            wasmparser::Operator::I32LtS => todo!(),
            wasmparser::Operator::I32LtU => todo!(),
            wasmparser::Operator::I32GtS => todo!(),
            wasmparser::Operator::I32GtU => todo!(),
            wasmparser::Operator::I32LeS => todo!(),
            wasmparser::Operator::I32LeU => todo!(),
            wasmparser::Operator::I32GeS => todo!(),
            wasmparser::Operator::I32GeU => todo!(),
            wasmparser::Operator::I64Eqz => todo!(),
            wasmparser::Operator::I64Eq => todo!(),
            wasmparser::Operator::I64Ne => todo!(),
            wasmparser::Operator::I64LtS => todo!(),
            wasmparser::Operator::I64LtU => todo!(),
            wasmparser::Operator::I64GtS => todo!(),
            wasmparser::Operator::I64GtU => todo!(),
            wasmparser::Operator::I64LeS => todo!(),
            wasmparser::Operator::I64LeU => todo!(),
            wasmparser::Operator::I64GeS => todo!(),
            wasmparser::Operator::I64GeU => todo!(),
            wasmparser::Operator::I32Clz => todo!(),
            wasmparser::Operator::I32Ctz => todo!(),
            wasmparser::Operator::I32Popcnt => todo!(),
            wasmparser::Operator::I32Add => vec![Directive::Instruction(instruction_builder::add(
                output.unwrap().start as usize,
                inputs[0].start as usize,
                inputs[1].start as usize,
            ))],
            wasmparser::Operator::I32Sub => todo!(),
            wasmparser::Operator::I32Mul => todo!(),
            wasmparser::Operator::I32DivS => todo!(),
            wasmparser::Operator::I32DivU => todo!(),
            wasmparser::Operator::I32RemS => todo!(),
            wasmparser::Operator::I32RemU => todo!(),
            wasmparser::Operator::I32And => todo!(),
            wasmparser::Operator::I32Or => todo!(),
            wasmparser::Operator::I32Xor => todo!(),
            wasmparser::Operator::I32Shl => todo!(),
            wasmparser::Operator::I32ShrS => todo!(),
            wasmparser::Operator::I32ShrU => todo!(),
            wasmparser::Operator::I32Rotl => todo!(),
            wasmparser::Operator::I32Rotr => todo!(),
            wasmparser::Operator::I64Clz => todo!(),
            wasmparser::Operator::I64Ctz => todo!(),
            wasmparser::Operator::I64Popcnt => todo!(),
            wasmparser::Operator::I64Add => todo!(),
            wasmparser::Operator::I64Sub => todo!(),
            wasmparser::Operator::I64Mul => todo!(),
            wasmparser::Operator::I64DivS => todo!(),
            wasmparser::Operator::I64DivU => todo!(),
            wasmparser::Operator::I64RemS => todo!(),
            wasmparser::Operator::I64RemU => todo!(),
            wasmparser::Operator::I64And => todo!(),
            wasmparser::Operator::I64Or => todo!(),
            wasmparser::Operator::I64Xor => todo!(),
            wasmparser::Operator::I64Shl => todo!(),
            wasmparser::Operator::I64ShrS => todo!(),
            wasmparser::Operator::I64ShrU => todo!(),
            wasmparser::Operator::I64Rotl => todo!(),
            wasmparser::Operator::I64Rotr => todo!(),
            wasmparser::Operator::I32WrapI64 => todo!(),
            wasmparser::Operator::I64ExtendI32S => todo!(),
            wasmparser::Operator::I64ExtendI32U => todo!(),
            wasmparser::Operator::I32Extend8S => todo!(),
            wasmparser::Operator::I32Extend16S => todo!(),
            wasmparser::Operator::I64Extend8S => todo!(),
            wasmparser::Operator::I64Extend16S => todo!(),
            wasmparser::Operator::I64Extend32S => todo!(),

            // Parametric instruction
            wasmparser::Operator::Select => todo!(),

            // Global instructions
            wasmparser::Operator::GlobalGet { global_index } => todo!(),
            wasmparser::Operator::GlobalSet { global_index } => todo!(),

            // Memory instructions
            wasmparser::Operator::I32Load { memarg } => todo!(),
            wasmparser::Operator::I64Load { memarg } => todo!(),
            wasmparser::Operator::I32Load8S { memarg } => todo!(),
            wasmparser::Operator::I32Load8U { memarg } => todo!(),
            wasmparser::Operator::I32Load16S { memarg } => todo!(),
            wasmparser::Operator::I32Load16U { memarg } => todo!(),
            wasmparser::Operator::I64Load8S { memarg } => todo!(),
            wasmparser::Operator::I64Load8U { memarg } => todo!(),
            wasmparser::Operator::I64Load16S { memarg } => todo!(),
            wasmparser::Operator::I64Load16U { memarg } => todo!(),
            wasmparser::Operator::I64Load32S { memarg } => todo!(),
            wasmparser::Operator::I64Load32U { memarg } => todo!(),
            wasmparser::Operator::I32Store { memarg } => todo!(),
            wasmparser::Operator::I64Store { memarg } => todo!(),
            wasmparser::Operator::I32Store8 { memarg } => todo!(),
            wasmparser::Operator::I32Store16 { memarg } => todo!(),
            wasmparser::Operator::I64Store8 { memarg } => todo!(),
            wasmparser::Operator::I64Store16 { memarg } => todo!(),
            wasmparser::Operator::I64Store32 { memarg } => todo!(),
            wasmparser::Operator::MemorySize { mem } => todo!(),
            wasmparser::Operator::MemoryGrow { mem } => todo!(),
            wasmparser::Operator::MemoryInit { data_index, mem } => todo!(),
            wasmparser::Operator::MemoryCopy { dst_mem, src_mem } => todo!(),
            wasmparser::Operator::MemoryFill { mem } => todo!(),
            wasmparser::Operator::DataDrop { data_index } => todo!(),

            // Table instructions
            wasmparser::Operator::TableInit { elem_index, table } => todo!(),
            wasmparser::Operator::TableCopy {
                dst_table,
                src_table,
            } => todo!(),
            wasmparser::Operator::TableFill { table } => todo!(),
            wasmparser::Operator::TableGet { table } => todo!(),
            wasmparser::Operator::TableSet { table } => todo!(),
            wasmparser::Operator::TableGrow { table } => todo!(),
            wasmparser::Operator::TableSize { table } => todo!(),
            wasmparser::Operator::ElemDrop { elem_index } => todo!(),

            // Reference instructions
            wasmparser::Operator::RefNull { hty } => todo!(),
            wasmparser::Operator::RefIsNull => todo!(),
            wasmparser::Operator::RefFunc { function_index } => todo!(),

            // Float instructions
            wasmparser::Operator::F32Load { memarg } => todo!(),
            wasmparser::Operator::F64Load { memarg } => todo!(),
            wasmparser::Operator::F32Store { memarg } => todo!(),
            wasmparser::Operator::F64Store { memarg } => todo!(),
            wasmparser::Operator::F32Const { value } => todo!(),
            wasmparser::Operator::F64Const { value } => todo!(),
            wasmparser::Operator::F32Eq => todo!(),
            wasmparser::Operator::F32Ne => todo!(),
            wasmparser::Operator::F32Lt => todo!(),
            wasmparser::Operator::F32Gt => todo!(),
            wasmparser::Operator::F32Le => todo!(),
            wasmparser::Operator::F32Ge => todo!(),
            wasmparser::Operator::F64Eq => todo!(),
            wasmparser::Operator::F64Ne => todo!(),
            wasmparser::Operator::F64Lt => todo!(),
            wasmparser::Operator::F64Gt => todo!(),
            wasmparser::Operator::F64Le => todo!(),
            wasmparser::Operator::F64Ge => todo!(),
            wasmparser::Operator::F32Abs => todo!(),
            wasmparser::Operator::F32Neg => todo!(),
            wasmparser::Operator::F32Ceil => todo!(),
            wasmparser::Operator::F32Floor => todo!(),
            wasmparser::Operator::F32Trunc => todo!(),
            wasmparser::Operator::F32Nearest => todo!(),
            wasmparser::Operator::F32Sqrt => todo!(),
            wasmparser::Operator::F32Add => todo!(),
            wasmparser::Operator::F32Sub => todo!(),
            wasmparser::Operator::F32Mul => todo!(),
            wasmparser::Operator::F32Div => todo!(),
            wasmparser::Operator::F32Min => todo!(),
            wasmparser::Operator::F32Max => todo!(),
            wasmparser::Operator::F32Copysign => todo!(),
            wasmparser::Operator::F64Abs => todo!(),
            wasmparser::Operator::F64Neg => todo!(),
            wasmparser::Operator::F64Ceil => todo!(),
            wasmparser::Operator::F64Floor => todo!(),
            wasmparser::Operator::F64Trunc => todo!(),
            wasmparser::Operator::F64Nearest => todo!(),
            wasmparser::Operator::F64Sqrt => todo!(),
            wasmparser::Operator::F64Add => todo!(),
            wasmparser::Operator::F64Sub => todo!(),
            wasmparser::Operator::F64Mul => todo!(),
            wasmparser::Operator::F64Div => todo!(),
            wasmparser::Operator::F64Min => todo!(),
            wasmparser::Operator::F64Max => todo!(),
            wasmparser::Operator::F64Copysign => todo!(),
            wasmparser::Operator::I32TruncF32S => todo!(),
            wasmparser::Operator::I32TruncF32U => todo!(),
            wasmparser::Operator::I32TruncF64S => todo!(),
            wasmparser::Operator::I32TruncF64U => todo!(),
            wasmparser::Operator::I64TruncF32S => todo!(),
            wasmparser::Operator::I64TruncF32U => todo!(),
            wasmparser::Operator::I64TruncF64S => todo!(),
            wasmparser::Operator::I64TruncF64U => todo!(),
            wasmparser::Operator::F32ConvertI32S => todo!(),
            wasmparser::Operator::F32ConvertI32U => todo!(),
            wasmparser::Operator::F32ConvertI64S => todo!(),
            wasmparser::Operator::F32ConvertI64U => todo!(),
            wasmparser::Operator::F32DemoteF64 => todo!(),
            wasmparser::Operator::F64ConvertI32S => todo!(),
            wasmparser::Operator::F64ConvertI32U => todo!(),
            wasmparser::Operator::F64ConvertI64S => todo!(),
            wasmparser::Operator::F64ConvertI64U => todo!(),
            wasmparser::Operator::F64PromoteF32 => todo!(),
            wasmparser::Operator::I32ReinterpretF32 => todo!(),
            wasmparser::Operator::I64ReinterpretF64 => todo!(),
            wasmparser::Operator::F32ReinterpretI32 => todo!(),
            wasmparser::Operator::F64ReinterpretI64 => todo!(),
            _ => todo!(),
        },
    }
}
