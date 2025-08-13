use std::{collections::HashMap, ops::Range, vec};

use crate::{instruction_builder as ib, to_field::ToField};
use openvm_instructions::{exe::VmExe, instruction::Instruction, program::Program, riscv};
use openvm_stark_backend::p3_field::PrimeField32;
use wasmparser::{Operator as Op, ValType};
use womir::{
    linker::LabelValue,
    loader::{
        flattening::{
            settings::{ComparisonFunction, JumpCondition, Settings},
            Generators, LabelType, WriteOnceASM,
        },
        func_idx_to_label, CommonProgram,
    },
};

pub fn program_from_womir<F: PrimeField32>(
    ir_program: womir::loader::Program<OpenVMSettings<F>>,
    entry_point: &str,
) -> VmExe<F> {
    let functions = ir_program
        .functions
        .into_iter()
        .map(|f| {
            let directives = f.directives.into_iter().collect();
            WriteOnceASM {
                directives,
                func_idx: f.func_idx,
                _frame_size: f._frame_size,
            }
        })
        .collect::<Vec<_>>();

    let (linked_program, mut label_map) = womir::linker::link(&functions, 1);
    drop(functions);

    for v in label_map.values_mut() {
        v.pc *= riscv::RV32_REGISTER_NUM_LIMBS as u32;
    }

    // Now we need a little bit of startup code to call the entry point function.
    // We assume the initial frame has space for at least one word: the frame pointer
    // to the entry point function.
    let start_offset = linked_program.len();

    let mut linked_program = linked_program
        .into_iter()
        // Remove `nop` added by the linker to avoid pc=0 being a valid instruction.
        .skip(1)
        .map(|d| {
            if let Some(i) = d.into_instruction(&label_map) {
                i
            } else {
                unreachable!("All remaining directives should be instructions")
            }
        })
        .collect::<Vec<_>>();

    // We assume that the loop above removes a single `nop` introduced by the linker.
    assert_eq!(linked_program.len(), start_offset - 1);

    // Create the startup code to call the entry point function.
    let entry_point = &label_map[entry_point];
    linked_program.extend(create_startup_code(&ir_program.c, entry_point));

    // TODO: make womir read and carry debug info
    // Skip the first instruction, which is a nop inserted by the linker, and adjust pc_base accordingly.
    let program = Program::new_without_debug_infos(&linked_program, 4, 4);
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
                    label_map[&label].pc
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
        .with_pc_start((start_offset * riscv::RV32_REGISTER_NUM_LIMBS) as u32)
        .with_init_memory(memory_image)
}

fn create_startup_code<F>(ctx: &CommonProgram, entry_point: &LabelValue) -> Vec<Instruction<F>>
where
    F: PrimeField32,
{
    let fn_fp = 1;
    let zero_reg = 0;
    let mut code = vec![
        ib::allocate_frame_imm(fn_fp, entry_point.frame_size.unwrap() as usize),
        ib::const_32_imm(zero_reg, 0, 0),
    ];

    let entry_point_func_type = &ctx.get_func_type(entry_point.func_idx.unwrap()).ty;

    // If the entry point function has arguments, we need to fill them with the result from read32
    let params = entry_point_func_type.params();
    let num_input_words = womir::word_count_types::<OpenVMSettings<F>>(params);
    // This is a little hacky because we know the initial first allocated frame starts at 2,
    // so we can just write directly into it by calculating the offset.
    // address 2: reserved for return address
    // address 3: reserved for frame pointer
    // address 4: first argument
    // address 4 + i: i-th argument
    let mut ptr = 4;
    for _ in 0..num_input_words {
        //code.push(ib::read32(ptr as usize));
        // code.push(ib::const_32_imm(ptr as usize, 10, 0));
        code.push(ib::pre_read_u32::<F>());
        code.push(ib::read_u32::<F>(ptr as usize));
        ptr += 1;
    }

    code.push(ib::call(0, 1, entry_point.pc as usize, fn_fp));

    // We can also read the return values directly from the function's frame, which happens
    // to be right after the arguments.
    let results = entry_point_func_type.results();
    let num_output_words = womir::word_count_types::<OpenVMSettings<F>>(results);
    for i in 0..num_output_words {
        code.push(ib::reveal_imm(ptr as usize, zero_reg, i as usize));
        ptr += 1;
    }

    code.push(ib::halt());

    code
}

// The instructions in this IR are 1-to-1 mapped to OpenVM instructions,
// and it is needed because we can only resolve the labels to PCs during linking.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub enum Directive<F> {
    Nop,
    Label {
        id: String,
        frame_size: Option<u32>,
    },
    AllocateFrameI {
        target_frame: String,
        result_ptr: u32,
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

pub struct OpenVMSettings<F> {
    _phantom: std::marker::PhantomData<F>,
}

impl<F> OpenVMSettings<F> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

#[allow(refining_impl_trait)]
impl<'a, F: PrimeField32> Settings<'a> for OpenVMSettings<F> {
    type Directive = Directive<F>;

    fn bytes_per_word() -> u32 {
        4
    }

    fn words_per_ptr() -> u32 {
        1
    }

    fn is_jump_condition_available(_cond: JumpCondition) -> bool {
        true
    }

    fn is_relative_jump_available() -> bool {
        true
    }

    fn to_plain_local_jump(directive: Self::Directive) -> Result<String, Self::Directive> {
        if let Directive::Jump { target } = directive {
            Ok(target)
        } else {
            Err(directive)
        }
    }

    fn emit_label(
        &self,
        _g: &mut Generators<'a, '_, Self>,
        name: String,
        frame_size: Option<u32>,
    ) -> Self::Directive {
        Directive::Label {
            id: name,
            frame_size,
        }
    }

    fn emit_trap(
        &self,
        _g: &mut Generators<'a, '_, Self>,
        _trap: womir::loader::flattening::TrapReason,
    ) -> Self::Directive {
        todo!()
    }

    fn emit_allocate_label_frame(
        &self,
        _g: &mut Generators<'a, '_, Self>,
        label: String,
        result_ptr: Range<u32>,
    ) -> Self::Directive {
        Directive::AllocateFrameI {
            target_frame: label,
            result_ptr: result_ptr.start,
        }
    }

    fn emit_allocate_value_frame(
        &self,
        _g: &mut Generators<'a, '_, Self>,
        frame_size_ptr: Range<u32>,
        result_ptr: Range<u32>,
    ) -> Self::Directive {
        Directive::Instruction(ib::allocate_frame_reg(
            result_ptr.start as usize,
            frame_size_ptr.start as usize,
        ))
    }

    fn emit_copy(
        &self,
        _g: &mut Generators<'a, '_, Self>,
        src_ptr: Range<u32>,
        dest_ptr: Range<u32>,
    ) -> Self::Directive {
        Directive::Instruction(ib::addi(
            dest_ptr.start as usize,
            src_ptr.start as usize,
            F::ZERO,
        ))
    }

    fn emit_copy_into_frame(
        &self,
        _g: &mut Generators<'a, '_, Self>,
        src_ptr: Range<u32>,
        dest_frame_ptr: Range<u32>,
        dest_offset: Range<u32>,
    ) -> Self::Directive {
        Directive::Instruction(ib::copy_into_frame(
            dest_offset.start as usize,
            src_ptr.start as usize,
            dest_frame_ptr.start as usize,
        ))
    }

    fn emit_jump(&self, label: String) -> Self::Directive {
        Directive::Jump { target: label }
    }

    fn emit_jump_into_loop(
        &self,
        _g: &mut Generators<'a, '_, Self>,
        loop_label: String,
        loop_frame_ptr: Range<u32>,
        ret_info_to_copy: Option<womir::loader::flattening::settings::ReturnInfosToCopy>,
        saved_curr_fp_ptr: Option<Range<u32>>,
    ) -> Vec<Self::Directive> {
        let mut directives = if let Some(to_copy) = ret_info_to_copy {
            assert_eq!(Self::words_per_ptr(), 1);
            vec![
                Directive::Instruction(ib::copy_into_frame(
                    to_copy.dest.ret_pc.start as usize,
                    to_copy.src.ret_pc.start as usize,
                    loop_frame_ptr.start as usize,
                )),
                Directive::Instruction(ib::copy_into_frame(
                    to_copy.dest.ret_fp.start as usize,
                    to_copy.src.ret_fp.start as usize,
                    loop_frame_ptr.start as usize,
                )),
            ]
        } else {
            Vec::new()
        };

        let jump = if let Some(saved_caller_fp) = saved_curr_fp_ptr {
            Directive::JaafSave {
                target: loop_label,
                new_frame_ptr: loop_frame_ptr.start,
                saved_caller_fp: saved_caller_fp.start,
            }
        } else {
            Directive::Jaaf {
                target: loop_label,
                new_frame_ptr: loop_frame_ptr.start,
            }
        };

        directives.push(jump);
        directives
    }

    fn emit_conditional_jump(
        &self,
        _g: &mut Generators<'a, '_, Self>,
        condition_type: womir::loader::flattening::settings::JumpCondition,
        label: String,
        condition_ptr: Range<u32>,
    ) -> Self::Directive {
        match condition_type {
            JumpCondition::IfNotZero => Directive::JumpIf {
                target: label,
                condition_reg: condition_ptr.start,
            },
            JumpCondition::IfZero => Directive::JumpIfZero {
                target: label,
                condition_reg: condition_ptr.start,
            },
        }
    }

    fn emit_conditional_jump_cmp_immediate(
        &self,
        g: &mut Generators<'a, '_, Self>,
        cmp: womir::loader::flattening::settings::ComparisonFunction,
        value_ptr: Range<u32>,
        immediate: u32,
        label: String,
    ) -> Vec<Self::Directive> {
        let cmp_insn = match cmp {
            ComparisonFunction::Equal => todo!(), // i32eq
            ComparisonFunction::GreaterThanOrEqualUnsigned => todo!(), // i32geu
            ComparisonFunction::LessThanUnsigned => ib::lt_u,
        };

        let const_value = g.r.allocate_type(ValType::I32);
        let comparison = g.r.allocate_type(ValType::I32);

        let imm_lo: u16 = (immediate & 0xffff) as u16;
        let imm_hi: u16 = ((immediate >> 16) & 0xffff) as u16;

        vec![
            Directive::Instruction(ib::const_32_imm(const_value.start as usize, imm_lo, imm_hi)),
            Directive::Instruction(cmp_insn(
                comparison.start as usize,
                value_ptr.start as usize,
                const_value.start as usize,
            )),
            Directive::JumpIf {
                target: label,
                condition_reg: comparison.start,
            },
        ]
    }

    fn emit_relative_jump(
        &self,
        _g: &mut Generators<'a, '_, Self>,
        _offset_ptr: Range<u32>,
    ) -> Self::Directive {
        todo!()
    }

    fn emit_jump_out_of_loop(
        &self,
        _g: &mut Generators<'a, '_, Self>,
        target_label: String,
        target_frame_ptr: Range<u32>,
    ) -> Self::Directive {
        Directive::Jaaf {
            target: target_label,
            new_frame_ptr: target_frame_ptr.start,
        }
    }

    fn emit_return(
        &self,
        _g: &mut Generators<'a, '_, Self>,
        ret_pc_ptr: Range<u32>,
        caller_fp_ptr: Range<u32>,
    ) -> Self::Directive {
        Directive::Instruction(ib::ret(
            ret_pc_ptr.start as usize,
            caller_fp_ptr.start as usize,
        ))
    }

    fn emit_imported_call(
        &self,
        _g: &mut Generators<'a, '_, Self>,
        _module: &'a str,
        _function: &'a str,
        _inputs: Vec<Range<u32>>,
        _outputs: Vec<Range<u32>>,
    ) -> Self::Directive {
        todo!()
    }

    fn emit_function_call(
        &self,
        _g: &mut Generators<'a, '_, Self>,
        function_label: String,
        function_frame_ptr: Range<u32>,
        saved_ret_pc_ptr: Range<u32>,
        saved_caller_fp_ptr: Range<u32>,
    ) -> Self::Directive {
        Directive::Call {
            target_pc: function_label,
            new_frame_ptr: function_frame_ptr.start,
            saved_ret_pc: saved_ret_pc_ptr.start,
            saved_caller_fp: saved_caller_fp_ptr.start,
        }
    }

    fn emit_indirect_call(
        &self,
        _g: &mut Generators<'a, '_, Self>,
        target_pc_ptr: Range<u32>,
        function_frame_ptr: Range<u32>,
        saved_ret_pc_ptr: Range<u32>,
        saved_caller_fp_ptr: Range<u32>,
    ) -> Self::Directive {
        Directive::Instruction(ib::call(
            saved_ret_pc_ptr.start as usize,
            saved_caller_fp_ptr.start as usize,
            target_pc_ptr.start as usize,
            function_frame_ptr.start as usize,
        ))
    }

    fn emit_table_get(
        &self,
        _g: &mut Generators<'a, '_, Self>,
        _table_idx: u32,
        _entry_idx_ptr: Range<u32>,
        _dest_ptr: Range<u32>,
    ) -> Self::Directive {
        todo!()
    }

    fn emit_wasm_op(
        &self,
        g: &mut Generators<'a, '_, Self>,
        op: Op<'a>,
        inputs: Vec<Range<u32>>,
        output: Option<Range<u32>>,
    ) -> Vec<Self::Directive> {
        // First handle single-instruction binary operations.
        type BinaryOpFn<F> = fn(usize, usize, usize) -> Instruction<F>;
        let binary_op: Result<BinaryOpFn<F>, Op> = match op {
            // 32-bit integer instructions
            Op::I32Eq => Ok(ib::eq),
            Op::I32Ne => Ok(ib::neq),
            Op::I32LtS => Ok(ib::lt_s),
            Op::I32LtU => Ok(ib::lt_u),
            Op::I32GtS => Ok(ib::gt_s),
            Op::I32GtU => Ok(ib::gt_u),
            Op::I32Add => Ok(ib::add),
            Op::I32Sub => Ok(ib::sub),
            Op::I32Mul => Ok(ib::mul),
            Op::I32DivS => Ok(ib::div),
            Op::I32DivU => Ok(ib::divu),
            Op::I32RemS => Ok(ib::rem),
            Op::I32RemU => Ok(ib::remu),
            Op::I32And => Ok(ib::and),
            Op::I32Or => Ok(ib::or),
            Op::I32Xor => Ok(ib::xor),
            Op::I32Shl => Ok(ib::shl),
            Op::I32ShrS => Ok(ib::shr_s),
            Op::I32ShrU => Ok(ib::shr_u),

            // 64-bit integer instructions
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
                return vec![Directive::Instruction(op_fn(output, input1, input2))];
            }
            Err(op) => op,
        };

        // Handle the remaining operations
        match op {
            // 32-bit integer instructions
            Op::I32Const { value } => {
                let output = output.unwrap().start as usize;
                let value_u = value as u32;
                let imm_lo: u16 = (value_u & 0xffff) as u16;
                let imm_hi: u16 = ((value_u >> 16) & 0xffff) as u16;
                vec![Directive::Instruction(ib::const_32_imm(
                    output, imm_lo, imm_hi,
                ))]
            }
            Op::I64Const { value } => {
                let output = output.unwrap().start as usize;
                let lower = value as u32;
                let lower_lo: u16 = (lower & 0xffff) as u16;
                let lower_hi: u16 = ((lower >> 16) & 0xffff) as u16;
                let upper = (value >> 32) as u32;
                let upper_lo: u16 = (upper & 0xffff) as u16;
                let upper_hi: u16 = ((upper >> 16) & 0xffff) as u16;
                vec![
                    Directive::Instruction(ib::const_32_imm(output, lower_lo, lower_hi)),
                    Directive::Instruction(ib::const_32_imm(output + 1, upper_lo, upper_hi)),
                ]
            }
            Op::I32Rotl => {
                let input1 = inputs[0].start as usize;
                let input2 = inputs[1].start as usize;
                let output = output.unwrap().start as usize;
                let shiftl_amount = g.r.allocate_type(ValType::I32).start as usize;
                let shiftl = g.r.allocate_type(ValType::I32).start as usize;
                let const32 = g.r.allocate_type(ValType::I32).start as usize;
                let shiftr_amount = g.r.allocate_type(ValType::I32).start as usize;
                let shiftr = g.r.allocate_type(ValType::I32).start as usize;
                vec![
                    // get least significant 5 bits for rotation amount
                    Directive::Instruction(ib::andi(shiftl_amount, input2, 0x1f.to_f().unwrap())),
                    // shift left
                    Directive::Instruction(ib::shl(shiftl, input1, shiftl_amount)),
                    // get right shift amount
                    Directive::Instruction(ib::const_32_imm(const32, 0x20, 0x0)),
                    Directive::Instruction(ib::sub(shiftr_amount, const32, shiftl_amount)),
                    // shift right
                    Directive::Instruction(ib::shr_u(shiftr, input1, shiftr_amount)),
                    // or the two results
                    Directive::Instruction(ib::or(output, shiftl, shiftr)),
                ]
            }
            Op::I32Rotr => {
                let input1 = inputs[0].start as usize;
                let input2 = inputs[1].start as usize;
                let output = output.unwrap().start as usize;
                let shiftl_amount = g.r.allocate_type(ValType::I32).start as usize;
                let shiftl = g.r.allocate_type(ValType::I32).start as usize;
                let const32 = g.r.allocate_type(ValType::I32).start as usize;
                let shiftr_amount = g.r.allocate_type(ValType::I32).start as usize;
                let shiftr = g.r.allocate_type(ValType::I32).start as usize;
                vec![
                    // get least significant 5 bits for rotation amount
                    Directive::Instruction(ib::andi(shiftr_amount, input2, 0x1f.to_f().unwrap())),
                    // shift right
                    Directive::Instruction(ib::shr_u(shiftr, input1, shiftr_amount)),
                    // get left shift amount
                    Directive::Instruction(ib::const_32_imm(const32, 0x20, 0x0)),
                    Directive::Instruction(ib::sub(shiftl_amount, const32, shiftr_amount)),
                    // shift left
                    Directive::Instruction(ib::shl(shiftl, input1, shiftl_amount)),
                    // or the two results
                    Directive::Instruction(ib::or(output, shiftl, shiftr)),
                ]
            }
            Op::I32LeS | Op::I32LeU | Op::I32GeS | Op::I32GeU => {
                let inverse_op = match op {
                    Op::I32LeS => ib::gt_s,
                    Op::I32LeU => ib::gt_u,
                    Op::I32GeS => ib::lt_s,
                    Op::I32GeU => ib::lt_u,
                    _ => unreachable!(),
                };

                let input1 = inputs[0].start as usize;
                let input2 = inputs[1].start as usize;
                let output = output.unwrap().start as usize;

                let inverse_result = g.r.allocate_type(ValType::I32).start as usize;

                // Perform the inverse operation and invert the result
                vec![
                    Directive::Instruction(inverse_op(inverse_result, input1, input2)),
                    Directive::Instruction(ib::eqi(output, inverse_result, F::ZERO)),
                ]
            }
            Op::I32Eqz => {
                let input = inputs[0].start as usize;
                let output = output.unwrap().start as usize;
                vec![Directive::Instruction(ib::eqi(output, input, F::ZERO))]
            }
            Op::I32Clz => todo!(),
            Op::I32Ctz => todo!(),
            Op::I32Popcnt => todo!(),

            Op::I32WrapI64 => {
                // TODO: considering we are using a single address space for both i32 and i64,
                // this instruction could be elided at womir level.

                let lower_limb = inputs[0].start as usize;
                // The higher limb is ignored.
                let output = output.unwrap().start as usize;

                // Just copy the lower limb to the output.
                vec![Directive::Instruction(ib::addi(
                    output,
                    lower_limb,
                    F::ZERO,
                ))]
            }
            Op::I32Extend8S | Op::I32Extend16S => {
                let input = inputs[0].start as usize;
                let output = output.unwrap().start as usize;

                let shift = match op {
                    Op::I32Extend8S => 24,
                    Op::I32Extend16S => 16,
                    _ => unreachable!(),
                }
                .to_f()
                .unwrap();

                // Left shift followed by arithmetic right shift
                vec![
                    Directive::Instruction(ib::shl_imm(output, input, shift)),
                    Directive::Instruction(ib::shr_s_imm(output, output, shift)),
                ]
            }

            // 64-bit integer instructions
            Op::I64Eqz => todo!(),
            Op::I64Clz => todo!(),
            Op::I64Ctz => todo!(),
            Op::I64Popcnt => todo!(),
            Op::I64ExtendI32S => todo!(),
            Op::I64ExtendI32U => todo!(),
            Op::I64Extend8S => todo!(),
            Op::I64Extend16S => todo!(),
            Op::I64Extend32S => todo!(),

            // Parametric instruction
            Op::Select => {
                // Works like a ternary operator: if the condition (3rd input) is non-zero,
                // select the 1st input, otherwise select the 2nd input.
                let if_set_val = inputs[0].clone();
                let if_zero_val = inputs[1].clone();
                let output = output.unwrap();
                let condition = inputs[2].start;

                let if_set_label = g.new_label(LabelType::Local);
                let continuation_label = g.new_label(LabelType::Local);

                let mut directives = vec![
                    // if condition != 0 jump to if_set_label
                    Directive::JumpIf {
                        target: if_set_label.clone(),
                        condition_reg: condition,
                    },
                ];
                // if jump is not taken, copy the value for "if zero"
                directives.extend(if_zero_val.zip(output.clone()).map(|(src, dest)| {
                    Directive::Instruction(ib::addi(dest as usize, src as usize, F::ZERO))
                }));
                // jump to continuation
                directives.push(Directive::Jump {
                    target: continuation_label.clone(),
                });

                // if jump is taken, copy the value for "if set"
                directives.push(Directive::Label {
                    id: if_set_label,
                    frame_size: None,
                });
                directives.extend(if_set_val.zip(output).map(|(src, dest)| {
                    Directive::Instruction(ib::addi(dest as usize, src as usize, F::ZERO))
                }));

                // continuation label
                directives.push(Directive::Label {
                    id: continuation_label,
                    frame_size: None,
                });

                directives
            }

            // Global instructions
            Op::GlobalGet { global_index: _ } => todo!(),
            Op::GlobalSet { global_index: _ } => todo!(),

            // Memory instructions
            Op::I32Load { memarg: _ } => todo!(),
            Op::I64Load { memarg: _ } => todo!(),
            Op::I32Load8S { memarg: _ } => todo!(),
            Op::I32Load8U { memarg: _ } => todo!(),
            Op::I32Load16S { memarg: _ } => todo!(),
            Op::I32Load16U { memarg: _ } => todo!(),
            Op::I64Load8S { memarg: _ } => todo!(),
            Op::I64Load8U { memarg: _ } => todo!(),
            Op::I64Load16S { memarg: _ } => todo!(),
            Op::I64Load16U { memarg: _ } => todo!(),
            Op::I64Load32S { memarg: _ } => todo!(),
            Op::I64Load32U { memarg: _ } => todo!(),
            Op::I32Store { memarg: _ } => todo!(),
            Op::I64Store { memarg: _ } => todo!(),
            Op::I32Store8 { memarg: _ } => todo!(),
            Op::I32Store16 { memarg: _ } => todo!(),
            Op::I64Store8 { memarg: _ } => todo!(),
            Op::I64Store16 { memarg: _ } => todo!(),
            Op::I64Store32 { memarg: _ } => todo!(),
            Op::MemorySize { mem: _ } => todo!(),
            Op::MemoryGrow { mem: _ } => todo!(),
            Op::MemoryInit {
                data_index: _,
                mem: _,
            } => todo!(),
            Op::MemoryCopy {
                dst_mem: _,
                src_mem: _,
            } => todo!(),
            Op::MemoryFill { mem: _ } => todo!(),
            Op::DataDrop { data_index: _ } => todo!(),

            // Table instructions
            Op::TableInit {
                elem_index: _,
                table: _,
            } => todo!(),
            Op::TableCopy {
                dst_table: _,
                src_table: _,
            } => todo!(),
            Op::TableFill { table: _ } => todo!(),
            Op::TableGet { table: _ } => todo!(),
            Op::TableSet { table: _ } => todo!(),
            Op::TableGrow { table: _ } => todo!(),
            Op::TableSize { table: _ } => todo!(),
            Op::ElemDrop { elem_index: _ } => todo!(),

            // Reference instructions
            Op::RefNull { hty: _ } => todo!(),
            Op::RefIsNull => todo!(),
            Op::RefFunc { function_index: _ } => todo!(),

            // Float instructions
            Op::F32Load { memarg: _ } => todo!(),
            Op::F64Load { memarg: _ } => todo!(),
            Op::F32Store { memarg: _ } => todo!(),
            Op::F64Store { memarg: _ } => todo!(),
            Op::F32Const { value } => {
                let output = output.unwrap().start as usize;
                let value_u = value.bits();
                let value_lo: u16 = (value_u & 0xffff) as u16;
                let value_hi: u16 = ((value_u >> 16) & 0xffff) as u16;
                vec![Directive::Instruction(ib::const_32_imm(
                    output, value_lo, value_hi,
                ))]
            }
            Op::F64Const { value } => {
                let output = output.unwrap().start as usize;
                let value = value.bits();
                let lower = value as u32;
                let lower_lo: u16 = (lower & 0xffff) as u16;
                let lower_hi: u16 = ((lower >> 16) & 0xffff) as u16;
                let upper = (value >> 32) as u32;
                let upper_lo: u16 = (upper & 0xffff) as u16;
                let upper_hi: u16 = ((upper >> 16) & 0xffff) as u16;
                vec![
                    Directive::Instruction(ib::const_32_imm(output, lower_lo, lower_hi)),
                    Directive::Instruction(ib::const_32_imm(output + 1, upper_lo, upper_hi)),
                ]
            }
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

            Op::I32ReinterpretF32
            | Op::F32ReinterpretI32
            | Op::I64ReinterpretF64
            | Op::F64ReinterpretI64 => {
                // TODO: considering we are using a single address space for all types,
                // these reinterpret instruction could be elided at womir level.

                // Just copy the input to the output.
                inputs[0]
                    .clone()
                    .zip(output.unwrap())
                    .map(|(input, output)| {
                        Directive::Instruction(ib::addi(output as usize, input as usize, F::ZERO))
                    })
                    .collect()
            }

            _ => todo!(),
        }
    }
}

impl<F: PrimeField32> Directive<F> {
    fn into_instruction(self, label_map: &HashMap<String, LabelValue>) -> Option<Instruction<F>> {
        match self {
            Directive::Nop | Directive::Label { .. } => None,
            Directive::AllocateFrameI {
                target_frame,
                result_ptr,
            } => {
                let frame_size = label_map.get(&target_frame).unwrap().frame_size.unwrap();
                Some(ib::allocate_frame_imm(
                    result_ptr as usize,
                    frame_size as usize,
                ))
            }
            Directive::Jump { target } => {
                let pc = label_map.get(&target).unwrap().pc;
                Some(ib::jump(pc as usize))
            }
            Directive::JumpIf {
                target,
                condition_reg,
            } => {
                let pc = label_map.get(&target)?.pc;
                Some(ib::jump_if(condition_reg as usize, pc as usize))
            }
            Directive::JumpIfZero {
                target,
                condition_reg,
            } => {
                let pc = label_map.get(&target)?.pc;
                Some(ib::jump_if_zero(condition_reg as usize, pc as usize))
            }
            Directive::Jaaf {
                target,
                new_frame_ptr,
            } => {
                let pc = label_map.get(&target)?.pc;
                Some(ib::jaaf(pc as usize, new_frame_ptr as usize))
            }
            Directive::JaafSave {
                target,
                new_frame_ptr,
                saved_caller_fp,
            } => {
                let pc = label_map.get(&target)?.pc;
                Some(ib::jaaf_save(
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
                Some(ib::call(
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
