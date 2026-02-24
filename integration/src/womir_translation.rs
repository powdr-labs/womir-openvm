use std::{collections::HashMap, ops::Range, vec};

use crate::{
    WomirConfig,
    const_collapse::can_turn_to_lt,
    instruction_builder::{self as ib, AluImm},
};
use arrayvec::ArrayVec;
use itertools::Itertools;
use openvm_circuit::{
    arch::{ExecutionError, Streams, VmExecutor},
    system::memory::{merkle::public_values::extract_public_values, online::LinearMemory},
};
use openvm_instructions::{
    exe::{SparseMemoryImage, VmExe},
    instruction::Instruction,
    program::{DEFAULT_PC_STEP, Program},
    riscv,
};
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_womir_transpiler::{
    BaseAlu64Opcode, BaseAluOpcode, DivRem64Opcode, DivRemOpcode, Eq64Opcode, EqOpcode,
    LessThan64Opcode, LessThanOpcode, Mul64Opcode, MulOpcode, Shift64Opcode, ShiftOpcode,
};
use wasmparser::{MemArg, Operator as Op, ValType};
use womir::{
    interpreter::linker::LabelValue,
    loader::{
        FunctionAsm, FunctionRef, Global, MemoryEntry, Module,
        passes::dag::WasmValue,
        rwm::flattening::Context,
        settings::{
            ComparisonFunction, JumpCondition, LabelType, MaybeConstant, TrapReason, WasmOpInput,
            func_idx_to_label,
        },
    },
    utils::tree::Tree,
};

/// This is our convention for null function references.
///
/// The first value is the function type identifier, and we are reserving u32::MAX for null,
/// so that when indirectly calling a null reference, the type check will trigger a fault.
///
/// The second value is the function's frame size, and doesn't interfere with the WASM
/// instructions that handle function references.
///
/// The third value is the actual address of the function, and no valid function is in
/// position 0, because the linker places the function starting at address 0x4. This
/// value is used by instruction ref.is_null to decide if the reference is null.
const NULL_REF: [u32; 3] = [u32::MAX, 0, 0];

/// Traps from Womir will terminate with its error code plus this offset.
pub const ERROR_CODE_OFFSET: u32 = 100;

pub const ERROR_ABORT_CODE: u32 = 200;

/// Error code emitted in place of unimplemented instructions.
/// This is to allow programs that has such instruction to compile,
/// but the runtime error only happens if they are actually executed.
pub const ERROR_UNIMPLEMENTED_CODE: u32 = 201;

pub struct LinkedProgram<'a, F: PrimeField32> {
    module: Module<'a>,
    label_map: HashMap<String, LabelValue>,
    /// Linked instructions without the startup code:
    linked_instructions: Vec<Instruction<F>>,
    memory_image: SparseMemoryImage,
}

impl<'a, F: PrimeField32> LinkedProgram<'a, F> {
    pub fn new(mut module: Module<'a>, functions: Vec<FunctionAsm<Directive<F>>>) -> Self {
        let (linked_program, mut label_map) = womir::interpreter::linker::link(functions, 1);

        for v in label_map.values_mut() {
            v.pc *= riscv::RV32_REGISTER_NUM_LIMBS as u32;
        }

        let start_offset = linked_program.len();

        let linked_instructions = linked_program
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
        assert_eq!(linked_instructions.len(), start_offset - 1);

        let memory_image = std::mem::take(&mut module.initial_memory)
            .into_iter()
            .flat_map(|(addr, value)| {
                let v = match value {
                    MemoryEntry::Value(v) => v,
                    MemoryEntry::FuncAddr(idx) => {
                        let label = func_idx_to_label(idx);
                        label_map[&label].pc
                    }
                    MemoryEntry::FuncFrameSize(func_idx) => {
                        // In RWM mode, frame sizes are not used.
                        // Return frame_size if available, otherwise 0.
                        let label = func_idx_to_label(func_idx);
                        label_map[&label].frame_size.unwrap_or(0)
                    }
                    MemoryEntry::NullFuncType => {
                        NULL_REF[FunctionRef::<OpenVMSettings<F>>::TYPE_ID]
                    }
                    MemoryEntry::NullFuncFrameSize => {
                        NULL_REF[FunctionRef::<OpenVMSettings<F>>::FUNC_FRAME_SIZE]
                    }
                    MemoryEntry::NullFuncAddr => {
                        NULL_REF[FunctionRef::<OpenVMSettings<F>>::FUNC_ADDR]
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
                        Some(((ROM_ID, addr + i as u32), byte as u8))
                    } else {
                        None
                    }
                })
            })
            .collect();

        Self {
            module,
            label_map,
            linked_instructions,
            memory_image,
        }
    }

    pub fn program_with_entry_point(&self, entry_point: &str) -> VmExe<F> {
        // Create the startup code to call the entry point function.
        let entry_point = &self.label_map[entry_point];
        let entry_point_start = self.linked_instructions.len();
        let mut linked_instructions = self.linked_instructions.clone();
        linked_instructions.extend(create_startup_code(&self.module, entry_point));

        // TODO: make womir read and carry debug info
        // The first instruction was removed, which was a nop inserted by the linker,
        // so we need to set PC base to DEFAULT_PC_STEP, skipping position 0.
        let program = Program::new_without_debug_infos(&linked_instructions, DEFAULT_PC_STEP);

        // Create the executor using the current memory image.
        VmExe::new(program)
            .with_pc_start(((1 + entry_point_start) * riscv::RV32_REGISTER_NUM_LIMBS) as u32)
            .with_init_memory(self.memory_image.clone())
    }

    pub fn execute(
        &mut self,
        vm_config: WomirConfig,
        entry_point: &str,
        inputs: impl Into<Streams<F>>,
    ) -> Result<Vec<u8>, ExecutionError> {
        let exe = self.program_with_entry_point(entry_point);

        let vm = VmExecutor::new(vm_config.clone()).unwrap();
        let instance = vm.instance(&exe).unwrap();
        let final_state = instance.execute(inputs, None)?;
        let public_values = extract_public_values(
            vm_config.system.num_public_values,
            &final_state.memory.memory,
        );

        // TODO the block below may not work after the refactor.
        // TODO: extract_public_values() already converts the final_memory to a MemoryImage<F>,
        // and this is a kinda expensive redundant work. Find a way to do it only once.
        self.memory_image = final_state
            .memory
            .memory
            .get_memory()
            .iter()
            .enumerate()
            .flat_map(|(addr_space, linear_mem)| {
                linear_mem
                    .as_slice()
                    .iter()
                    .enumerate()
                    .map(move |(addr, v)| (addr_space as u32, addr as u32, *v))
            })
            .filter_map(|(addr_space, addr, v)| (v != 0).then_some(((addr_space, addr), v)))
            .collect();

        Ok(public_values)
    }
}

fn create_startup_code<F>(ctx: &Module, entry_point: &LabelValue) -> Vec<Instruction<F>>
where
    F: PrimeField32,
{
    use openvm_instructions::riscv::RV32_REGISTER_NUM_LIMBS as NUM_LIMBS;

    let entry_point_func_type = &ctx.get_func_type(entry_point.func_idx.unwrap()).ty;
    let params = entry_point_func_type.params();
    let results = entry_point_func_type.results();
    let num_input_words = womir::loader::word_count_types::<OpenVMSettings<F>>(params) as usize;
    let num_output_words = womir::loader::word_count_types::<OpenVMSettings<F>>(results) as usize;

    // Registers used by startup code (relative to initial FP):
    //   reg 0: zero_reg (holds value 0)
    //   reg 1: scratch_reg (holds safe memory address for hint_storew)
    let zero_reg: usize = 0;
    let scratch_reg: usize = 1;

    // Frame offset for entry function = 2 (startup uses 2 registers)
    let frame_offset: usize = 2;

    // Callee frame layout (relative to callee's FP = initial_FP + frame_offset * NUM_LIMBS):
    //   reg 0..num_args: function parameters
    //   reg 0..num_outputs: function return values (overlapping with params)
    //   reg max(args,outs): saved_ret_pc
    //   reg max(args,outs)+1: saved_caller_fp
    let max_io = num_input_words.max(num_output_words);
    let save_pc_reg = max_io;
    let save_fp_reg = max_io + 1;

    // Safe scratch address for hint_storew in memory address space.
    // This address is near the top of the 512MB memory address space (1 << 29),
    // high enough that no WASM linear memory will reach it.
    let safe_addr: u32 = 0x1FFF_FFE0;

    let mut code = vec![
        // 1. Set zero_reg = 0
        ib::const_32_imm(zero_reg, 0, 0),
        // 2. Set scratch_reg = SAFE_ADDR
        ib::const_32_imm(
            scratch_reg,
            (safe_addr & 0xFFFF) as u16,
            (safe_addr >> 16) as u16,
        ),
    ];

    // 3. For each input word, read from hint stream into callee's argument register
    for i in 0..num_input_words {
        code.push(ib::prepare_read());
        code.push(ib::hint_storew(scratch_reg)); // skip length word
        code.push(ib::hint_storew(scratch_reg)); // write data to MEM[SAFE_ADDR]
        code.push(ib::loadw(frame_offset + i, scratch_reg, 0)); // load from MEM[SAFE_ADDR] to callee's arg register
    }

    // 4. Call the entry function
    //    save_pc and save_fp are relative to callee's FP
    //    fp_offset must be frame_offset * NUM_LIMBS because ib::call does NOT multiply fp_offset
    code.push(ib::call(
        save_pc_reg,
        save_fp_reg,
        entry_point.pc as usize,
        frame_offset * NUM_LIMBS,
    ));

    // 5. Reveal output values
    for i in 0..num_output_words {
        code.push(ib::reveal_imm(frame_offset + i, zero_reg, i * 4));
    }

    // 6. Halt
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
    Call {
        target_pc: String,
        new_frame_offset: u32,
        saved_ret_pc: u32,
        saved_caller_fp: u32,
    },
    ConstFuncAddr {
        func_idx: u32,
        reg_dest: u32,
    },
    Instruction(Instruction<F>),
}

type Ctx<'a, 'b> = Context<'a, 'b>;

#[derive(Debug)]
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

impl<F: PrimeField32> womir::loader::settings::Settings for OpenVMSettings<F> {
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

    fn get_const_collapse_processor() -> Option<impl Fn(&Op, &[MaybeConstant])> {
        Some(crate::const_collapse::collapse_const_if_possible)
    }

    fn to_plain_local_jump(directive: Directive<F>) -> Result<String, Directive<F>> {
        if let Directive::Jump { target } = directive {
            Ok(target)
        } else {
            Err(directive)
        }
    }

    fn is_label(directive: &Directive<F>) -> Option<&str> {
        if let Directive::Label { id, .. } = directive {
            Some(id)
        } else {
            None
        }
    }

    fn emit_jump(&self, target: String) -> Directive<F> {
        Directive::Jump { target }
    }
}

#[allow(refining_impl_trait)]
impl<'a, F: PrimeField32> womir::loader::rwm::settings::Settings<'a> for OpenVMSettings<F> {
    fn emit_label(&self, _c: &mut Ctx<'a, '_>, name: String) -> Directive<F> {
        Directive::Label {
            id: name,
            frame_size: None,
        }
    }

    fn emit_trap(&self, _c: &mut Ctx<'a, '_>, error_code: TrapReason) -> Directive<F> {
        Directive::Instruction(ib::trap(error_code as u32 as usize))
    }

    fn emit_copy(&self, _c: &mut Ctx<'a, '_>, src_reg: u32, dest_reg: u32) -> Directive<F> {
        Directive::Instruction(ib::add_imm(
            dest_reg as usize,
            src_reg as usize,
            AluImm::from(0),
        ))
    }

    fn emit_conditional_jump(
        &self,
        _c: &mut Ctx<'a, '_>,
        condition_type: JumpCondition,
        label: String,
        condition_ptr: Range<u32>,
    ) -> Directive<F> {
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
        c: &mut Ctx<'a, '_>,
        cmp: ComparisonFunction,
        value_ptr: Range<u32>,
        immediate: u32,
        label: String,
    ) -> Vec<Directive<F>> {
        let comparison = c.allocate_tmp_type::<Self>(ValType::I32);

        let mut directives = if let Ok(imm_f) = AluImm::try_from(immediate) {
            // If immediate fits into field, we can save one instruction:
            let cmp_insn = match cmp {
                ComparisonFunction::Equal => ib::eq_imm,
                ComparisonFunction::GreaterThanOrEqualUnsigned
                | ComparisonFunction::LessThanUnsigned => ib::lt_u_imm,
            };
            vec![Directive::Instruction(cmp_insn(
                comparison.start as usize,
                value_ptr.start as usize,
                imm_f,
            ))]
        } else {
            // Otherwise, we need to compose the immediate into a register:
            let cmp_insn = match cmp {
                ComparisonFunction::Equal => ib::eq,
                ComparisonFunction::GreaterThanOrEqualUnsigned
                | ComparisonFunction::LessThanUnsigned => ib::lt_u,
            };

            let const_value = c.allocate_tmp_type::<Self>(ValType::I32);

            let imm_lo: u16 = (immediate & 0xffff) as u16;
            let imm_hi: u16 = ((immediate >> 16) & 0xffff) as u16;

            vec![
                Directive::Instruction(ib::const_32_imm(
                    const_value.start as usize,
                    imm_lo,
                    imm_hi,
                )),
                Directive::Instruction(cmp_insn(
                    comparison.start as usize,
                    value_ptr.start as usize,
                    const_value.start as usize,
                )),
            ]
        };

        // We use "less than" to compare both "less than" and "greater than or equal",
        // so, if it is the later, we must jump if the condition is false.
        directives.push(match cmp {
            ComparisonFunction::Equal | ComparisonFunction::LessThanUnsigned => Directive::JumpIf {
                target: label,
                condition_reg: comparison.start,
            },
            ComparisonFunction::GreaterThanOrEqualUnsigned => Directive::JumpIfZero {
                target: label,
                condition_reg: comparison.start,
            },
        });

        directives
    }

    fn emit_relative_jump(&self, _c: &mut Ctx<'a, '_>, offset_ptr: Range<u32>) -> Directive<F> {
        Directive::Instruction(ib::skip(offset_ptr.start as usize))
    }

    fn emit_return(
        &self,
        _c: &mut Ctx<'a, '_>,
        ret_pc_ptr: Range<u32>,
        caller_fp_ptr: Range<u32>,
    ) -> Directive<F> {
        Directive::Instruction(ib::ret(
            ret_pc_ptr.start as usize,
            caller_fp_ptr.start as usize,
        ))
    }

    fn emit_imported_call(
        &self,
        c: &mut Ctx<'a, '_>,
        module: &'a str,
        function: &'a str,
        inputs: &[WasmOpInput],
        outputs: Vec<Range<u32>>,
    ) -> Vec<Directive<F>> {
        assert!(
            outputs.is_empty(),
            "imported functions should not need outputs"
        );
        match (module, function) {
            ("env", "__debug_print") => {
                let mem_start = c
                    .module()
                    .linear_memory_start()
                    .expect("no memory allocated");

                assert!(mem_start < (1 << 16));
                let mem_ptr = inputs[0].as_register().unwrap().start as usize;
                let num_bytes = inputs[1].as_register().unwrap().start as usize;
                vec![Directive::Instruction(ib::debug_print(
                    mem_ptr,
                    num_bytes,
                    mem_start as u16,
                ))]
            }
            ("env", "__hint_input") => {
                // prepare an object to be read
                vec![Directive::Instruction(ib::prepare_read())]
            }
            ("env", "__hint_buffer") => {
                let mem_start = c
                    .module()
                    .linear_memory_start()
                    .expect("no memory allocated");

                let mem_ptr = inputs[0].as_register().unwrap().start as usize;
                let num_words = inputs[1].as_register().unwrap().start as usize;
                let mut directives = vec![];
                // Use a temporary register for the adjusted pointer to avoid
                // clobbering the input register (which may be a live WASM local).
                let effective_ptr = if mem_start > 0 {
                    let tmp = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    directives.push(Directive::Instruction(ib::add_imm(
                        tmp,
                        mem_ptr,
                        AluImm::try_from(mem_start).unwrap(),
                    )));
                    tmp
                } else {
                    mem_ptr
                };
                directives.push(Directive::Instruction(ib::hint_buffer(
                    num_words,
                    effective_ptr,
                )));
                directives
            }
            ("env", "abort") => {
                vec![Directive::Instruction(ib::abort())]
            }
            ("gojs", _) => {
                // Just NOP for GoJS intrinsics
                vec![]
            }
            _ => unimplemented!(
                "Imported function `{}` from module `{}` is not supported",
                function,
                module
            ),
        }
    }

    fn emit_function_call(
        &self,
        _c: &mut Ctx<'a, '_>,
        function_label: String,
        function_frame_offset: u32,
        saved_ret_pc_ptr: Range<u32>,
        saved_caller_fp_ptr: Range<u32>,
    ) -> Directive<F> {
        Directive::Call {
            target_pc: function_label,
            new_frame_offset: function_frame_offset,
            saved_ret_pc: saved_ret_pc_ptr.start,
            saved_caller_fp: saved_caller_fp_ptr.start,
        }
    }

    fn emit_indirect_call(
        &self,
        _c: &mut Ctx<'a, '_>,
        target_pc_ptr: Range<u32>,
        function_frame_offset: u32,
        saved_ret_pc_ptr: Range<u32>,
        saved_caller_fp_ptr: Range<u32>,
    ) -> Directive<F> {
        use openvm_instructions::riscv::RV32_REGISTER_NUM_LIMBS as NUM_LIMBS;
        Directive::Instruction(ib::call_indirect(
            saved_ret_pc_ptr.start as usize,
            saved_caller_fp_ptr.start as usize,
            target_pc_ptr.start as usize,
            function_frame_offset as usize * NUM_LIMBS,
        ))
    }

    fn emit_wasm_op(
        &self,
        c: &mut Ctx<'a, '_>,
        op: Op<'a>,
        inputs: &[WasmOpInput],
        output: Option<Range<u32>>,
    ) -> Tree<Directive<F>> {
        let module = c.module();
        output
            .as_ref()
            .and_then(|output| {
                translate_most_binary_ops(&op, inputs, output)
                    .or_else(|| translate_less_than_ops(&op, inputs, output))
                    .or_else(|| translate_greater_than_ops(&op, inputs, output))
                    .map(|instruction| Directive::Instruction(instruction).into())
                    .or_else(|| translate_complex_ins_with_const(c, module, &op, inputs, output))
            })
            .unwrap_or_else(|| translate_complex_ins(c, module, op, inputs, output))
    }
}

impl<F: PrimeField32> Directive<F> {
    fn into_instruction(self, label_map: &HashMap<String, LabelValue>) -> Option<Instruction<F>> {
        match self {
            Directive::Nop | Directive::Label { .. } => None,
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
            Directive::Call {
                target_pc,
                new_frame_offset,
                saved_ret_pc,
                saved_caller_fp,
            } => {
                use openvm_instructions::riscv::RV32_REGISTER_NUM_LIMBS as NUM_LIMBS;
                let pc = label_map.get(&target_pc)?.pc;
                Some(ib::call(
                    saved_ret_pc as usize,
                    saved_caller_fp as usize,
                    pc as usize,
                    new_frame_offset as usize * NUM_LIMBS,
                ))
            }
            Directive::ConstFuncAddr { func_idx, reg_dest } => {
                let label = func_idx_to_label(func_idx);
                let pc = label_map.get(&label)?.pc;
                Some(ib::const_32_imm(
                    reg_dest as usize,
                    pc as u16,
                    (pc >> 16) as u16,
                ))
            }
            Directive::Instruction(i) => Some(i),
        }
    }
}

/// Translate most binary operations (those with two inputs and one output).
///
/// Except for relational operations, which are handled separately.
fn translate_most_binary_ops<'a, F: PrimeField32>(
    op: &Op<'a>,
    inputs: &[WasmOpInput],
    output: &Range<u32>,
) -> Option<Instruction<F>> {
    use openvm_instructions::LocalOpcode;

    let binary_op = match op {
        // 32-bit integer instructions
        Op::I32Eq => EqOpcode::EQ.global_opcode(),
        Op::I32Ne => EqOpcode::NEQ.global_opcode(),
        Op::I32Add => BaseAluOpcode::ADD.global_opcode(),
        Op::I32Sub => BaseAluOpcode::SUB.global_opcode(),
        Op::I32And => BaseAluOpcode::AND.global_opcode(),
        Op::I32Or => BaseAluOpcode::OR.global_opcode(),
        Op::I32Xor => BaseAluOpcode::XOR.global_opcode(),
        Op::I32Mul => MulOpcode::MUL.global_opcode(),
        Op::I32DivS => DivRemOpcode::DIV.global_opcode(),
        Op::I32DivU => DivRemOpcode::DIVU.global_opcode(),
        Op::I32RemS => DivRemOpcode::REM.global_opcode(),
        Op::I32RemU => DivRemOpcode::REMU.global_opcode(),
        Op::I32Shl => ShiftOpcode::SLL.global_opcode(),
        Op::I32ShrS => ShiftOpcode::SRA.global_opcode(),
        Op::I32ShrU => ShiftOpcode::SRL.global_opcode(),

        // 64-bit integer instructions
        Op::I64Eq => Eq64Opcode::EQ.global_opcode(),
        Op::I64Ne => Eq64Opcode::NEQ.global_opcode(),
        Op::I64Add => BaseAlu64Opcode::ADD.global_opcode(),
        Op::I64Sub => BaseAlu64Opcode::SUB.global_opcode(),
        Op::I64And => BaseAlu64Opcode::AND.global_opcode(),
        Op::I64Or => BaseAlu64Opcode::OR.global_opcode(),
        Op::I64Xor => BaseAlu64Opcode::XOR.global_opcode(),
        Op::I64Mul => Mul64Opcode::MUL.global_opcode(),
        Op::I64DivS => DivRem64Opcode::DIV.global_opcode(),
        Op::I64DivU => DivRem64Opcode::DIVU.global_opcode(),
        Op::I64RemS => DivRem64Opcode::REM.global_opcode(),
        Op::I64RemU => DivRem64Opcode::REMU.global_opcode(),
        Op::I64Shl => Shift64Opcode::SLL.global_opcode(),
        Op::I64ShrS => Shift64Opcode::SRA.global_opcode(),
        Op::I64ShrU => Shift64Opcode::SRL.global_opcode(),

        // Float instructions
        Op::F32Eq
        | Op::F32Ne
        | Op::F32Lt
        | Op::F32Gt
        | Op::F32Le
        | Op::F32Ge
        | Op::F64Eq
        | Op::F64Ne
        | Op::F64Lt
        | Op::F64Gt
        | Op::F64Le
        | Op::F64Ge
        | Op::F32Add
        | Op::F32Sub
        | Op::F32Mul
        | Op::F32Div
        | Op::F32Min
        | Op::F32Max
        | Op::F32Copysign
        | Op::F64Add
        | Op::F64Sub
        | Op::F64Mul
        | Op::F64Div
        | Op::F64Min
        | Op::F64Max
        | Op::F64Copysign => {
            return Some(ib::trap(ERROR_UNIMPLEMENTED_CODE as usize));
        }

        // Not an operation we handle here
        _ => return None,
    };

    let opcode = binary_op.as_usize();
    let output = output.start as usize;

    Some(match inputs {
        [WasmOpInput::Register(input1), WasmOpInput::Register(input2)] => {
            // Case of two register inputs
            ib::instr_r(opcode, output, input1.start as usize, input2.start as usize)
        }
        [WasmOpInput::Register(reg), WasmOpInput::Constant(c)]
        | [WasmOpInput::Constant(c), WasmOpInput::Register(reg)] => {
            // Case of one register input and one constant input.

            // The order doesn't matter, because only commutative operations will
            // have the constant operand on the left side, as const folding ensures.

            // The constant folding step guarantees that the constant can be safely
            // truncated to i16.
            let c = const_i16_as_field(c);
            ib::instr_i(opcode, output, reg.start as usize, c)
        }
        _ => unreachable!("combination of inputs not possible for binary op"),
    })
}

fn translate_less_than_ops<'a, F: PrimeField32>(
    op: &Op<'a>,
    inputs: &[WasmOpInput],
    output: &Range<u32>,
) -> Option<Instruction<F>> {
    use openvm_instructions::LocalOpcode;

    let rel_op = match op {
        Op::I32LtS => LessThanOpcode::SLT.global_opcode(),
        Op::I32LtU => LessThanOpcode::SLTU.global_opcode(),
        Op::I64LtS => LessThan64Opcode::SLT.global_opcode(),
        Op::I64LtU => LessThan64Opcode::SLTU.global_opcode(),
        _ => return None,
    };

    let opcode = rel_op.as_usize();
    let output = output.start as usize;

    Some(match inputs {
        [WasmOpInput::Register(input1), WasmOpInput::Register(input2)] => {
            ib::instr_r(opcode, output, input1.start as usize, input2.start as usize)
        }
        [WasmOpInput::Register(reg), WasmOpInput::Constant(c)] => {
            // The constant folding step guarantees that the constant can be safely
            // truncated to i16.
            let c = const_i16_as_field(c);
            ib::instr_i(opcode, output, reg.start as usize, c)
        }
        [
            WasmOpInput::Constant(WasmValue::I32(0) | WasmValue::I64(0)),
            WasmOpInput::Register(reg),
        ] => {
            // This must be the case of unsigned "0 < reg", where we turn it into "reg != 0".
            ib::neq_imm(output, reg.start as usize, AluImm::from(0))
        }
        _ => unreachable!("combination of inputs not possible for binary op"),
    })
}

fn translate_greater_than_ops<'a, F: PrimeField32>(
    op: &Op<'a>,
    inputs: &[WasmOpInput],
    output: &Range<u32>,
) -> Option<Instruction<F>> {
    use openvm_instructions::LocalOpcode;

    // GT instructions are just reversed LT
    let op = match op {
        Op::I32GtS => LessThanOpcode::SLT.global_opcode(),
        Op::I32GtU => LessThanOpcode::SLTU.global_opcode(),
        Op::I64GtS => LessThan64Opcode::SLT.global_opcode(),
        Op::I64GtU => LessThan64Opcode::SLTU.global_opcode(),
        _ => return None,
    };

    let op = op.as_usize();
    let output = output.start as usize;
    Some(match inputs {
        [
            WasmOpInput::Register(greater_side),
            WasmOpInput::Register(lesser_side),
        ] => {
            // Case of two register inputs
            ib::instr_r(
                op,
                output,
                lesser_side.start as usize,
                greater_side.start as usize,
            )
        }
        [WasmOpInput::Constant(c), WasmOpInput::Register(reg)] => {
            // Case of one register input and one constant input.
            //
            // The constant is only allowed to be on the left side, because GT
            // is just LT with the operands reversed, and LT expects the immediate as
            // the greater side.
            let c = const_i16_as_field(c);
            ib::instr_i(op, output, reg.start as usize, c)
        }
        [
            WasmOpInput::Register(reg),
            WasmOpInput::Constant(WasmValue::I32(0) | WasmValue::I64(0)),
        ] => {
            // This must be the case of unsigned "reg > 0", where we turn it into "reg != 0".
            ib::neq_imm(output, reg.start as usize, AluImm::from(0))
        }
        _ => unreachable!("combination of inputs not possible for GT op"),
    })
}

/// Translates the complex instructions that supports constant inlining
fn translate_complex_ins_with_const<'a, F: PrimeField32>(
    c: &mut Ctx<'a, '_>,
    module: &Module<'a>,
    op: &Op,
    inputs: &[WasmOpInput],
    output: &Range<u32>,
) -> Option<Tree<Directive<F>>> {
    use openvm_instructions::LocalOpcode;

    Some(match op {
        Op::I32GeS
        | Op::I32GeU
        | Op::I64GeS
        | Op::I64GeU
        | Op::I32LeS
        | Op::I32LeU
        | Op::I64LeS
        | Op::I64LeU => {
            let inverse_result =
                c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
            let output = output.start as usize;

            let comp_instruction = match [&inputs[0], &inputs[1]] {
                [WasmOpInput::Register(input1), WasmOpInput::Register(input2)] => {
                    let inverse_op = match op {
                        Op::I32GeS => ib::lt_s,
                        Op::I32GeU => ib::lt_u,
                        Op::I64GeS => ib::lt_s_64,
                        Op::I64GeU => ib::lt_u_64,
                        Op::I32LeS => ib::gt_s,
                        Op::I32LeU => ib::gt_u,
                        Op::I64LeS => ib::gt_s_64,
                        Op::I64LeU => ib::gt_u_64,
                        _ => unreachable!(),
                    };

                    let input1 = input1.start as usize;
                    let input2 = input2.start as usize;

                    // Perform the inverse operation and invert the result
                    inverse_op(inverse_result, input1, input2)
                }
                [WasmOpInput::Register(reg), WasmOpInput::Constant(c)]
                | [WasmOpInput::Constant(c), WasmOpInput::Register(reg)] => {
                    // Handle the cases where unsigned "reg >= 1" and "1 <= reg" can be turned into "reg != 0"
                    if matches!(
                        (&op, &inputs[1]),
                        (
                            Op::I32GeU | Op::I64GeU,
                            WasmOpInput::Constant(WasmValue::I32(1) | WasmValue::I64(1))
                        )
                    ) || matches!(
                        (&op, &inputs[0]),
                        (
                            Op::I32LeU | Op::I64LeU,
                            WasmOpInput::Constant(WasmValue::I32(1) | WasmValue::I64(1))
                        )
                    ) {
                        return Some(
                            Directive::Instruction(ib::neq_imm(
                                output,
                                reg.start as usize,
                                AluImm::from(0),
                            ))
                            .into(),
                        );
                    }

                    let opcode = match op {
                        Op::I32GeS | Op::I32LeS => LessThanOpcode::SLT.global_opcode(),
                        Op::I32GeU | Op::I32LeU => LessThanOpcode::SLTU.global_opcode(),
                        Op::I64GeS | Op::I64LeS => LessThan64Opcode::SLT.global_opcode(),
                        Op::I64GeU | Op::I64LeU => LessThan64Opcode::SLTU.global_opcode(),
                        _ => unreachable!(),
                    };

                    // Check whether this is the case where "c >= r" or "r <= c" can be turned into "r < c + 1".
                    if ((matches!(op, Op::I32GeS | Op::I32GeU | Op::I64GeS | Op::I64GeU)
                        && matches!(&inputs[0], WasmOpInput::Constant(_)))
                        || (matches!(op, Op::I32LeS | Op::I32LeU | Op::I64LeS | Op::I64LeU)
                            && matches!(&inputs[0], WasmOpInput::Register(_))))
                        && let Some(inc_c) = can_turn_to_lt(op, c)
                    {
                        // The operation can be turned into a single less-than comparison with incremented constant.
                        // There is no need to do the inverse operation and invert the result.
                        return Some(
                            Directive::Instruction(ib::instr_i(
                                opcode.as_usize(),
                                output,
                                reg.start as usize,
                                inc_c,
                            ))
                            .into(),
                        );
                    }

                    let c = const_i16_as_field(c);
                    ib::instr_i(opcode.as_usize(), inverse_result, reg.start as usize, c)
                }
                _ => unreachable!("combination of inputs not possible for GE/LE operations"),
            };

            // Perform the inverse operation and invert the result
            vec![
                Directive::Instruction(comp_instruction),
                Directive::Instruction(ib::eq_imm(output, inverse_result, AluImm::from(0))),
            ]
            .into()
        }
        Op::I32Rotl => translate_rot::<F, I32Rot>(c, RotDirection::Left, inputs, output),
        Op::I32Rotr => translate_rot::<F, I32Rot>(c, RotDirection::Right, inputs, output),
        Op::I64Rotl => translate_rot::<F, I64Rot>(c, RotDirection::Left, inputs, output),
        Op::I64Rotr => translate_rot::<F, I64Rot>(c, RotDirection::Right, inputs, output),
        Op::Select | Op::TypedSelect { .. } => {
            // Works like a ternary operator: if the condition (3rd input) is non-zero,
            // select the 1st input, otherwise select the 2nd input.
            let condition = inputs[2].as_register().unwrap().start;

            // The directives to set the output to either of the alternatives.
            // For multi-word register copies, use memmove strategy (reverse order
            // when dest >= src) to avoid clobbering source words before they're read.
            let [if_non_zero, if_zero] =
                [&inputs[0], &inputs[1]].map(|alternative| match alternative {
                    WasmOpInput::Register(r) => {
                        let mut pairs: Vec<_> = r.clone().zip(output.clone()).collect();
                        if output.start >= r.start {
                            pairs.reverse();
                        }
                        pairs
                            .into_iter()
                            .map(|(src, dest)| {
                                Directive::Instruction(ib::add_imm(
                                    dest as usize,
                                    src as usize,
                                    AluImm::from(0),
                                ))
                            })
                            .collect_vec()
                    }
                    WasmOpInput::Constant(value) => const_wasm_value(module, value, output.clone()),
                });

            let non_zero_label = c.new_label(LabelType::Local);
            let continuation_label = c.new_label(LabelType::Local);

            Tree::Node(vec![
                // if condition != 0 jump to non_zero_label
                Tree::Leaf(Directive::JumpIf {
                    target: non_zero_label.clone(),
                    condition_reg: condition,
                }),
                // if jump is not taken, copy the value for "if zero"
                if_zero.into(),
                // jump to continuation
                Directive::Jump {
                    target: continuation_label.clone(),
                }
                .into(),
                // alternative label
                Directive::Label {
                    id: non_zero_label,
                    frame_size: None,
                }
                .into(),
                // if jump is taken, copy the value for "if set"
                if_non_zero.into(),
                // continuation label
                Directive::Label {
                    id: continuation_label,
                    frame_size: None,
                }
                .into(),
            ])
        }
        _ => return None,
    })
}

fn translate_complex_ins<'a, F: PrimeField32>(
    c: &mut Ctx<'a, '_>,
    module: &Module<'a>,
    op: Op,
    inputs: &[WasmOpInput],
    output: Option<Range<u32>>,
) -> Tree<Directive<F>> {
    // Handle the remaining operations, whose inputs are all registers
    let inputs = inputs
        .iter()
        .map(|input| {
            input
                .as_register()
                .expect("input must be a register")
                .clone()
        })
        .collect_vec();
    match op {
        Op::Nop => Tree::Empty,

        Op::I32Const { value } => {
            let output = output.unwrap().start as usize;
            let value_u = value as u32;
            let imm_lo: u16 = (value_u & 0xffff) as u16;
            let imm_hi: u16 = ((value_u >> 16) & 0xffff) as u16;
            Directive::Instruction(ib::const_32_imm(output, imm_lo, imm_hi)).into()
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
            .into()
        }

        Op::I32Eqz => {
            let input = inputs[0].start as usize;
            let output = output.unwrap().start as usize;
            Directive::Instruction(ib::eq_imm(output, input, AluImm::from(0))).into()
        }
        Op::I64Eqz => {
            let input = inputs[0].start as usize;
            let output = output.unwrap().start as usize;
            Directive::Instruction(ib::eq_imm_64(output, input, AluImm::from(0))).into()
        }

        Op::I32WrapI64 => {
            // TODO: considering we are using a single address space for both i32 and i64,
            // this instruction could be elided at womir level.

            let lower_limb = inputs[0].start as usize;
            // The higher limb is ignored.
            let output = output.unwrap().start as usize;

            // Just copy the lower limb to the output.
            Directive::Instruction(ib::add_imm(output, lower_limb, AluImm::from(0))).into()
        }
        Op::I32Extend8S | Op::I32Extend16S => {
            let input = inputs[0].start as usize;
            let output = output.unwrap().start as usize;

            let tmp = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;

            let shift = match op {
                Op::I32Extend8S => 24_i16,
                Op::I32Extend16S => 16_i16,
                _ => unreachable!(),
            };

            // Left shift followed by arithmetic right shift
            vec![
                Directive::Instruction(ib::shl_imm(tmp, input, shift)),
                Directive::Instruction(ib::shr_s_imm(output, tmp, shift)),
            ]
            .into()
        }
        Op::I64Extend8S | Op::I64Extend16S | Op::I64Extend32S => {
            let input = inputs[0].start as usize;
            let output = output.unwrap().start as usize;

            let tmp = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I64).start as usize;

            let shift = match op {
                Op::I64Extend8S => 56_i16,
                Op::I64Extend16S => 48_i16,
                Op::I64Extend32S => 32_i16,
                _ => unreachable!(),
            };

            // Left shift followed by arithmetic right shift
            vec![
                Directive::Instruction(ib::shl_imm_64(tmp, input, shift)),
                Directive::Instruction(ib::shr_s_imm_64(output, tmp, shift)),
            ]
            .into()
        }

        // 64-bit integer instructions
        Op::I64ExtendI32S => {
            let input = inputs[0].start as usize;
            let output = output.unwrap().start as usize;

            let high_shifted =
                c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I64).start as usize;

            vec![
                // Copy the 32 bit values to the high 32 bits of the temporary value.
                // Leave the low bits undefined.
                Directive::Instruction(ib::add_imm(high_shifted + 1, input, AluImm::from(0))),
                // shr will read 64 bits, so we need to zero the other half due to WOM
                Directive::Instruction(ib::const_32_imm(high_shifted, 0, 0)),
                // Arithmetic shift right to fill the high bits with the sign bit.
                Directive::Instruction(ib::shr_s_imm_64(output, high_shifted, 32_i16)),
            ]
            .into()
        }
        Op::I64ExtendI32U => {
            let input = inputs[0].start as usize;
            let output = output.unwrap().start as usize;

            vec![
                // Copy the 32 bit value to the low 32 bits of the output.
                Directive::Instruction(ib::add_imm(output, input, AluImm::from(0))),
                // Zero the high 32 bits.
                Directive::Instruction(ib::const_32_imm(output + 1, 0, 0)),
            ]
            .into()
        }

        // Global instructions
        Op::GlobalSet { global_index } => {
            let Global::Mutable(allocated_var) = &module.globals[global_index as usize] else {
                unreachable!()
            };

            store_to_const_addr(c, allocated_var.address, inputs[0].clone()).into()
        }
        Op::GlobalGet { global_index } => {
            let global = &module.globals[global_index as usize];
            match global {
                Global::Mutable(allocated_var) => {
                    load_from_const_addr(c, allocated_var.address, output.unwrap())
                        .0
                        .into()
                }
                Global::Immutable(op) => translate_complex_ins(c, module, op.clone(), &[], output),
            }
        }

        Op::I32Load { memarg } => {
            let imm = mem_offset(memarg, module);
            let base_addr = inputs[0].start as usize;
            let output = output.unwrap().start as usize;
            match memarg.align {
                0 => {
                    // read four bytes
                    let b0 = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let b1 = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let b2 = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let b3 = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let b1_shifted =
                        c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let b2_shifted =
                        c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let b3_shifted =
                        c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let lo = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let hi = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    vec![
                        Directive::Instruction(ib::loadbu(b0, base_addr, imm)),
                        Directive::Instruction(ib::loadbu(b1, base_addr, imm + 1)),
                        Directive::Instruction(ib::loadbu(b2, base_addr, imm + 2)),
                        Directive::Instruction(ib::loadbu(b3, base_addr, imm + 3)),
                        Directive::Instruction(ib::shl_imm(b1_shifted, b1, 8_i16)),
                        Directive::Instruction(ib::shl_imm(b2_shifted, b2, 16_i16)),
                        Directive::Instruction(ib::shl_imm(b3_shifted, b3, 24_i16)),
                        Directive::Instruction(ib::or(lo, b0, b1_shifted)),
                        Directive::Instruction(ib::or(hi, b2_shifted, b3_shifted)),
                        Directive::Instruction(ib::or(output, lo, hi)),
                    ]
                    .into()
                }
                1 => {
                    // read two half words
                    let hi = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let hi_shifted =
                        c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let lo = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    vec![
                        Directive::Instruction(ib::loadhu(lo, base_addr, imm)),
                        Directive::Instruction(ib::loadhu(hi, base_addr, imm + 2)),
                        Directive::Instruction(ib::shl_imm(hi_shifted, hi, 16_i16)),
                        Directive::Instruction(ib::or(output, lo, hi_shifted)),
                    ]
                    .into()
                }
                2.. => Directive::Instruction(ib::loadw(output, base_addr, imm)).into(),
            }
        }

        Op::I32Load16U { memarg } | Op::I32Load16S { memarg } => {
            let imm = mem_offset(memarg, module);
            let base_addr = inputs[0].start as usize;
            let output = output.unwrap().start as usize;

            match memarg.align {
                0 => {
                    // read four bytes
                    let b0 = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let b1 = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let b1_shifted =
                        c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    vec![
                        Directive::Instruction(ib::loadbu(b0, base_addr, imm)),
                        if let Op::I32Load16S { .. } = op {
                            Directive::Instruction(ib::loadb(b1, base_addr, imm + 1))
                        } else {
                            Directive::Instruction(ib::loadbu(b1, base_addr, imm + 1))
                        },
                        Directive::Instruction(ib::shl_imm(b1_shifted, b1, 8_i16)),
                        Directive::Instruction(ib::or(output, b0, b1_shifted)),
                    ]
                    .into()
                }
                1.. => if let Op::I32Load16S { .. } = op {
                    Directive::Instruction(ib::loadh(output, base_addr, imm))
                } else {
                    Directive::Instruction(ib::loadhu(output, base_addr, imm))
                }
                .into(),
            }
        }
        Op::I32Load8U { memarg } => {
            let imm = mem_offset(memarg, module);
            let base_addr = inputs[0].start as usize;
            let output = output.unwrap().start as usize;

            Directive::Instruction(ib::loadbu(output, base_addr, imm)).into()
        }
        Op::I32Load8S { memarg } => {
            let imm = mem_offset(memarg, module);
            let base_addr = inputs[0].start as usize;
            let output = output.unwrap().start as usize;

            Directive::Instruction(ib::loadb(output, base_addr, imm)).into()
        }
        Op::I64Load { memarg } => {
            let imm = mem_offset(memarg, module);
            let base_addr = inputs[0].start as usize;
            let output = (output.unwrap().start) as usize;

            match memarg.align {
                0 => {
                    // read byte by byte
                    let b0 = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let b1 = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let b2 = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let b3 = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let b4 = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let b5 = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let b6 = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let b7 = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let b1_shifted =
                        c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let b2_shifted =
                        c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let b3_shifted =
                        c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let b5_shifted =
                        c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let b6_shifted =
                        c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let b7_shifted =
                        c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;

                    let hi0 = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let hi1 = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let lo0 = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let lo1 = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;

                    vec![
                        // load each byte
                        Directive::Instruction(ib::loadbu(b0, base_addr, imm)),
                        Directive::Instruction(ib::loadbu(b1, base_addr, imm + 1)),
                        Directive::Instruction(ib::loadbu(b2, base_addr, imm + 2)),
                        Directive::Instruction(ib::loadbu(b3, base_addr, imm + 3)),
                        Directive::Instruction(ib::loadbu(b4, base_addr, imm + 4)),
                        Directive::Instruction(ib::loadbu(b5, base_addr, imm + 5)),
                        Directive::Instruction(ib::loadbu(b6, base_addr, imm + 6)),
                        Directive::Instruction(ib::loadbu(b7, base_addr, imm + 7)),
                        // build lo 32 bits
                        Directive::Instruction(ib::shl_imm(b1_shifted, b1, 8_i16)),
                        Directive::Instruction(ib::shl_imm(b2_shifted, b2, 16_i16)),
                        Directive::Instruction(ib::shl_imm(b3_shifted, b3, 24_i16)),
                        Directive::Instruction(ib::or(lo0, b0, b1_shifted)),
                        Directive::Instruction(ib::or(lo1, b2_shifted, b3_shifted)),
                        Directive::Instruction(ib::or(output, lo0, lo1)),
                        // build hi 32 bits
                        Directive::Instruction(ib::shl_imm(b5_shifted, b5, 8_i16)),
                        Directive::Instruction(ib::shl_imm(b6_shifted, b6, 16_i16)),
                        Directive::Instruction(ib::shl_imm(b7_shifted, b7, 24_i16)),
                        Directive::Instruction(ib::or(hi0, b4, b5_shifted)),
                        Directive::Instruction(ib::or(hi1, b6_shifted, b7_shifted)),
                        Directive::Instruction(ib::or(output + 1, hi0, hi1)),
                    ]
                }
                1 => {
                    // read four halfwords
                    let h0 = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let h1 = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let h2 = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let h3 = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let h1_shifted =
                        c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let h3_shifted =
                        c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;

                    vec![
                        Directive::Instruction(ib::loadhu(h0, base_addr, imm)),
                        Directive::Instruction(ib::loadhu(h1, base_addr, imm + 2)),
                        Directive::Instruction(ib::loadhu(h2, base_addr, imm + 4)),
                        Directive::Instruction(ib::loadhu(h3, base_addr, imm + 6)),
                        Directive::Instruction(ib::shl_imm(h1_shifted, h1, 16_i16)),
                        Directive::Instruction(ib::shl_imm(h3_shifted, h3, 16_i16)),
                        Directive::Instruction(ib::or(output, h0, h1_shifted)),
                        Directive::Instruction(ib::or(output + 1, h2, h3_shifted)),
                    ]
                }
                2.. => {
                    // read two words
                    // Load hi word first: if output == base_addr, loading lo first would
                    // clobber the address before the second load reads it.
                    vec![
                        Directive::Instruction(ib::loadw(output + 1, base_addr, imm + 4)),
                        Directive::Instruction(ib::loadw(output, base_addr, imm)),
                    ]
                }
            }
            .into()
        }
        Op::I64Load8U { memarg } | Op::I64Load8S { memarg } => {
            let imm = mem_offset(memarg, module);
            let base_addr = inputs[0].start as usize;
            let output = (output.unwrap().start) as usize;
            let val = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I64).start as usize;

            vec![
                // load signed or unsigned byte as i32 hi part
                if let Op::I64Load8S { .. } = op {
                    Directive::Instruction(ib::loadb(val + 1, base_addr, imm))
                } else {
                    Directive::Instruction(ib::loadbu(val + 1, base_addr, imm))
                },
                // shr will read 64 bits, so we need to zero the other half due to WOM
                Directive::Instruction(ib::const_32_imm(val, 0, 0)),
                // shift i64 val right, keeping the sign
                Directive::Instruction(ib::shr_s_imm_64(output, val, 32_i16)),
            ]
            .into()
        }
        Op::I64Load16U { memarg } | Op::I64Load16S { memarg } => {
            let imm = mem_offset(memarg, module);
            let base_addr = inputs[0].start as usize;
            let output = (output.unwrap().start) as usize;
            let val = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I64).start as usize;

            match memarg.align {
                0 => {
                    let b0 = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let b1 = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let b1_shifted =
                        c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    vec![
                        // load b1 signed/unsigned
                        if let Op::I64Load16S { .. } = op {
                            Directive::Instruction(ib::loadb(b1, base_addr, imm + 1))
                        } else {
                            Directive::Instruction(ib::loadbu(b1, base_addr, imm + 1))
                        },
                        // shift b1
                        Directive::Instruction(ib::shl_imm(b1_shifted, b1, 8_i16)),
                        // load b0
                        Directive::Instruction(ib::loadbu(b0, base_addr, imm)),
                        // combine b0 and b1
                        Directive::Instruction(ib::or(val + 1, b0, b1_shifted)),
                        // shr will read 64 bits, so we need to zero the other half due to WOM
                        Directive::Instruction(ib::const_32_imm(val, 0, 0)),
                        // shift i64 val right, keeping the sign
                        Directive::Instruction(ib::shr_s_imm_64(output, val, 32_i16)),
                    ]
                }
                1.. => {
                    if let Op::I64Load16S { .. } = op {
                        vec![
                            // load signed halfword as i32 on the hi part of the i64 val
                            Directive::Instruction(ib::loadh(val + 1, base_addr, imm)),
                            // shr will read 64 bits, so we need to zero the other half due to WOM
                            Directive::Instruction(ib::const_32_imm(val, 0, 0)),
                            // shift i64 val right, keeping the sign
                            Directive::Instruction(ib::shr_s_imm_64(output, val, 32_i16)),
                        ]
                    } else {
                        vec![
                            // load unsigned lo i32
                            Directive::Instruction(ib::loadhu(output, base_addr, imm)),
                            // zero out hi i32
                            Directive::Instruction(ib::const_32_imm(output + 1, 0x0, 0x0)),
                        ]
                    }
                }
            }
            .into()
        }
        Op::I64Load32U { memarg } | Op::I64Load32S { memarg } => {
            let imm = mem_offset(memarg, module);
            let base_addr = inputs[0].start as usize;
            let output = (output.unwrap().start) as usize;

            // if signed, we need to place the loaded 32-bit value in a new register
            // bit extend it with shr_s_imm_64, otherwise we can load directly into lo part
            let (val_32, zeroed) = if let Op::I64Load32S { .. } = op {
                let val = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I64).start as usize;
                // Load the value into the hi word, and zero the lo word for the shr
                (val + 1, val)
            } else {
                // Load directly into lo part, zero the hi part
                (output, output + 1)
            };

            let mut directives = vec![
                // zero the other half due to WOM
                Directive::Instruction(ib::const_32_imm(zeroed, 0, 0)),
            ];

            match memarg.align {
                0 => {
                    let b0 = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let b1 = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let b2 = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let b3 = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let b1_shifted =
                        c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let b2_shifted =
                        c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let b3_shifted =
                        c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let hi = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let lo = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    directives.extend([
                        Directive::Instruction(ib::loadbu(b0, base_addr, imm)),
                        Directive::Instruction(ib::loadbu(b1, base_addr, imm + 1)),
                        Directive::Instruction(ib::loadbu(b2, base_addr, imm + 2)),
                        Directive::Instruction(ib::loadbu(b3, base_addr, imm + 3)),
                        // shifts
                        Directive::Instruction(ib::shl_imm(b1_shifted, b1, 8_i16)),
                        Directive::Instruction(ib::shl_imm(b2_shifted, b2, 16_i16)),
                        Directive::Instruction(ib::shl_imm(b3_shifted, b3, 24_i16)),
                        // build hi and lo
                        Directive::Instruction(ib::or(lo, b0, b1_shifted)),
                        Directive::Instruction(ib::or(hi, b2_shifted, b3_shifted)),
                        // build hi i32 in val
                        Directive::Instruction(ib::or(val_32, lo, hi)),
                    ]);
                }
                1 => {
                    let h0 = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let h1 = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let h1_shifted =
                        c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    directives.extend([
                        // load h0, h1
                        Directive::Instruction(ib::loadhu(h0, base_addr, imm)),
                        Directive::Instruction(ib::loadhu(h1, base_addr, imm + 2)),
                        // shift h1
                        Directive::Instruction(ib::shl_imm(h1_shifted, h1, 16_i16)),
                        // combine h0 and h1
                        Directive::Instruction(ib::or(val_32, h0, h1_shifted)),
                    ]);
                }
                2.. => directives.push(Directive::Instruction(ib::loadw(val_32, base_addr, imm))),
            }

            // In case of signed, we need to right shift the loaded value into the output
            if let Op::I64Load32S { .. } = op {
                let d = Directive::Instruction(ib::shr_s_imm_64(output, zeroed, 32_i16));
                directives.push(d);
            }

            directives.into()
        }
        Op::I32Store { memarg } | Op::I64Store32 { memarg } => {
            let imm = mem_offset(memarg, module);
            let base_addr = inputs[0].start as usize;
            let value = inputs[1].start as usize;

            match memarg.align {
                0 => {
                    // write byte by byte
                    let b1 = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let b2 = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let b3 = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    vec![
                        // store byte 0
                        Directive::Instruction(ib::storeb(value, base_addr, imm)),
                        // shift and store byte 1
                        Directive::Instruction(ib::shr_u_imm(b1, value, 8_i16)),
                        Directive::Instruction(ib::storeb(b1, base_addr, imm + 1)),
                        // shift and store byte 2
                        Directive::Instruction(ib::shr_u_imm(b2, value, 16_i16)),
                        Directive::Instruction(ib::storeb(b2, base_addr, imm + 2)),
                        // shift and store byte 3
                        Directive::Instruction(ib::shr_u_imm(b3, value, 24_i16)),
                        Directive::Instruction(ib::storeb(b3, base_addr, imm + 3)),
                    ]
                    .into()
                }
                1 => {
                    let h1 = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    vec![
                        // store halfword 0
                        Directive::Instruction(ib::storeh(value, base_addr, imm)),
                        // shift and store halfword 1
                        Directive::Instruction(ib::shr_u_imm(h1, value, 16_i16)),
                        Directive::Instruction(ib::storeh(h1, base_addr, imm + 2)),
                    ]
                    .into()
                }
                2.. => Directive::Instruction(ib::storew(value, base_addr, imm)).into(),
            }
        }
        Op::I32Store8 { memarg } | Op::I64Store8 { memarg } => {
            let imm = mem_offset(memarg, module);
            let base_addr = inputs[0].start as usize;
            let value = inputs[1].start as usize;

            Directive::Instruction(ib::storeb(value, base_addr, imm)).into()
        }
        Op::I32Store16 { memarg } | Op::I64Store16 { memarg } => {
            let imm = mem_offset(memarg, module);
            let base_addr = inputs[0].start as usize;
            let value = inputs[1].start as usize;

            match memarg.align {
                0 => {
                    let b1 = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    vec![
                        // store byte 0
                        Directive::Instruction(ib::storeb(value, base_addr, imm)),
                        // shift and store byte 1
                        Directive::Instruction(ib::shr_u_imm(b1, value, 8_i16)),
                        Directive::Instruction(ib::storeb(b1, base_addr, imm + 1)),
                    ]
                    .into()
                }
                1.. => Directive::Instruction(ib::storeh(value, base_addr, imm)).into(),
            }
        }
        Op::I64Store { memarg } => {
            let imm = mem_offset(memarg, module);
            let base_addr = inputs[0].start as usize;
            let value_lo = inputs[1].start as usize;
            let value_hi = (inputs[1].start + 1) as usize;

            match memarg.align {
                0 => {
                    // write byte by byte
                    let b1 = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let b2 = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let b3 = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let b5 = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let b6 = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let b7 = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    vec![
                        // store byte 0
                        Directive::Instruction(ib::storeb(value_lo, base_addr, imm)),
                        // shift and store byte 1
                        Directive::Instruction(ib::shr_u_imm(b1, value_lo, 8_i16)),
                        Directive::Instruction(ib::storeb(b1, base_addr, imm + 1)),
                        // shift and store byte 2
                        Directive::Instruction(ib::shr_u_imm(b2, value_lo, 16_i16)),
                        Directive::Instruction(ib::storeb(b2, base_addr, imm + 2)),
                        // shift and store byte 3
                        Directive::Instruction(ib::shr_u_imm(b3, value_lo, 24_i16)),
                        Directive::Instruction(ib::storeb(b3, base_addr, imm + 3)),
                        // store byte 4
                        Directive::Instruction(ib::storeb(value_hi, base_addr, imm + 4)),
                        // shift and store byte 5
                        Directive::Instruction(ib::shr_u_imm(b5, value_hi, 8_i16)),
                        Directive::Instruction(ib::storeb(b5, base_addr, imm + 5)),
                        // shift and store byte 6
                        Directive::Instruction(ib::shr_u_imm(b6, value_hi, 16_i16)),
                        Directive::Instruction(ib::storeb(b6, base_addr, imm + 6)),
                        // shift and store byte 7
                        Directive::Instruction(ib::shr_u_imm(b7, value_hi, 24_i16)),
                        Directive::Instruction(ib::storeb(b7, base_addr, imm + 7)),
                    ]
                }
                1 => {
                    // write by halfwords
                    let h1 = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    let h3 = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
                    vec![
                        // store halfword 0
                        Directive::Instruction(ib::storeh(value_lo, base_addr, imm)),
                        // shift and store halfword 1
                        Directive::Instruction(ib::shr_u_imm(h1, value_lo, 16_i16)),
                        Directive::Instruction(ib::storeh(h1, base_addr, imm + 2)),
                        // store halfword 2
                        Directive::Instruction(ib::storeh(value_hi, base_addr, imm + 4)),
                        // shift and store halfword 3
                        Directive::Instruction(ib::shr_u_imm(h3, value_hi, 16_i16)),
                        Directive::Instruction(ib::storeh(h3, base_addr, imm + 6)),
                    ]
                }
                2.. => {
                    vec![
                        Directive::Instruction(ib::storew(value_lo, base_addr, imm)),
                        Directive::Instruction(ib::storew(value_hi, base_addr, imm + 4)),
                    ]
                }
            }
            .into()
        }

        Op::MemorySize { mem } => {
            assert_eq!(mem, 0, "Only a single linear memory is supported");
            load_from_const_addr(c, module.memory.unwrap().start, output.unwrap())
                .0
                .into()
        }
        Op::MemoryGrow { mem } => {
            assert_eq!(mem, 0, "Only a single linear memory is supported");
            let header_addr = module.memory.unwrap().start;

            let output = output.unwrap().start as usize;

            let header_regs = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I64);
            let size_reg = header_regs.start as usize;
            let max_size_reg = (header_regs.start + 1) as usize;

            let new_size = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;
            let is_gt_max = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start;
            let is_lt_curr = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start;

            let error_label = c.new_label(LabelType::Local);
            let continuation_label = c.new_label(LabelType::Local);

            // Load the current size and max size.
            let (mut directives, header_addr_reg) =
                load_from_const_addr(c, header_addr, header_regs);

            directives.extend([
                // Calculate the new size:
                Directive::Instruction(ib::add(new_size, size_reg, inputs[0].start as usize)),
                // Check if the new size is greater than the max size.
                Directive::Instruction(ib::gt_u(is_gt_max as usize, new_size, max_size_reg)),
                // If the new size is greater than the max size, branch to the error label.
                Directive::JumpIf {
                    target: error_label.clone(),
                    condition_reg: is_gt_max,
                },
                // Check if the new size is less than to the current size (which means an overflow occurred),
                // which means the requested size is too large.
                Directive::Instruction(ib::lt_u(is_lt_curr as usize, new_size, size_reg)),
                // If the requested size overflows, branch to the error label.
                Directive::JumpIf {
                    target: error_label.clone(),
                    condition_reg: is_lt_curr,
                },
                // Success case:
                // - write new size to header.
                Directive::Instruction(ib::storew(new_size, header_addr_reg.start as usize, 0)),
                // - write old size to output.
                Directive::Instruction(ib::add_imm(output, size_reg, AluImm::from(0))),
                // - jump to continuation label.
                Directive::Jump {
                    target: continuation_label.clone(),
                },
                // Error case: write 0xFFFFFFFF to output.
                Directive::Label {
                    id: error_label,
                    frame_size: None,
                },
                Directive::Instruction(ib::const_32_imm(output, 0xFFFF, 0xFFFF)),
                // Continue:
                Directive::Label {
                    id: continuation_label,
                    frame_size: None,
                },
            ]);

            directives.into()
        }
        Op::MemoryInit {
            data_index: _,
            mem: _,
        } => todo!(),
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
        Op::TableGet { table } => {
            emit_table_get(c, module, table, inputs[0].clone(), output.unwrap()).into()
        }
        Op::TableSet { table: _ } => todo!(),
        Op::TableGrow { table: _ } => todo!(),
        Op::TableSize { table: _ } => todo!(),
        Op::ElemDrop { elem_index: _ } => todo!(),

        // Reference instructions
        Op::RefNull { hty: _ } => NULL_REF
            .iter()
            .zip_eq(output.unwrap())
            .map(|(v, reg)| {
                Directive::Instruction(ib::const_32_imm(reg as usize, *v as u16, (*v >> 16) as u16))
            })
            .collect_vec()
            .into(),
        Op::RefIsNull => todo!(),
        Op::RefFunc { function_index } => {
            const_function_reference(module, function_index, output.unwrap()).into()
        }

        // Float instructions
        Op::F32Const { value } => {
            let output = output.unwrap().start as usize;
            let value_u = value.bits();
            let value_lo: u16 = (value_u & 0xffff) as u16;
            let value_hi: u16 = ((value_u >> 16) & 0xffff) as u16;
            Directive::Instruction(ib::const_32_imm(output, value_lo, value_hi)).into()
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
            .into()
        }

        Op::F32Load { .. }
        | Op::F64Load { .. }
        | Op::F32Store { .. }
        | Op::F64Store { .. }
        | Op::F32Abs
        | Op::F32Neg
        | Op::F32Ceil
        | Op::F32Floor
        | Op::F32Trunc
        | Op::F32Nearest
        | Op::F32Sqrt
        | Op::F64Abs
        | Op::F64Neg
        | Op::F64Ceil
        | Op::F64Floor
        | Op::F64Trunc
        | Op::F64Nearest
        | Op::F64Sqrt
        | Op::I32TruncF32S
        | Op::I32TruncF32U
        | Op::I32TruncF64S
        | Op::I32TruncF64U
        | Op::I64TruncF32S
        | Op::I64TruncF32U
        | Op::I64TruncF64S
        | Op::I64TruncF64U
        | Op::F32ConvertI32S
        | Op::F32ConvertI32U
        | Op::F32ConvertI64S
        | Op::F32ConvertI64U
        | Op::F32DemoteF64
        | Op::F64ConvertI32S
        | Op::F64ConvertI32U
        | Op::F64ConvertI64S
        | Op::F64ConvertI64U
        | Op::F64PromoteF32 => {
            Directive::Instruction(ib::trap(ERROR_UNIMPLEMENTED_CODE as usize)).into()
        }

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
                    Directive::Instruction(ib::add_imm(
                        output as usize,
                        input as usize,
                        AluImm::from(0),
                    ))
                })
                .collect_vec()
                .into()
        }

        // Instructions that are implemented as function calls
        Op::MemoryCopy { .. }
        | Op::MemoryFill { .. }
        | Op::I32Popcnt
        | Op::I64Popcnt
        | Op::I32Ctz
        | Op::I64Ctz
        | Op::I32Clz
        | Op::I64Clz => {
            unreachable!("These ops should have been replaced with function calls")
        }
        _ => todo!("{op:?}"),
    }
}

fn emit_table_get<'a, F: PrimeField32>(
    c: &mut Ctx<'a, '_>,
    module: &Module<'a>,
    table_idx: u32,
    entry_idx_ptr: Range<u32>,
    dest_ptr: Range<u32>,
) -> Vec<Directive<F>> {
    const TABLE_ENTRY_SIZE: i16 = 12;
    const TABLE_SEGMENT_HEADER_SIZE: u32 = 8;

    let table_segment = module.tables[table_idx as usize];

    let mul_result = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32).start as usize;

    let base_addr = table_segment.start + TABLE_SEGMENT_HEADER_SIZE;

    // Read the 3 words of the reference into contiguous registers
    assert_eq!(dest_ptr.len(), 3);

    let mut instrs = vec![Directive::Instruction(ib::mul_imm(
        mul_result,
        entry_idx_ptr.start as usize,
        TABLE_ENTRY_SIZE,
    ))];

    instrs.extend(dest_ptr.enumerate().map(|(i, dest_reg)| {
        Directive::Instruction(ib::loadw(
            dest_reg as usize,
            mul_result,
            base_addr + (i as u32) * 4,
        ))
    }));

    instrs
}

enum RotDirection {
    Left,
    Right,
}

trait RotOps<F: PrimeField32> {
    fn val_type() -> ValType;
    fn num_bits() -> i16;
    fn mask_rot_bits(dest: usize, src: usize) -> Instruction<F>;
    fn num_bits_const(dest: usize) -> Vec<Directive<F>>;
    fn shl(dest: usize, val: usize, amount: usize) -> Instruction<F>;
    fn shr_u(dest: usize, val: usize, amount: usize) -> Instruction<F>;
    fn shl_imm(dest: usize, val: usize, amount: impl Into<AluImm>) -> Instruction<F>;
    fn shr_u_imm(dest: usize, val: usize, amount: impl Into<AluImm>) -> Instruction<F>;
    fn sub(dest: usize, lhs: usize, rhs: usize) -> Instruction<F>;
    fn or(dest: usize, lhs: usize, rhs: usize) -> Instruction<F>;
}

struct I32Rot;

impl<F: PrimeField32> RotOps<F> for I32Rot {
    fn val_type() -> ValType {
        ValType::I32
    }
    fn num_bits() -> i16 {
        32
    }
    fn mask_rot_bits(dest: usize, src: usize) -> Instruction<F> {
        ib::and_imm(dest, src, 32_i16 - 1)
    }

    fn num_bits_const(dest: usize) -> Vec<Directive<F>> {
        vec![Directive::Instruction(ib::const_32_imm(dest, 32, 0))]
    }
    fn shl(dest: usize, val: usize, amount: usize) -> Instruction<F> {
        ib::shl(dest, val, amount)
    }
    fn shr_u(dest: usize, val: usize, amount: usize) -> Instruction<F> {
        ib::shr_u(dest, val, amount)
    }
    fn shl_imm(dest: usize, val: usize, amount: impl Into<AluImm>) -> Instruction<F> {
        ib::shl_imm(dest, val, amount)
    }
    fn shr_u_imm(dest: usize, val: usize, amount: impl Into<AluImm>) -> Instruction<F> {
        ib::shr_u_imm(dest, val, amount)
    }
    fn sub(dest: usize, lhs: usize, rhs: usize) -> Instruction<F> {
        ib::sub(dest, lhs, rhs)
    }
    fn or(dest: usize, lhs: usize, rhs: usize) -> Instruction<F> {
        ib::or(dest, lhs, rhs)
    }
}

struct I64Rot;

impl<F: PrimeField32> RotOps<F> for I64Rot {
    fn val_type() -> ValType {
        ValType::I64
    }
    fn num_bits() -> i16 {
        64
    }
    fn mask_rot_bits(dest: usize, src: usize) -> Instruction<F> {
        ib::and_imm_64(dest, src, 64_i16 - 1)
    }
    fn num_bits_const(dest: usize) -> Vec<Directive<F>> {
        vec![
            Directive::Instruction(ib::const_32_imm(dest, 64, 0)),
            Directive::Instruction(ib::const_32_imm(dest + 1, 0, 0)),
        ]
    }
    fn shl(dest: usize, val: usize, amount: usize) -> Instruction<F> {
        ib::shl_64(dest, val, amount)
    }
    fn shr_u(dest: usize, val: usize, amount: usize) -> Instruction<F> {
        ib::shr_u_64(dest, val, amount)
    }
    fn shl_imm(dest: usize, val: usize, amount: impl Into<AluImm>) -> Instruction<F> {
        ib::shl_imm_64(dest, val, amount)
    }
    fn shr_u_imm(dest: usize, val: usize, amount: impl Into<AluImm>) -> Instruction<F> {
        ib::shr_u_imm_64(dest, val, amount)
    }
    fn sub(dest: usize, lhs: usize, rhs: usize) -> Instruction<F> {
        ib::sub_64(dest, lhs, rhs)
    }
    fn or(dest: usize, lhs: usize, rhs: usize) -> Instruction<F> {
        ib::or_64(dest, lhs, rhs)
    }
}

/// Issue the instructions to perform a rotate operation.
fn translate_rot<'a, F: PrimeField32, R: RotOps<F>>(
    c: &mut Ctx<'a, '_>,
    direction: RotDirection,
    inputs: &[WasmOpInput],
    output: &Range<u32>,
) -> Tree<Directive<F>> {
    // Const collapse ensures input[0] is always a register.
    let value = inputs[0].as_register().unwrap().start as usize;
    let output = output.start as usize;

    let shift_ref = c
        .allocate_tmp_type::<OpenVMSettings<F>>(R::val_type())
        .start as usize;
    let shift_back = c
        .allocate_tmp_type::<OpenVMSettings<F>>(R::val_type())
        .start as usize;

    let mut directives = match &inputs[1] {
        WasmOpInput::Register(reg) => {
            let reg = reg.start as usize;
            let shift_ref_amount = c
                .allocate_tmp_type::<OpenVMSettings<F>>(R::val_type())
                .start as usize;
            // TODO: if the transformation from rot to this sequence of instructions
            // was done earlier in the pipeline, at DAG level, this const could be
            // optimized away, sometimes.
            let const_num_bits = c
                .allocate_tmp_type::<OpenVMSettings<F>>(R::val_type())
                .start as usize;
            let shift_back_amount = c
                .allocate_tmp_type::<OpenVMSettings<F>>(R::val_type())
                .start as usize;

            let mut directives = R::num_bits_const(const_num_bits);

            directives.extend(
                [
                    // mask the shift amount to the valid range
                    R::mask_rot_bits(shift_ref_amount, reg),
                    // calculate the shift amount for the opposite direction
                    R::sub(shift_back_amount, const_num_bits, shift_ref_amount),
                ]
                .map(Directive::Instruction),
            );

            directives.extend(
                match direction {
                    RotDirection::Left => [
                        // shift left
                        R::shl(shift_ref, value, shift_ref_amount),
                        // shift right
                        R::shr_u(shift_back, value, shift_back_amount),
                    ],
                    RotDirection::Right => [
                        // shift right
                        R::shr_u(shift_ref, value, shift_ref_amount),
                        // shift left
                        R::shl(shift_back, value, shift_back_amount),
                    ],
                }
                .map(Directive::Instruction),
            );

            directives
        }
        WasmOpInput::Constant(wasm_value) => {
            let shift_bits_ref = match *wasm_value {
                WasmValue::I32(v) => v as i16,
                WasmValue::I64(v) => v as i16,
                _ => panic!("not valid i16"),
            } & (R::num_bits() - 1);
            let shift_bits_back = R::num_bits() - shift_bits_ref;

            match direction {
                RotDirection::Left => vec![
                    // shift left
                    Directive::Instruction(R::shl_imm(shift_ref, value, shift_bits_ref)),
                    // shift right
                    Directive::Instruction(R::shr_u_imm(shift_back, value, shift_bits_back)),
                ],
                RotDirection::Right => vec![
                    // shift right
                    Directive::Instruction(R::shr_u_imm(shift_ref, value, shift_bits_ref)),
                    // shift left
                    Directive::Instruction(R::shl_imm(shift_back, value, shift_bits_back)),
                ],
            }
        }
    };

    // or the two shifts
    directives.push(Directive::Instruction(R::or(output, shift_ref, shift_back)));

    directives.into()
}

/// Precondition: `value` is either `WasmValue::I32` or `WasmValue::I64`
fn const_i16_as_field(value: &WasmValue) -> AluImm {
    let c: i16 = match *value {
        WasmValue::I32(v) => v as i16,
        WasmValue::I64(v) => v as i16,
        _ => panic!("not valid i16"),
    };
    c.into()
}

fn mem_offset(memarg: MemArg, module: &Module) -> u32 {
    assert_eq!(memarg.memory, 0, "no multiple memories supported");
    let mem_start = module.linear_memory_start().expect("no memory allocated");
    mem_start.wrapping_add(u32::try_from(memarg.offset).expect("offset too large"))
}

fn load_from_const_addr<'a, F: PrimeField32>(
    c: &mut Ctx<'a, '_>,
    base_addr: u32,
    output: Range<u32>,
) -> (Vec<Directive<F>>, Range<u32>) {
    let base_addr_reg = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32);
    let mut directives = vec![Directive::Instruction(ib::const_32_imm(
        base_addr_reg.start as usize,
        base_addr as u16,
        (base_addr >> 16) as u16,
    ))];

    directives.extend(output.enumerate().map(|(i, dest_reg)| {
        Directive::Instruction(ib::loadw(
            dest_reg as usize,
            base_addr_reg.start as usize,
            (i as u32) * 4,
        ))
    }));

    (directives, base_addr_reg)
}

fn store_to_const_addr<'a, F: PrimeField32>(
    c: &mut Ctx<'a, '_>,
    base_addr: u32,
    input: Range<u32>,
) -> Vec<Directive<F>> {
    let base_addr_reg = c.allocate_tmp_type::<OpenVMSettings<F>>(ValType::I32);
    let mut directives = vec![Directive::Instruction(ib::const_32_imm(
        base_addr_reg.start as usize,
        base_addr as u16,
        (base_addr >> 16) as u16,
    ))];

    directives.extend(input.enumerate().map(|(i, input_reg)| {
        Directive::Instruction(ib::storew(
            input_reg as usize,
            base_addr_reg.start as usize,
            i as u32 * 4,
        ))
    }));

    directives
}

impl<F: Clone> womir::interpreter::linker::Directive for Directive<F> {
    fn nop() -> Directive<F> {
        Directive::Nop
    }

    fn as_label(&self) -> Option<womir::interpreter::linker::Label<'_>> {
        if let Directive::Label { id, frame_size } = self {
            Some(womir::interpreter::linker::Label {
                id,
                frame_size: *frame_size,
            })
        } else {
            None
        }
    }
}

fn const_function_reference<F: PrimeField32>(
    module: &Module,
    func_idx: u32,
    reg_dest: Range<u32>,
) -> Vec<Directive<F>> {
    assert_eq!(reg_dest.len(), 3);
    let reg_dest = reg_dest.start;

    let ty = module.get_func_type(func_idx);
    vec![
        // Unique function type identifier
        Directive::Instruction(ib::const_32_imm(
            reg_dest as usize,
            ty.unique_id as u16,
            (ty.unique_id >> 16) as u16,
        )),
        // Function address
        Directive::ConstFuncAddr {
            func_idx,
            reg_dest: reg_dest + 1,
        },
        // Function frame size (unused in RWM, set to 0)
        Directive::Instruction(ib::const_32_imm((reg_dest + 2) as usize, 0, 0)),
    ]
}

fn const_wasm_value<F: PrimeField32>(
    module: &Module,
    value: &WasmValue,
    reg_dest: Range<u32>,
) -> Vec<Directive<F>> {
    let mut words = ArrayVec::<_, 4>::new();
    match value {
        WasmValue::I32(v) => words.push(*v as u32),
        WasmValue::I64(v) => words.extend([*v as u32, (*v >> 32) as u32]),
        WasmValue::F64(v) => words.extend([v.bits() as u32, (v.bits() >> 32) as u32]),
        WasmValue::F32(ieee32) => words.push(ieee32.bits()),
        WasmValue::V128(v128) => words.extend(
            v128.bytes()
                .chunks(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap())),
        ),
        WasmValue::RefNull => words.extend(NULL_REF),
        WasmValue::RefFunc(func_ref) => {
            return const_function_reference(module, *func_ref, reg_dest);
        }
    };
    words
        .into_iter()
        .zip_eq(reg_dest)
        .map(|(word, reg)| {
            Directive::Instruction(ib::const_32_imm(
                reg as usize,
                word as u16,
                (word >> 16) as u16,
            ))
        })
        .collect()
}
