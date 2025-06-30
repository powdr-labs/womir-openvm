use std::{fmt::Display, ops::Range};

use openvm_instructions::instruction::Instruction;
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use wasmparser::Operator as WasmOp;
use womir::loader::flattening::{
    settings::{ComparisonFunction, JumpCondition, LoopFrameLayout, ReturnInfosToCopy, Settings},
    Generators, RegisterGenerator, TrapReason, Tree,
};

struct WomirSettings;

#[derive(Debug, Clone)]
enum Directive {
    Ins(Instruction<BabyBear>),
    Label(String),
}

impl Display for Directive {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Directive::Ins(ins) => write!(
                f,
                "opcode: {} [{}, {}, {}, {}, {}, {}]",
                ins.opcode, ins.a, ins.b, ins.c, ins.d, ins.e, ins.f
            ),
            Directive::Label(label) => write!(f, "label: {}", label),
        }
    }
}

#[allow(refining_impl_trait)]
impl<'a> Settings<'a> for WomirSettings {
    type Directive = Directive;

    fn bytes_per_word() -> u32 {
        4
    }

    fn words_per_ptr() -> u32 {
        1
    }

    fn is_jump_condition_available(cond: JumpCondition) -> bool {
        // Both would be available in RISC-V if:
        // - x0 was available, which it is not in WOM;
        //   - (can be simulated by assigning 0 to some register and using it)
        // - beq and bne were absolute jumps, but in RISC-V they are relative relative to PC.
        //   - (can be simulated with a block of code with beq/bne+auipc+jalr)
        match cond {
            JumpCondition::IfZero => true,
            JumpCondition::IfNotZero => true,
        }
    }

    fn is_relative_jump_available() -> bool {
        // Pretty much all RISC-V jumps are relative, except for jalr.
        true
    }

    fn allocate_loop_frame_slots(
        &self,
        need_ret_info: bool,
        saved_fps: std::collections::BTreeSet<u32>,
    ) -> (RegisterGenerator<'a, Self>, LoopFrameLayout) {
        // No tricks here, can use the same allocation from Womir generic IR.
        todo!()
    }

    fn to_plain_local_jump(directive: Self::Directive) -> Result<String, Directive> {
        // We actually don't have plain local jumps in RISC-V, see `emit_jump()`
        todo!()
    }

    fn emit_label(
        &self,
        g: &mut Generators<'a, '_, Self>,
        name: String,
        frame_size: Option<u32>,
    ) -> Directive {
        Directive::Label(name)
    }

    fn emit_trap(
        &self,
        g: &mut Generators<'a, '_, Self>,
        trap: TrapReason,
    ) -> impl Into<Tree<Directive>> {
        // We can use unimp instruction, but we won't be able to encode the trap reason.
        todo!()
    }

    fn emit_allocate_label_frame(
        &self,
        g: &mut Generators<'a, '_, Self>,
        label: String,
        result_ptr: Range<u32>,
    ) -> impl Into<Tree<Directive>> {
        // Needs new instruction to allocate a frame from immediate size. No such concept exists in RISC-V.
        todo!()
    }

    fn emit_allocate_value_frame(
        &self,
        g: &mut Generators<'a, '_, Self>,
        frame_size_ptr: Range<u32>,
        result_ptr: Range<u32>,
    ) -> impl Into<Tree<Directive>> {
        // Needs new instruction to allocate a frame from size in register. No such concept exists in RISC-V.
        todo!()
    }

    fn emit_copy(
        &self,
        g: &mut Generators<'a, '_, Self>,
        src_ptr: Range<u32>,
        dest_ptr: Range<u32>,
    ) -> impl Into<Tree<Directive>> {
        // Would be easy if we had x0, we could use either addi or ori.
        // Without it, we first define a 0 register, then use either addi or ori to copy the value.
        todo!()
    }

    fn emit_copy_into_frame(
        &self,
        g: &mut Generators<'a, '_, Self>,
        src_ptr: Range<u32>,
        dest_frame_ptr: Range<u32>,
        dest_offset: Range<u32>,
    ) -> impl Into<Tree<Directive>> {
        // New instruction required.
        todo!()
    }

    fn emit_jump(&self, label: String) -> Directive {
        // This is supposed to be a plain absolute jump, but they don't exist in RISC-V.
        // With a lot of effort, they can be simulated with auipc+jalr.
        todo!()
    }

    fn emit_jump_into_loop(
        &self,
        g: &mut Generators<'a, '_, Self>,
        loop_label: String,
        loop_frame_ptr: Range<u32>,
        ret_info_to_copy: Option<ReturnInfosToCopy>,
        saved_curr_fp_ptr: Option<Range<u32>>,
    ) -> impl Into<Tree<Directive>> {
        // New instruction required.
        todo!()
    }

    fn emit_conditional_jump(
        &self,
        g: &mut Generators<'a, '_, Self>,
        condition_type: JumpCondition,
        label: String,
        condition_ptr: Range<u32>,
    ) -> impl Into<Tree<Directive>> {
        // Problematic, see `is_jump_condition_available()`.
        todo!()
    }

    fn emit_conditional_jump_cmp_immediate(
        &self,
        g: &mut Generators<'a, '_, Self>,
        cmp: ComparisonFunction,
        value_ptr: Range<u32>,
        immediate: u32,
        label: String,
    ) -> impl Into<Tree<Directive>> {
        // Hard on account of RISC-V jumps being relative to PC.
        todo!()
    }

    fn emit_relative_jump(
        &self,
        g: &mut Generators<'a, '_, Self>,
        offset_ptr: Range<u32>,
    ) -> impl Into<Tree<Directive>> {
        // Just a RISC-V `j`.
        todo!()
    }

    fn emit_jump_out_of_loop(
        &self,
        g: &mut Generators<'a, '_, Self>,
        target_label: String,
        target_frame_ptr: Range<u32>,
    ) -> impl Into<Tree<Directive>> {
        // New instruction required.
        todo!()
    }

    fn emit_return(
        &self,
        g: &mut Generators<'a, '_, Self>,
        ret_pc_ptr: Range<u32>,
        caller_fp_ptr: Range<u32>,
    ) -> impl Into<Tree<Directive>> {
        // New instruction required.
        todo!()
    }

    fn emit_imported_call(
        &self,
        g: &mut Generators<'a, '_, Self>,
        module: &'a str,
        function: &'a str,
        inputs: Vec<Range<u32>>,
        outputs: Vec<Range<u32>>,
    ) -> impl Into<Tree<Directive>> {
        // This is a system call. Issues one of OpenVM custom instructions.
        todo!()
    }

    fn emit_function_call(
        &self,
        g: &mut Generators<'a, '_, Self>,
        function_label: String,
        function_frame_ptr: Range<u32>,
        saved_ret_pc_ptr: Range<u32>,
        saved_caller_fp_ptr: Range<u32>,
    ) -> impl Into<Tree<Directive>> {
        // Requires new instruction that is both absolute and handles the frame pointer.
        todo!()
    }

    fn emit_indirect_call(
        &self,
        g: &mut Generators<'a, '_, Self>,
        target_pc_ptr: Range<u32>,
        function_frame_ptr: Range<u32>,
        saved_ret_pc_ptr: Range<u32>,
        saved_caller_fp_ptr: Range<u32>,
    ) -> impl Into<Tree<Directive>> {
        // Requires new instruction that is both absolute and handles the frame pointer.
        todo!()
    }

    fn emit_table_get(
        &self,
        g: &mut Generators<'a, '_, Self>,
        table_idx: u32,
        entry_idx_ptr: Range<u32>,
        dest_ptr: Range<u32>,
    ) -> impl Into<Tree<Directive>> {
        // This is a simple 3-word load from memory, but we don't have enough information to do it.
        // Must fix the Womir interface trait.
        todo!()
    }

    fn emit_wasm_op(
        &self,
        g: &mut Generators<'a, '_, Self>,
        op: WasmOp<'a>,
        inputs: Vec<Range<u32>>,
        output: Option<Range<u32>>,
    ) -> impl Into<Tree<Directive>> {
        // All ALU, memory and much more goes here...
        // Pretty much everything must be adapted to use FP-relative addressing.
        todo!()
    }
}
