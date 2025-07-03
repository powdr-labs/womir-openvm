use openvm_instructions::{exe::VmExe, instruction::Instruction, program::Program};
use openvm_stark_backend::p3_field::Field;
use womir::{
    generic_ir::GenericIrSetting,
    loader::{flattening::WriteOnceASM, func_idx_to_label},
};

pub fn program_from_wasm<F: Field>(wasm_path: &str, entry_point: &str) -> VmExe<F> {
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
            if let Directive::Instruction(i) = d {
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

#[derive(Clone)]
enum Directive<F: Clone> {
    Nop,
    Label { id: String, frame_size: Option<u32> },
    Instruction(Instruction<F>),
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

fn translate_directives<F: Clone>(directive: womir::generic_ir::Directive) -> Vec<Directive<F>> {
    use womir::generic_ir::Directive::*;
    match directive {
        Label { id, frame_size } => {
            vec![Directive::Label { id, frame_size }]
        }
        AllocateFrameI {
            target_frame,
            result_ptr,
        } => todo!(),
        AllocateFrameV {
            frame_size,
            result_ptr,
        } => todo!(),
        Copy {
            src_word,
            dest_word,
        } => todo!(),
        CopyIntoFrame {
            src_word,
            dest_frame,
            dest_word,
        } => todo!(),
        Jump { target } => todo!(),
        JumpOffset { offset } => todo!(),
        JumpIf { target, condition } => todo!(),
        JumpAndActivateFrame {
            target,
            new_frame_ptr,
            saved_caller_fp,
        } => todo!(),
        Return { ret_pc, ret_fp } => todo!(),
        Call {
            target,
            new_frame_ptr,
            saved_ret_pc,
            saved_caller_fp,
        } => todo!(),
        CallIndirect {
            target_pc,
            new_frame_ptr,
            saved_ret_pc,
            saved_caller_fp,
        } => todo!(),
        ImportedCall {
            module,
            function,
            inputs,
            outputs,
        } => todo!(),
        Trap { reason } => todo!(),
        WASMOp { op, inputs, output } => todo!(),
    }
}
