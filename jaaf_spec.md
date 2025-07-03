# jaaf

Implement the following WOMIR instructions in the existing jaaf chip, but only changing the `preprocess` and `postprocess` functions in the jaaf adapter, leave the chip and trace gen alone for now.

jaaf / jaaf_save / ret / call / call_indirect

Add the missing instructions to transpiler.rs in the same enum as Jaaf.
Use the current implementation of jaaf and alu to detect which local opcode it is on top of the enum.
Also use https://github.com/powdr-labs/womir/blob/main/src/interpreter.rs#L146 to infer the behavior of each instruction.

- jaaf always sets the pc using the imm argument, sets fp and does not save any
- jaaf_save same but saves the fp
- ret sets both (pc from reg), saves neither
- call sets pc from label, sets fp, and saves both
- call_indirect sets pc from reg, sets fp, and saves both

Arguments {a,b..g}:

rd1 (where to save current pc if need), rd2 (where to save current fp if needed), rs1 (target pc from reg if needed), imm: F (target pc from label if needed), rs2 (target fp)
