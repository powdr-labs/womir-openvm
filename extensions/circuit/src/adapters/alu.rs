use std::{borrow::Borrow, marker::PhantomData};

use openvm_circuit::{
    arch::{
        AdapterAirContext, AdapterRuntimeContext, BasicAdapterInterface, ExecutionBridge,
        ExecutionBus, ExecutionState, MinimalInstruction, Result as ResultVm, VmAdapterAir,
        VmAdapterInterface,
    },
    system::{
        memory::{MemoryController, OfflineMemory, offline_checker::MemoryBridge},
        program::ProgramBus,
    },
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_IMM_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::ColumnsAir,
};
use serde::{Deserialize, Serialize};
use struct_reflection::{StructReflection, StructReflectionHelper};

use crate::{
    FrameBridge, FrameBus, FrameState, VmAdapterChipWom, WomBridge, WomController, WomRecord,
};

use super::RV32_CELL_BITS;

/// Reads instructions of the form OP a, b, c, d, e where \[a:4\]_d = \[b:4\]_d op \[c:4\]_e.
/// Operand d can only be 1, and e can be either 1 (for register reads) or 0 (when c
/// is an immediate).
pub struct WomBaseAluAdapterChip<
    F: Field,
    // How many limbs we need to read per register.
    const READ_NUM_LIMBS: usize,
    // How many limbs we need to write per register.
    const WRITE_NUM_LIMBS: usize,
    // This is just WRITE_NUM_LIMBS / RV32_REGISTER_NUM_LIMBS, but we can't use the
    // expression due to const generics limitations
    const WRITE_NUM_RV32: usize,
> {
    pub air: WomBaseAluAdapterAir<READ_NUM_LIMBS, WRITE_NUM_LIMBS, WRITE_NUM_RV32>,
    _bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    _marker: PhantomData<F>,
}

impl<
    F: PrimeField32,
    const READ_NUM_LIMBS: usize,
    const WRITE_NUM_LIMBS: usize,
    const WRITE_NUM_RV32: usize,
> WomBaseAluAdapterChip<F, READ_NUM_LIMBS, WRITE_NUM_LIMBS, WRITE_NUM_RV32>
{
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        frame_bus: FrameBus,
        memory_bridge: MemoryBridge,
        wom_bridge: WomBridge,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    ) -> Self {
        assert_eq!(READ_NUM_LIMBS % RV32_REGISTER_NUM_LIMBS, 0);
        assert_eq!(WRITE_NUM_LIMBS % RV32_REGISTER_NUM_LIMBS, 0);
        assert_eq!(WRITE_NUM_RV32 * RV32_REGISTER_NUM_LIMBS, WRITE_NUM_LIMBS);

        Self {
            air: WomBaseAluAdapterAir {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                frame_bridge: FrameBridge::new(frame_bus),
                memory_bridge,
                wom_bridge,
                bitwise_lookup_bus: bitwise_lookup_chip.bus(),
            },
            _bitwise_lookup_chip: bitwise_lookup_chip,
            _marker: PhantomData,
        }
    }
}

#[repr(C)]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "F: Field")]
pub struct WomBaseAluReadRecord<F: Field> {
    /// Read register value
    pub rs1: WomRecord<F>,
    /// Either
    /// - read rs2 register value or
    /// - if `rs2_is_imm` is true, this is None
    pub rs2: Option<WomRecord<F>>,
    /// immediate value of rs2 or 0
    pub rs2_imm: F,
}

#[repr(C)]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "[F; WRITE_NUM_LIMBS]: Serialize",
    deserialize = "[F; WRITE_NUM_LIMBS]: Deserialize<'de>"
))]
pub struct WomBaseAluWriteRecord<F: Field, const WRITE_NUM_LIMBS: usize> {
    pub from_state: ExecutionState<u32>,
    pub from_frame: FrameState<u32>,
    /// Write to destination register
    pub rd: WomRecord<F>,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct WomBaseAluAdapterCols<T, const WRITE_NUM_RV32: usize> {
    pub from_state: ExecutionState<T>,
    pub from_frame: FrameState<T>,
    pub rd_ptr: T,
    pub rs1_ptr: T,
    /// Pointer if rs2 was a read, immediate value otherwise
    pub rs2: T,
    /// 1 if rs2 was a read, 0 if an immediate
    pub rs2_as: T,
    /// TODO: needs generic NUM_WRITES parameter!
    pub write_mult: [T; WRITE_NUM_RV32],
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct WomBaseAluAdapterAir<
    const READ_NUM_LIMBS: usize,
    const WRITE_NUM_LIMBS: usize,
    const WRITE_NUM_RV32: usize,
> {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) frame_bridge: FrameBridge,
    pub(super) memory_bridge: MemoryBridge,
    pub(super) wom_bridge: WomBridge,
    bitwise_lookup_bus: BitwiseOperationLookupBus,
}

impl<F: Field, const N: usize, const M: usize, const T: usize> BaseAir<F>
    for WomBaseAluAdapterAir<N, M, T>
{
    fn width(&self) -> usize {
        WomBaseAluAdapterCols::<F, N>::width()
    }
}

impl<F: Field, const N: usize, const M: usize, const T: usize> ColumnsAir<F>
    for WomBaseAluAdapterAir<N, M, T>
{
    fn columns(&self) -> Option<Vec<String>> {
        WomBaseAluAdapterCols::<F, N>::struct_reflection()
    }
}

impl<
    AB: InteractionBuilder,
    const READ_NUM_LIMBS: usize,
    const WRITE_NUM_LIMBS: usize,
    const WRITE_NUM_RV32: usize,
> VmAdapterAir<AB> for WomBaseAluAdapterAir<READ_NUM_LIMBS, WRITE_NUM_LIMBS, WRITE_NUM_RV32>
{
    type Interface = BasicAdapterInterface<
        AB::Expr,
        MinimalInstruction<AB::Expr>,
        2,
        1,
        READ_NUM_LIMBS,
        WRITE_NUM_LIMBS,
    >;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local: &WomBaseAluAdapterCols<_, WRITE_NUM_LIMBS> = local.borrow();

        builder.assert_bool(local.rs2_as);

        // we need the following to handle the 64-bit case: wom bridge works in
        // 32-bit words, so we need 2 interactions per operation.
        let read_ops = READ_NUM_LIMBS / RV32_REGISTER_NUM_LIMBS;
        let write_ops = WRITE_NUM_RV32;

        for r in 0..read_ops {
            let reads0: [AB::Expr; RV32_REGISTER_NUM_LIMBS] =
                std::array::from_fn(|i| ctx.reads[0][r * RV32_REGISTER_NUM_LIMBS + i].clone());
            self.wom_bridge
                .read(local.rs1_ptr, reads0)
                .eval(builder, ctx.instruction.is_valid.clone());

            let reads1: [AB::Expr; RV32_REGISTER_NUM_LIMBS] =
                std::array::from_fn(|i| ctx.reads[1][r * RV32_REGISTER_NUM_LIMBS + i].clone());
            self.wom_bridge
                .read(local.rs2, reads1)
                .eval(builder, local.rs2_as);
        }
        for w in 0..write_ops {
            let writes0: [AB::Expr; RV32_REGISTER_NUM_LIMBS] =
                std::array::from_fn(|i| ctx.writes[0][w * RV32_REGISTER_NUM_LIMBS + i].clone());
            self.wom_bridge
                .write(local.rd_ptr, writes0, local.write_mult[w])
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        // we don't read memory, but ovm expects a timestamp increase
        let timestamp_change = AB::Expr::ONE;

        self.execution_bridge
            .execute_and_increment_or_set_pc(
                ctx.instruction.opcode,
                [
                    local.rd_ptr.into(),
                    local.rs1_ptr.into(),
                    local.rs2.into(),
                    AB::Expr::from_canonical_u32(RV32_REGISTER_AS),
                    local.rs2_as.into(),
                ],
                local.from_state,
                timestamp_change.clone(),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.frame_bridge
            .keep_fp(local.from_frame, timestamp_change)
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &WomBaseAluAdapterCols<_, WRITE_NUM_LIMBS> = local.borrow();
        cols.from_state.pc
    }
}

impl<
    F: PrimeField32,
    const READ_NUM_LIMBS: usize,
    const WRITE_NUM_LIMBS: usize,
    const WRITE_NUM_RV32: usize,
> VmAdapterChipWom<F> for WomBaseAluAdapterChip<F, READ_NUM_LIMBS, WRITE_NUM_LIMBS, WRITE_NUM_RV32>
where
    [F; WRITE_NUM_LIMBS]: Serialize + for<'de> Deserialize<'de>,
{
    type ReadRecord = WomBaseAluReadRecord<F>;
    type WriteRecord = WomBaseAluWriteRecord<F, WRITE_NUM_LIMBS>;
    type Air = WomBaseAluAdapterAir<READ_NUM_LIMBS, WRITE_NUM_LIMBS, WRITE_NUM_RV32>;
    type Interface =
        BasicAdapterInterface<F, MinimalInstruction<F>, 2, 1, READ_NUM_LIMBS, WRITE_NUM_LIMBS>;

    fn preprocess(
        &mut self,
        _memory: &mut MemoryController<F>,
        wom: &mut WomController<F>,
        fp: u32,
        instruction: &Instruction<F>,
    ) -> ResultVm<(
        <Self::Interface as VmAdapterInterface<F>>::Reads,
        Self::ReadRecord,
    )> {
        let Instruction { b, c, d, e, .. } = *instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        debug_assert!(
            e.as_canonical_u32() == RV32_IMM_AS || e.as_canonical_u32() == RV32_REGISTER_AS
        );

        let fp_f = F::from_canonical_u32(fp);
        assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        let (rs1, rs1_data) = wom.read::<READ_NUM_LIMBS>(b + fp_f);
        let (rs2, rs2_data, rs2_imm) = if e.is_zero() {
            let c_u32 = c.as_canonical_u32();
            debug_assert_eq!(c_u32 >> 24, 0);
            let mut c_bytes = [0u8; READ_NUM_LIMBS];
            c_bytes[0] = c_u32 as u8;
            c_bytes[1] = (c_u32 >> 8) as u8;
            let bit_extension = (c_u32 >> 16) as u8;
            for byte in &mut c_bytes[2..] {
                *byte = bit_extension;
            }
            (None, c_bytes.map(F::from_canonical_u8), c)
        } else {
            assert_eq!(e.as_canonical_u32(), RV32_REGISTER_AS);
            let (rs2, rs2_data) = wom.read::<READ_NUM_LIMBS>(c + fp_f);
            (Some(rs2), rs2_data, F::ZERO)
        };

        Ok(([rs1_data, rs2_data], Self::ReadRecord { rs1, rs2, rs2_imm }))
    }

    fn postprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        wom: &mut WomController<F>,
        instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
        from_frame: FrameState<u32>,
        output: AdapterRuntimeContext<F, Self::Interface>,
        _read_record: &Self::ReadRecord,
    ) -> ResultVm<(ExecutionState<u32>, u32, Self::WriteRecord)> {
        let Instruction { a, d, .. } = instruction;
        let fp_f = F::from_canonical_u32(from_frame.fp);
        assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        let rd = wom.write(*a + fp_f, output.writes[0]);

        memory.increment_timestamp();

        Ok((
            ExecutionState {
                pc: from_state.pc + DEFAULT_PC_STEP,
                timestamp: memory.timestamp(),
            },
            from_frame.fp,
            Self::WriteRecord {
                from_state,
                from_frame,
                rd,
            },
        ))
    }

    fn generate_trace_row(
        &self,
        _row_slice: &mut [F],
        _read_record: Self::ReadRecord,
        _write_record: Self::WriteRecord,
        _memory: &OfflineMemory<F>,
    ) {
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
