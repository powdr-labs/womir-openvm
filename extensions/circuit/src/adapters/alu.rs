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
    riscv::{RV32_IMM_AS, RV32_REGISTER_AS},
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, PrimeField32},
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
    // How many 32-bit words we need to read in total. In 32-bit arch we need 2, in 64-bit arch we need 4.
    const READ_32BIT_WORDS: usize,
    // How many bytes we need to read per register.
    const READ_BYTES: usize,
    // How many bytes we need to write per register.
    const WRITE_BYTES: usize,
> {
    pub air: WomBaseAluAdapterAir<READ_32BIT_WORDS, READ_BYTES, WRITE_BYTES>,
    _bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    _marker: PhantomData<F>,
}

impl<
    F: PrimeField32,
    const READ_32BIT_WORDS: usize,
    const READ_BYTES: usize,
    const WRITE_BYTES: usize,
> WomBaseAluAdapterChip<F, READ_32BIT_WORDS, READ_BYTES, WRITE_BYTES>
{
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        frame_bus: FrameBus,
        memory_bridge: MemoryBridge,
        wom_bridge: WomBridge,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    ) -> Self {
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
    serialize = "[F; WRITE_BYTES]: Serialize",
    deserialize = "[F; WRITE_BYTES]: Deserialize<'de>"
))]
pub struct WomBaseAluWriteRecord<F: Field, const WRITE_BYTES: usize> {
    pub from_state: ExecutionState<u32>,
    pub from_frame: FrameState<u32>,
    /// Write to destination register
    pub rd: WomRecord<F>,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct WomBaseAluAdapterCols<T, const READ_32BIT_WORDS: usize, const WRITE_BYTES: usize> {
    pub from_state: ExecutionState<T>,
    pub from_frame: FrameState<T>,
    pub rd_ptr: T,
    pub rs1_ptr: T,
    /// Pointer if rs2 was a read, immediate value otherwise
    pub rs2: T,
    /// 1 if rs2 was a read, 0 if an immediate
    pub rs2_as: T,
    pub reads_aux: [[T; 2]; READ_32BIT_WORDS],
    pub writes_aux: [T; WRITE_BYTES],
    pub write_mult: T,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct WomBaseAluAdapterAir<
    const READ_32BIT_WORDS: usize,
    const READ_BYTES: usize,
    const WRITE_BYTES: usize,
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
        WomBaseAluAdapterCols::<F, N, T>::width()
    }
}

impl<F: Field, const N: usize, const M: usize, const T: usize> ColumnsAir<F>
    for WomBaseAluAdapterAir<N, M, T>
{
    fn columns(&self) -> Option<Vec<String>> {
        WomBaseAluAdapterCols::<F, N, T>::struct_reflection()
    }
}

impl<
    AB: InteractionBuilder,
    const READ_32BIT_WORDS: usize,
    const READ_BYTES: usize,
    const WRITE_BYTES: usize,
> VmAdapterAir<AB> for WomBaseAluAdapterAir<READ_32BIT_WORDS, READ_BYTES, WRITE_BYTES>
{
    type Interface = BasicAdapterInterface<
        AB::Expr,
        MinimalInstruction<AB::Expr>,
        2,
        1,
        READ_BYTES,
        WRITE_BYTES,
    >;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local: &WomBaseAluAdapterCols<_, READ_32BIT_WORDS, WRITE_BYTES> = local.borrow();

        builder.assert_bool(local.rs2_as);

        self.wom_bridge
            .read(local.rs1_ptr, ctx.reads[0].clone())
            .eval(builder, ctx.instruction.is_valid.clone());
        builder
            .when(local.rs2_as)
            .assert_one(ctx.instruction.is_valid.clone());
        self.wom_bridge
            .read(local.rs2, ctx.reads[1].clone())
            .eval(builder, local.rs2_as.clone());
        self.wom_bridge
            .write(local.rd_ptr, ctx.writes[0].clone(), local.write_mult)
            .eval(builder, ctx.instruction.is_valid.clone());
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &WomBaseAluAdapterCols<_, READ_32BIT_WORDS, WRITE_BYTES> = local.borrow();
        cols.from_state.pc
    }
}

impl<
    F: PrimeField32,
    const READ_32BIT_WORDS: usize,
    const READ_BYTES: usize,
    const WRITE_BYTES: usize,
> VmAdapterChipWom<F> for WomBaseAluAdapterChip<F, READ_32BIT_WORDS, READ_BYTES, WRITE_BYTES>
where
    [F; WRITE_BYTES]: Serialize + for<'de> Deserialize<'de>,
{
    type ReadRecord = WomBaseAluReadRecord<F>;
    type WriteRecord = WomBaseAluWriteRecord<F, WRITE_BYTES>;
    type Air = WomBaseAluAdapterAir<READ_32BIT_WORDS, READ_BYTES, WRITE_BYTES>;
    type Interface = BasicAdapterInterface<F, MinimalInstruction<F>, 2, 1, READ_BYTES, WRITE_BYTES>;

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
        let (rs1, rs1_data) = wom.read::<READ_BYTES>(b + fp_f);
        let (rs2, rs2_data, rs2_imm) = if e.is_zero() {
            let c_u32 = c.as_canonical_u32();
            debug_assert_eq!(c_u32 >> 24, 0);
            let mut c_bytes = [0u8; READ_BYTES];
            c_bytes[0] = c_u32 as u8;
            c_bytes[1] = (c_u32 >> 8) as u8;
            let bit_extension = (c_u32 >> 16) as u8;
            for byte in &mut c_bytes[2..] {
                *byte = bit_extension;
            }
            (None, c_bytes.map(F::from_canonical_u8), c)
        } else {
            assert_eq!(e.as_canonical_u32(), RV32_REGISTER_AS);
            let (rs2, rs2_data) = wom.read::<READ_BYTES>(c + fp_f);
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

        // let timestamp_delta = memory.timestamp() - from_state.timestamp;
        // debug_assert!(
        //     timestamp_delta == 3,
        //     "timestamp delta is {timestamp_delta}, expected 3"
        // );

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
