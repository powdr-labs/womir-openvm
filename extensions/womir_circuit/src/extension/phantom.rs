use eyre::bail;
use openvm_circuit::{
    arch::{PhantomSubExecutor, Streams},
    system::memory::online::GuestMemory,
};
use openvm_instructions::PhantomDiscriminant;
use openvm_stark_backend::p3_field::{Field, PrimeField32};
use rand::{Rng, rngs::StdRng};

use crate::adapters::{memory_read, read_rv32_register};
use crate::memory_config::FpMemory;

use openvm_instructions::riscv::RV32_MEMORY_AS;

/// Reads a 32-bit register value from memory, applying FP offset.
fn read_register<F: PrimeField32>(memory: &GuestMemory, reg_offset: u32) -> u32 {
    let fp = memory.fp::<F>();
    read_rv32_register(memory, fp + reg_offset)
}

/// HintInput: Prepares the next input for hinting.
/// Pops from input_stream, prepends 4-byte length, extends hint_stream.
pub struct HintInputSubEx;

impl<F: Field> PhantomSubExecutor<F> for HintInputSubEx {
    fn phantom_execute(
        &self,
        _: &GuestMemory,
        streams: &mut Streams<F>,
        _: &mut StdRng,
        _: PhantomDiscriminant,
        _: u32,
        _: u32,
        _: u16,
    ) -> eyre::Result<()> {
        let mut hint = match streams.input_stream.pop_front() {
            Some(hint) => hint,
            None => {
                bail!("EndOfInputStream");
            }
        };
        streams.hint_stream.clear();
        // Prepend 4-byte LE length
        streams.hint_stream.extend(
            (hint.len() as u32)
                .to_le_bytes()
                .iter()
                .map(|b| F::from_canonical_u8(*b)),
        );
        // Pad to 4-byte alignment
        let capacity = hint.len().div_ceil(4) * 4;
        hint.resize(capacity, F::ZERO);
        streams.hint_stream.extend(hint);
        Ok(())
    }
}

/// PrintStr: Reads a string from memory and prints it.
/// a = mem_ptr register offset, b = num_bytes register offset, c_upper = mem_start
pub struct PrintStrSubEx;

impl<F: PrimeField32> PhantomSubExecutor<F> for PrintStrSubEx {
    fn phantom_execute(
        &self,
        memory: &GuestMemory,
        _: &mut Streams<F>,
        _: &mut StdRng,
        _: PhantomDiscriminant,
        a: u32,
        b: u32,
        c_upper: u16,
    ) -> eyre::Result<()> {
        let mem_start = c_upper as u32;
        let rd = read_register::<F>(memory, a);
        let rs1 = read_register::<F>(memory, b);
        let bytes = (0..rs1)
            .map(|i| memory_read::<1>(memory, RV32_MEMORY_AS, mem_start + rd + i)[0])
            .collect::<Vec<u8>>();
        let peeked_str = String::from_utf8(bytes)?;
        print!("{peeked_str}");
        Ok(())
    }
}

/// HintRandom: Prepares random numbers for hinting.
pub struct HintRandomSubEx;

impl<F: PrimeField32> PhantomSubExecutor<F> for HintRandomSubEx {
    fn phantom_execute(
        &self,
        memory: &GuestMemory,
        streams: &mut Streams<F>,
        rng: &mut StdRng,
        _: PhantomDiscriminant,
        a: u32,
        _: u32,
        _: u16,
    ) -> eyre::Result<()> {
        let len = read_register::<F>(memory, a) as usize;
        streams.hint_stream.clear();
        streams.hint_stream.extend(
            std::iter::repeat_with(|| F::from_canonical_u8(rng.r#gen::<u8>())).take(len * 4),
        );
        Ok(())
    }
}

/// HintLoadByKey: Loads values from the KV store into input stream.
pub struct HintLoadByKeySubEx;

impl<F: PrimeField32> PhantomSubExecutor<F> for HintLoadByKeySubEx {
    fn phantom_execute(
        &self,
        memory: &GuestMemory,
        streams: &mut Streams<F>,
        _: &mut StdRng,
        _: PhantomDiscriminant,
        a: u32,
        b: u32,
        _: u16,
    ) -> eyre::Result<()> {
        let ptr = read_register::<F>(memory, a);
        let len = read_register::<F>(memory, b);
        let key: Vec<u8> = (0..len)
            .map(|i| memory_read::<1>(memory, RV32_MEMORY_AS, ptr + i)[0])
            .collect();
        if let Some(val) = streams.kv_store.get(&key) {
            let to_push = hint_load_by_key_decode::<F>(val);
            for input in to_push.into_iter().rev() {
                streams.input_stream.push_front(input);
            }
        } else {
            bail!("HintLoadByKey: key not found");
        }
        Ok(())
    }
}

fn hint_load_by_key_decode<F: PrimeField32>(value: &[u8]) -> Vec<Vec<F>> {
    let mut offset = 0;
    let len = extract_u32(value, offset) as usize;
    offset += 4;
    let mut ret = Vec::with_capacity(len);
    for _ in 0..len {
        let v_len = extract_u32(value, offset) as usize;
        offset += 4;
        let v = (0..v_len)
            .map(|_| {
                let ret = F::from_canonical_u32(extract_u32(value, offset));
                offset += 4;
                ret
            })
            .collect();
        ret.push(v);
    }
    ret
}

fn extract_u32(value: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(value[offset..offset + 4].try_into().unwrap())
}
