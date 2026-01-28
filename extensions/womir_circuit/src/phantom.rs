// Phantom sub-executors for hint operations
// Re-implements the same logic as upstream rv32im for compatibility

use eyre::bail;
use openvm_circuit::{
    arch::{PhantomSubExecutor, Streams},
    system::memory::online::GuestMemory,
};
use openvm_instructions::PhantomDiscriminant;
use openvm_stark_backend::p3_field::{Field, PrimeField32};
use rand::rngs::StdRng;

// Helper function to decode bytes from KV store
fn hint_load_by_key_decode<F: PrimeField32>(value: &[u8]) -> Vec<Vec<F>> {
    fn extract_u32(bytes: &[u8], offset: usize) -> u32 {
        u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ])
    }

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

pub struct HintInputSubEx;
pub struct HintRandomSubEx;
pub struct PrintStrSubEx;
pub struct HintLoadByKeySubEx;

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
            None => bail!("EndOfInputStream"),
        };
        streams.hint_stream.clear();
        // Prepend length as 4 bytes
        streams.hint_stream.extend(
            (hint.len() as u32)
                .to_le_bytes()
                .iter()
                .map(|b| F::from_canonical_u8(*b)),
        );
        // Extend by 0 for 4 byte alignment
        let capacity = hint.len().div_ceil(4) * 4;
        hint.resize(capacity, F::ZERO);
        streams.hint_stream.extend(hint);
        Ok(())
    }
}

impl<F: Field> PhantomSubExecutor<F> for HintRandomSubEx {
    fn phantom_execute(
        &self,
        _: &GuestMemory,
        streams: &mut Streams<F>,
        rng: &mut StdRng,
        _: PhantomDiscriminant,
        a: u32,
        _: u32,
        _: u16,
    ) -> eyre::Result<()> {
        use rand::RngCore;
        streams.hint_stream.clear();
        for _ in 0..a {
            let random_value = rng.next_u32();
            streams
                .hint_stream
                .extend(random_value.to_le_bytes().map(F::from_canonical_u8));
        }
        Ok(())
    }
}

impl<F: Field> PhantomSubExecutor<F> for PrintStrSubEx {
    fn phantom_execute(
        &self,
        memory: &GuestMemory,
        _: &mut Streams<F>,
        _: &mut StdRng,
        _: PhantomDiscriminant,
        a: u32,
        b: u32,
        _: u16,
    ) -> eyre::Result<()> {
        use crate::adapters::{memory_read, read_rv32_register};

        let rd = read_rv32_register(memory, a);
        let rs1 = read_rv32_register(memory, b);
        let bytes = (0..rs1)
            .map(|i| memory_read::<1>(memory, 2, rd + i)[0])
            .collect::<Vec<u8>>();
        let peeked_str = String::from_utf8(bytes)?;
        print!("{peeked_str}");
        Ok(())
    }
}

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
        use crate::adapters::{memory_read, read_rv32_register};

        let ptr = read_rv32_register(memory, a);
        let len = read_rv32_register(memory, b);
        let key: Vec<u8> = (0..len)
            .map(|i| memory_read::<1>(memory, 2, ptr + i)[0])
            .collect();

        if let Some(val) = streams.kv_store.get(&key) {
            let to_push = hint_load_by_key_decode::<F>(val);
            for input in to_push.into_iter().rev() {
                streams.input_stream.push_front(input);
            }
            Ok(())
        } else {
            bail!("HintLoadByKey: key not found")
        }
    }
}
