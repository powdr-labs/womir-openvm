use itertools::Itertools;
use openvm_instructions::riscv::RV32_REGISTER_NUM_LIMBS;
use openvm_stark_backend::{
    interaction::{BusIndex, InteractionBuilder, PermutationCheckBus},
    p3_field::{FieldAlgebra, PrimeField32},
};
use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Serialize, Deserialize)]
pub struct WomController<F> {
    // TODO: Vec is fine for now with single segment as frames are sequentially allocated
    memory: Vec<Option<F>>,
}

impl<F: PrimeField32> WomController<F> {
    pub fn new() -> Self {
        WomController { memory: Vec::new() }
    }

    pub fn read<const N: usize>(&self, pointer: F) -> (WomRecord<F>, [F; N]) {
        let ptr_u32 = pointer.as_canonical_u32();
        let data = self
            .memory
            .iter()
            .skip(ptr_u32 as usize)
            .take(N)
            .map(|f| f.expect("WOM read before write"))
            .collect_vec();
        let data_array: [_; N] = data.as_slice().try_into().expect("WOM read before write");
        (WomRecord { pointer, data }, data_array)
    }

    pub fn unsafe_read_cell(&self, pointer: F) -> F {
        let ptr_u32 = pointer.as_canonical_u32();
        self.memory
            .get(ptr_u32 as usize)
            .expect("WOM read before write")
            .expect("WOM read before write")
    }

    pub fn write<const N: usize>(&mut self, pointer: F, data: [F; N]) -> WomRecord<F> {
        let ptr_u32 = pointer.as_canonical_u32();
        let needed_len = ptr_u32 as usize + N;
        if needed_len > self.memory.len() {
            self.memory.resize(needed_len, None);
        }
        // ensure data not yet written
        self.memory
            .iter_mut()
            .skip(ptr_u32 as usize)
            .take(N)
            .zip_eq(data)
            .for_each(|(old, new)| {
                assert!(old.is_none(), "WOM double write");
                *old = Some(new);
            });
        WomRecord {
            pointer,
            data: data.into(),
        }
    }

    /// Unsets all the memory entries outside of the given ranges.
    pub fn clear_unused(&mut self, sorted_used_ranges: impl Iterator<Item = (u32, u32)>) {
        let mut curr_addr = 0;
        for (start, end) in sorted_used_ranges {
            self.memory[curr_addr..start as usize].fill(None);
            curr_addr = end as usize;
        }
        self.memory.truncate(curr_addr);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
// Original OVM memory keeps a log of memory operations and returns a `RecordId`.
// This `RecordId` is later used during trace generatation to get a `MemoryRecord` with relevant data from the offline memory.
// Here, we don't keep a log and instead have operations directly return a `WomRecord` which is passed directly to trace generation.
pub struct WomRecord<F> {
    pub pointer: F,
    pub data: Vec<F>,
}

#[derive(Clone, Copy, Debug)]
pub struct WomBridge {
    bus: PermutationCheckBus,
}

impl WomBridge {
    pub const fn new(index: BusIndex) -> Self {
        Self {
            bus: PermutationCheckBus::new(index),
        }
    }

    pub fn read<T>(
        &self,
        address: impl Into<T>,
        data: [impl Into<T>; RV32_REGISTER_NUM_LIMBS],
    ) -> WomReadOperation<T> {
        WomReadOperation {
            bus: self.bus,
            address: address.into(),
            data: data.map(Into::into),
        }
    }

    pub fn write<T>(
        &self,
        address: impl Into<T>,
        data: [impl Into<T>; RV32_REGISTER_NUM_LIMBS],
        mult: impl Into<T>,
    ) -> WomWriteOperation<T> {
        WomWriteOperation {
            bus: self.bus,
            address: address.into(),
            data: data.map(Into::into),
            mult: mult.into(),
        }
    }
}

pub struct WomReadOperation<T> {
    bus: PermutationCheckBus,
    address: T,
    data: [T; RV32_REGISTER_NUM_LIMBS],
}

impl<F: FieldAlgebra> WomReadOperation<F> {
    pub fn eval<AB>(self, builder: &mut AB, enabled: impl Into<AB::Expr>)
    where
        AB: InteractionBuilder<Expr = F>,
    {
        let message = std::iter::once(self.address).chain(self.data);
        self.bus.receive(builder, message, enabled);
    }
}

pub struct WomWriteOperation<T> {
    bus: PermutationCheckBus,
    address: T,
    data: [T; RV32_REGISTER_NUM_LIMBS],
    mult: T,
}

impl<F: FieldAlgebra> WomWriteOperation<F> {
    pub fn eval<AB>(self, builder: &mut AB, enabled: impl Into<AB::Expr>)
    where
        AB: InteractionBuilder<Expr = F>,
    {
        let message = std::iter::once(self.address).chain(self.data);
        // `PermutationBus` comments say the `enabled` argument should be
        // boolean, but here we're using it as the write multiplicity.
        self.bus.send(builder, message, self.mult * enabled.into());
    }
}

#[cfg(test)]
mod test {
    use openvm_stark_backend::p3_field::FieldAlgebra;
    use openvm_stark_sdk::p3_baby_bear::BabyBear;

    use crate::WomRecord;

    use super::WomController;

    #[test]
    #[should_panic(expected = "WOM read before write")]
    fn test_wom_read_before_write1() {
        let wom = WomController::<BabyBear>::new();
        wom.read::<4>(BabyBear::from_canonical_u32(0));
    }

    #[test]
    #[should_panic(expected = "WOM read before write")]
    fn test_wom_read_before_write2() {
        let mut wom = WomController::<BabyBear>::new();
        // write to addr 4
        wom.write(
            BabyBear::from_canonical_u32(4),
            [
                BabyBear::from_canonical_u32(1),
                BabyBear::from_canonical_u32(2),
                BabyBear::from_canonical_u32(3),
                BabyBear::from_canonical_u32(4),
            ],
        );
        // read addr 0
        wom.read::<4>(BabyBear::from_canonical_u32(0));
    }

    #[test]
    #[should_panic(expected = "WOM double write")]
    fn test_wom_double_write() {
        let mut wom = WomController::<BabyBear>::new();
        // double write to addr 0
        wom.write(
            BabyBear::from_canonical_u32(0),
            [
                BabyBear::from_canonical_u32(1),
                BabyBear::from_canonical_u32(2),
                BabyBear::from_canonical_u32(3),
                BabyBear::from_canonical_u32(4),
            ],
        );
        wom.write(
            BabyBear::from_canonical_u32(0),
            [
                BabyBear::from_canonical_u32(5),
                BabyBear::from_canonical_u32(6),
                BabyBear::from_canonical_u32(7),
                BabyBear::from_canonical_u32(8),
            ],
        );
    }

    #[test]
    fn test_wom_write_read() {
        let mut wom = WomController::<BabyBear>::new();
        let p1 = BabyBear::from_canonical_u32(0);
        let w1 = [
            BabyBear::from_canonical_u32(1),
            BabyBear::from_canonical_u32(2),
            BabyBear::from_canonical_u32(3),
            BabyBear::from_canonical_u32(4),
        ];
        let p2 = BabyBear::from_canonical_u32(4);
        let w2 = [
            BabyBear::from_canonical_u32(5),
            BabyBear::from_canonical_u32(6),
            BabyBear::from_canonical_u32(7),
            BabyBear::from_canonical_u32(8),
        ];
        wom.write(p2, w2);
        wom.write(p1, w1);
        let (r1_rec, r1_data) = wom.read(p1);
        let (r2_rec, r2_data) = wom.read(p2);
        assert_eq!(
            r1_rec,
            WomRecord {
                pointer: p1,
                data: w1.into(),
            }
        );
        assert_eq!(r1_data, w1);
        assert_eq!(
            r2_rec,
            WomRecord {
                pointer: p2,
                data: w2.into(),
            }
        );
        assert_eq!(r2_data, w2);
    }
}
