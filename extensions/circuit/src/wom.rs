use itertools::Itertools;
use openvm_stark_backend::{
    interaction::{BusIndex, InteractionBuilder, PermutationCheckBus},
    p3_field::{FieldAlgebra, PrimeField32},
};

#[derive(Debug)]
pub struct WomController<F> {
    bus: PermutationCheckBus,
    // TODO: Vec is fine for now with single segment as frames are sequentially allocated
    memory: Vec<Option<F>>,
}

impl<F: PrimeField32> WomController<F> {
    pub fn new(bus: PermutationCheckBus) -> Self {
        WomController {
            bus,
            memory: Vec::new(),
        }
    }

    pub fn bridge(&self) -> WomBridge {
        WomBridge { bus: self.bus }
    }

    pub fn read<const N: usize>(&self, pointer: F) -> [F; N] {
        let ptr_u32 = pointer.as_canonical_u32();
        self.memory
            .iter()
            .skip(ptr_u32 as usize)
            .take(N)
            .map(|f| f.expect("WOM read before write"))
            .collect_array()
            .expect("WOM read before write")
    }

    pub fn write<const N: usize>(&mut self, pointer: F, data: [F; N]) {
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
    }
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

    pub fn read<T, const N: usize>(
        &self,
        address: impl Into<T>,
        data: [impl Into<T>; N],
    ) -> WomReadOperation<T, N> {
        WomReadOperation {
            bus: self.bus,
            address: address.into(),
            data: data.map(Into::into),
        }
    }

    pub fn write<T, const N: usize>(
        &self,
        address: impl Into<T>,
        data: [impl Into<T>; N],
        mult: T,
    ) -> WomWriteOperation<T, N> {
        WomWriteOperation {
            bus: self.bus,
            address: address.into(),
            data: data.map(Into::into),
            mult,
        }
    }
}

pub struct WomReadOperation<T, const N: usize> {
    bus: PermutationCheckBus,
    address: T,
    data: [T; N],
}

impl<F: FieldAlgebra, const N: usize> WomReadOperation<F, N> {
    pub fn eval<AB>(self, builder: &mut AB, enabled: impl Into<AB::Expr>)
    where
        AB: InteractionBuilder<Expr = F>,
    {
        let message = std::iter::once(self.address).chain(self.data);
        self.bus.receive(builder, message, enabled);
    }
}

pub struct WomWriteOperation<T, const N: usize> {
    bus: PermutationCheckBus,
    address: T,
    data: [T; N],
    mult: T,
}

impl<F: FieldAlgebra, const N: usize> WomWriteOperation<F, N> {
    pub fn eval<AB>(self, builder: &mut AB, enabled: impl Into<AB::Expr>)
    where
        AB: InteractionBuilder<Expr = F>,
    {
        let message = std::iter::once(self.address).chain(self.data);
        // TODO: how to handle multiplicity? does the PermutationBus handle it (comments say to ensure boolean)
        self.bus.send(builder, message, self.mult * enabled.into());
    }
}

#[cfg(test)]
mod test {
    use openvm_stark_backend::interaction::PermutationCheckBus;
    use openvm_stark_backend::p3_field::FieldAlgebra;
    use openvm_stark_sdk::p3_baby_bear::BabyBear;

    use super::WomController;

    #[test]
    #[should_panic]
    fn test_wom_read_before_write1() {
        let wom = WomController::<BabyBear>::new(PermutationCheckBus::new(0));
        wom.read::<4>(BabyBear::from_canonical_u32(0));
    }

    #[test]
    #[should_panic]
    fn test_wom_read_before_write2() {
        let mut wom = WomController::<BabyBear>::new(PermutationCheckBus::new(0));
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
    #[should_panic]
    fn test_wom_double_write() {
        let mut wom = WomController::<BabyBear>::new(PermutationCheckBus::new(0));
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
        let mut wom = WomController::<BabyBear>::new(PermutationCheckBus::new(0));
        let w1 = [
            BabyBear::from_canonical_u32(1),
            BabyBear::from_canonical_u32(2),
            BabyBear::from_canonical_u32(3),
            BabyBear::from_canonical_u32(4),
        ];
        let w2 = [
            BabyBear::from_canonical_u32(5),
            BabyBear::from_canonical_u32(6),
            BabyBear::from_canonical_u32(7),
            BabyBear::from_canonical_u32(8),
        ];
        wom.write(BabyBear::from_canonical_u32(4), w2);
        wom.write(BabyBear::from_canonical_u32(0), w1);
        assert_eq!(wom.read(BabyBear::from_canonical_u32(0)), w1);
        assert_eq!(wom.read(BabyBear::from_canonical_u32(4)), w2);
    }
}
