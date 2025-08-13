use std::{error::Error, num::TryFromIntError};

use openvm_stark_backend::p3_field::PrimeField32;

/// Convenience trait to safely convert an integer into a field element.
pub trait ToField<F: PrimeField32> {
    fn to_f(self) -> Result<F, Box<dyn Error>>;
}

impl<I: TryInto<u32, Error = TryFromIntError>, F: PrimeField32> ToField<F> for I {
    fn to_f(self) -> Result<F, Box<dyn Error>> {
        // This is a PrimeField32, so any value greater than 32 bits won't fit.
        let self32: u32 = self.try_into()?;
        match self32 {
            0 => Ok(F::ZERO),
            1 => Ok(F::ONE),
            _ => {
                let max = F::NEG_ONE.as_canonical_u32();

                if self32 > max {
                    Err(Box::new(ToFieldError {
                        value: self32,
                        max_value: max,
                    }))
                } else {
                    Ok(F::from_canonical_u32(self32))
                }
            }
        }
    }
}

#[derive(Debug)]
struct ToFieldError {
    value: u32,
    max_value: u32,
}

impl Error for ToFieldError {}

impl std::fmt::Display for ToFieldError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Value {} exceeds maximum allowed value {}",
            self.value, self.max_value
        )
    }
}
