use std::num::TryFromIntError;

use openvm_stark_backend::p3_field::PrimeField32;

/// Convenience trait to safely convert an integer into a field element.
pub trait ToField<F: PrimeField32> {
    fn to_f(self) -> Result<F, ToFieldError>;
}

impl<I: TryInto<u32, Error = TryFromIntError>, F: PrimeField32> ToField<F> for I {
    fn to_f(self) -> Result<F, ToFieldError> {
        // This is a PrimeField32, so any value greater than 32 bits won't fit.
        let self32: u32 = self.try_into()?;
        match self32 {
            0 => Ok(F::ZERO),
            1 => Ok(F::ONE),
            _ => {
                let max = F::NEG_ONE.as_canonical_u32();

                if self32 > max {
                    Err(ToFieldError::ToF {
                        value: self32,
                        max_value: max,
                    })
                } else {
                    Ok(F::from_canonical_u32(self32))
                }
            }
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ToFieldError {
    #[error("Failed to convert to u32: {0}")]
    ToU32(#[from] TryFromIntError),
    #[error("Value {value} exceeds maximum allowed value {max_value}")]
    ToF { value: u32, max_value: u32 },
}
