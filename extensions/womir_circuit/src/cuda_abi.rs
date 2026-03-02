#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]

use openvm_cuda_backend::prelude::F;
use openvm_cuda_common::{
    d_buffer::{DeviceBuffer, DeviceBufferView},
    error::CudaError,
};

pub mod alu_cuda {
    use super::*;
    unsafe extern "C" {
        fn _womir_alu_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            range_checker_bins: usize,
            d_bitwise_lookup: *mut u32,
            bitwise_num_bits: usize,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        range_bins: usize,
        d_bitwise_lookup: &DeviceBuffer<F>,
        bitwise_num_bits: usize,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        let width = d_trace.len() / height;
        unsafe {
            CudaError::from_result(_womir_alu_tracegen(
                d_trace.as_mut_ptr(),
                height,
                width,
                d_records.view(),
                d_range_checker.as_mut_ptr() as *mut u32,
                range_bins,
                d_bitwise_lookup.as_mut_ptr() as *mut u32,
                bitwise_num_bits,
                timestamp_max_bits,
            ))
        }
    }
}
