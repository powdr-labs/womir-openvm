use crate::{adapters::AllocateFrameAdapterExecutor, allocate_frame::core::AllocateFrameExecutor};

mod core;

pub type WomAllocateFrameExecutor<F> = AllocateFrameExecutor<AllocateFrameAdapterExecutor<F>>;
