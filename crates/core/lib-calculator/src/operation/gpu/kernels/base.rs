// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use std::marker::PhantomData;

pub(crate) use super::utils::*;
pub(crate) use crate::operation::error::Result;
pub(crate) use crate::operation::gpu::{pool::GpuMemoryPool, GpuMemRep};
pub(crate) use cubecl_core::cube;
pub(crate) use cubecl_core::prelude::*;
pub(crate) use cubecl_reduce::reduce;

pub trait KernelConfig: Send + Sync + Clone + 'static {}
pub trait KernelInput: Send + Sync + Clone + 'static {}
pub trait KernelOutput: Send + Sync + Clone + 'static {}

// impl KernelConfig for NcclClientHandle {}
impl KernelConfig for usize {}
impl KernelConfig for () {}
impl KernelConfig for i32 {}

#[cfg(feature = "nccl")]
impl KernelConfig for cubecl_cuda::ReduceOp {}

#[cfg(feature = "nccl")]
impl KernelConfig for (cubecl_cuda::ReduceOp, i32) {}

#[cfg(feature = "nccl")]
impl KernelConfig for cubecl_cuda::NcclInit {}

impl KernelInput for () {}
impl KernelInput for GpuMemRep {}
impl KernelInput for (GpuMemRep, GpuMemRep) {}
impl KernelInput for (GpuMemRep, GpuMemRep, GpuMemRep) {}

impl KernelOutput for () {}
impl KernelOutput for GpuMemRep {}
impl KernelOutput for (GpuMemRep, GpuMemRep) {}

pub trait Kernel<R: Runtime, N: Numeric + CubeElement>: Send + Sync + Sized + 'static {
	type Cfg: KernelConfig;
	type Input: KernelInput;
	type Output: KernelOutput;

	fn exec(
		&self,
		order: &KernelOrder<R, N, Self>,
		pool: &GpuMemoryPool<R, N>,
	) -> Result<()>;
}

pub struct KernelOrder<R: Runtime, N: Numeric + CubeElement, K: Kernel<R, N>> {
	pub op: K,
	pub config: K::Cfg,
	pub input: K::Input,
	pub output: K::Output,
	_phantom: PhantomData<(N)>,
}

impl<R: Runtime, N: Numeric + CubeElement, K: Kernel<R, N>> KernelOrder<R, N, K> {
	pub(crate) fn order(
		op: K,
		config: K::Cfg,
		input: K::Input,
		output: K::Output,
	) -> Self {
		Self {
			op,
			config,
			input,
			output,
			_phantom: PhantomData,
		}
	}
}

pub(crate) fn to_tref<R: Runtime>(
	tuple: &(GpuMemRep, cubecl_core::server::Handle)
) -> cubecl_core::prelude::TensorHandleRef<'_, R> {
	unsafe {
		cubecl_core::prelude::TensorHandleRef::<'_, R>::from_raw_parts(
			&tuple.1,
			Box::leak(tuple.0.metadata.stride.clone()),
			Box::leak(tuple.0.metadata.shape.clone()),
			tuple.0.byte_size,
		)
	}
}
