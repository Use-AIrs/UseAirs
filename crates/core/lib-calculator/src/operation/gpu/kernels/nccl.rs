// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use cubecl_cuda::{NcclClientHandle, NcclExtension, NcclServerHandle, ReduceOp};

use cubecl_runtime::ext::ExtensionError;

use cubecl_cuda::CudaRuntime;

use cudarc::nccl::NcclType;

use super::base::*;
use crate::operation::error::OpError;
use std::any::Any;

#[derive(Clone, Copy)]
pub struct NcclInitialize;

impl<R: Runtime, N: Numeric + CubeElement> Kernel<R, N> for NcclInitialize {
	type Cfg = cubecl_cuda::NcclInit;
	type Input = ();
	type Output = ();

	fn exec(
		&self,
		order: &KernelOrder<R, N, Self>,
		pool: &GpuMemoryPool<R, N>,
	) -> Result<()> {
		let pool_any: &dyn Any = pool;

		if let Some(cuda_pool) = pool_any.downcast_ref::<GpuMemoryPool<CudaRuntime, N>>() {
			cuda_pool
				.client()
				.init_extention::<NcclExtension>(order.config.clone())
				.map_err(|e| OpError::ExtensionError(format!("{:?}", e)))?;
			Ok(())
		} else {
			Err(OpError::InvalidConfiguration)
		}
	}
}

#[derive(Clone, Copy)]
pub struct AllReduce;

impl<R: Runtime, N: Numeric + CubeElement> Kernel<R, N> for AllReduce {
	type Cfg = ReduceOp;
	type Input = GpuMemRep;
	type Output = GpuMemRep;

	fn exec(
		&self,
		order: &KernelOrder<R, N, Self>,
		pool: &GpuMemoryPool<R, N>,
	) -> Result<()> {
		let pool_any: &dyn Any = pool;

		if let Some(cuda_pool) = pool_any.downcast_ref::<GpuMemoryPool<CudaRuntime, N>>() {
			let input = &order.input;
			let output = &order.output;

			let (_, input_handle) = cuda_pool.get_handles(input)?;
			let (_, output_handle) = cuda_pool.get_handles(output)?;
			let in_bind = input_handle.binding();
			let out_bind = output_handle.binding();
			let count = input.metadata.total_elements();

			let dt = get_nccl_type::<N>();

			let nccl = NcclClientHandle::new(Some(in_bind), Some(out_bind), count, dt);

			let op_fn: fn(NcclServerHandle) -> std::result::Result<(), ExtensionError> =
				match order.config {
					ReduceOp::Sum => all_reduce_sum,
					ReduceOp::Prod => all_reduce_prod,
					ReduceOp::Max => all_reduce_max,
					ReduceOp::Min => all_reduce_min,
					ReduceOp::Avg => all_reduce_avg,
					ReduceOp::Custom(_) => return Err(OpError::InvalidConfiguration),
				};

			cuda_pool
				.client()
				.fn_extension::<NcclExtension>(nccl, op_fn)?;

			Ok(())
		} else {
			Err(OpError::InvalidConfiguration)
		}
	}
}

fn all_reduce_sum(handle: NcclServerHandle) -> std::result::Result<(), ExtensionError> {
	handle.all_reduce(ReduceOp::Sum)
}

fn all_reduce_prod(handle: NcclServerHandle) -> std::result::Result<(), ExtensionError> {
	handle.all_reduce(ReduceOp::Prod)
}

fn all_reduce_max(handle: NcclServerHandle) -> std::result::Result<(), ExtensionError> {
	handle.all_reduce(ReduceOp::Max)
}

fn all_reduce_min(handle: NcclServerHandle) -> std::result::Result<(), ExtensionError> {
	handle.all_reduce(ReduceOp::Min)
}

fn all_reduce_avg(handle: NcclServerHandle) -> std::result::Result<(), ExtensionError> {
	handle.all_reduce(ReduceOp::Avg)
}

#[derive(Clone, Copy)]
pub struct Broadcast;

impl<R: Runtime, N: Numeric + CubeElement> Kernel<R, N> for Broadcast {
	type Cfg = i32;
	type Input = GpuMemRep;
	type Output = GpuMemRep;

	fn exec(
		&self,
		order: &KernelOrder<R, N, Self>,
		pool: &GpuMemoryPool<R, N>,
	) -> Result<()> {
		let pool_any: &dyn Any = pool;

		if let Some(cuda_pool) = pool_any.downcast_ref::<GpuMemoryPool<CudaRuntime, N>>() {
			let input = &order.input;
			let output = &order.output;

			let (_, input_handle) = cuda_pool.get_handles(input)?;
			let (_, output_handle) = cuda_pool.get_handles(output)?;
			let in_bind = input_handle.binding();
			let out_bind = output_handle.binding();
			let count = input.metadata.total_elements();
			let dt = get_nccl_type::<N>();

			let nccl = NcclClientHandle::new(Some(in_bind), Some(out_bind), count, dt);

			let op_fn: fn(NcclServerHandle) -> std::result::Result<(), ExtensionError> =
				match order.config {
					0 => broadcast_root_0,
					1 => broadcast_root_1,
					2 => broadcast_root_2,
					3 => broadcast_root_3,
					_ => return Err(OpError::InvalidConfiguration),
				};

			cuda_pool
				.client()
				.fn_extension::<NcclExtension>(nccl, op_fn)?;

			Ok(())
		} else {
			Err(OpError::InvalidConfiguration)
		}
	}
}

fn broadcast_root_0(handle: NcclServerHandle) -> std::result::Result<(), ExtensionError> {
	handle.broadcast(0)
}

fn broadcast_root_1(handle: NcclServerHandle) -> std::result::Result<(), ExtensionError> {
	handle.broadcast(1)
}

fn broadcast_root_2(handle: NcclServerHandle) -> std::result::Result<(), ExtensionError> {
	handle.broadcast(2)
}

fn broadcast_root_3(handle: NcclServerHandle) -> std::result::Result<(), ExtensionError> {
	handle.broadcast(3)
}

#[derive(Clone, Copy)]
pub struct AllGather;

impl<R: Runtime, N: Numeric + CubeElement> Kernel<R, N> for AllGather {
	type Cfg = ();
	type Input = GpuMemRep;
	type Output = GpuMemRep;

	fn exec(
		&self,
		order: &KernelOrder<R, N, Self>,
		pool: &GpuMemoryPool<R, N>,
	) -> Result<()> {
		let pool_any: &dyn Any = pool;

		if let Some(cuda_pool) = pool_any.downcast_ref::<GpuMemoryPool<CudaRuntime, N>>() {
			let input = &order.input;
			let output = &order.output;

			let (_, input_handle) = cuda_pool.get_handles(input)?;
			let (_, output_handle) = cuda_pool.get_handles(output)?;
			let in_bind = input_handle.binding();
			let out_bind = output_handle.binding();
			let count = input.metadata.total_elements();
			let dt = get_nccl_type::<N>();

			let nccl = NcclClientHandle::new(Some(in_bind), Some(out_bind), count, dt);

			cuda_pool
				.client()
				.fn_extension::<NcclExtension>(nccl, all_gather_op)?;

			Ok(())
		} else {
			Err(OpError::InvalidConfiguration)
		}
	}
}

fn all_gather_op(handle: NcclServerHandle) -> std::result::Result<(), ExtensionError> {
	handle.all_gather()
}

#[derive(Clone, Copy)]
pub struct ReduceScatter;

impl<R: Runtime, N: Numeric + CubeElement> Kernel<R, N> for ReduceScatter {
	type Cfg = ReduceOp;
	type Input = GpuMemRep;
	type Output = GpuMemRep;

	fn exec(
		&self,
		order: &KernelOrder<R, N, Self>,
		pool: &GpuMemoryPool<R, N>,
	) -> Result<()> {
		let pool_any: &dyn Any = pool;

		if let Some(cuda_pool) = pool_any.downcast_ref::<GpuMemoryPool<CudaRuntime, N>>() {
			let input = &order.input;
			let output = &order.output;

			let (_, input_handle) = cuda_pool.get_handles(input)?;
			let (_, output_handle) = cuda_pool.get_handles(output)?;
			let in_bind = input_handle.binding();
			let out_bind = output_handle.binding();
			let count = input.metadata.total_elements();
			let dt = get_nccl_type::<N>();

			let nccl = NcclClientHandle::new(Some(in_bind), Some(out_bind), count, dt);

			let op_fn: fn(NcclServerHandle) -> std::result::Result<(), ExtensionError> =
				match order.config {
					ReduceOp::Sum => reduce_scatter_sum,
					ReduceOp::Prod => reduce_scatter_prod,
					ReduceOp::Max => reduce_scatter_max,
					ReduceOp::Min => reduce_scatter_min,
					ReduceOp::Avg => reduce_scatter_avg,
					ReduceOp::Custom(_) => return Err(OpError::InvalidConfiguration),
				};

			cuda_pool
				.client()
				.fn_extension::<NcclExtension>(nccl, op_fn)?;

			Ok(())
		} else {
			Err(OpError::InvalidConfiguration)
		}
	}
}

fn reduce_scatter_sum(handle: NcclServerHandle) -> std::result::Result<(), ExtensionError> {
	handle.reduce_scatter(ReduceOp::Sum)
}

fn reduce_scatter_prod(handle: NcclServerHandle) -> std::result::Result<(), ExtensionError> {
	handle.reduce_scatter(ReduceOp::Prod)
}

fn reduce_scatter_max(handle: NcclServerHandle) -> std::result::Result<(), ExtensionError> {
	handle.reduce_scatter(ReduceOp::Max)
}

fn reduce_scatter_min(handle: NcclServerHandle) -> std::result::Result<(), ExtensionError> {
	handle.reduce_scatter(ReduceOp::Min)
}

fn reduce_scatter_avg(handle: NcclServerHandle) -> std::result::Result<(), ExtensionError> {
	handle.reduce_scatter(ReduceOp::Avg)
}

fn get_nccl_type<N: 'static>() -> cudarc::nccl::sys::ncclDataType_t {
	use cudarc::nccl::sys::ncclDataType_t;
	use std::any::TypeId;

	let type_id = TypeId::of::<N>();

	if type_id == TypeId::of::<f32>() {
		ncclDataType_t::ncclFloat32
	} else if type_id == TypeId::of::<f64>() {
		ncclDataType_t::ncclFloat64
	} else {
		// Default to f32 if type is unknown
		ncclDataType_t::ncclFloat32
	}
}
