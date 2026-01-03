// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use crate::operation::error::{OpError, Result};
use crate::operation::gpu::GpuMemRep;
use crate::{MetaData, Tensor};
use cubecl::client;
use cubecl_common::bytes::Bytes;
use cubecl_common::device::Device;
use cubecl_common::device::DeviceId;
use cubecl_core::prelude::*;
use cubecl_core::server::Handle;
use cubecl_reduce::ReduceStrategy;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

#[cfg(feature = "nccl")]
use cubecl_cuda::*;

#[cfg(feature = "nccl")]
use cudarc::nccl::sys::ncclUniqueId;

pub struct GpuMemoryPool<R: Runtime, N: Numeric + CubeElement> {
	pub(crate) dev: R::Device,
	client: ComputeClient<R::Server>,
	pub(crate) allocations: Arc<Mutex<HashMap<GpuMemRep, Handle>>>,
	next_id: Arc<Mutex<usize>>,
	pub(crate) strategy: ReduceStrategy,
	pub(crate) dev_count: usize,
	_pd: PhantomData<N>,
}

impl<R: Runtime, N: Numeric + CubeElement> GpuMemoryPool<R, N> {
	pub fn new(dev_id: DeviceId) -> Result<Self> {
		let dev_count = R::Device::device_count_total();
		let dev = R::Device::from_id(dev_id);
		let client = R::client(&dev);
		let allocations = Arc::new(Mutex::new(HashMap::new()));
		let next_id = Arc::new(Mutex::new(0));
		let strategy = ReduceStrategy::new::<R>(&client, true);

		Ok(GpuMemoryPool {
			dev,
			client,
			allocations,
			next_id,
			strategy,
			dev_count,
			_pd: PhantomData::<N>,
		})
	}

	fn create_handle(
		&self,
		metadata: &MetaData,
	) -> Result<GpuMemRep> {
		let mut next_id = self.next_id.lock()?;
		let id = *next_id;
		*next_id += 1;
		let byte_size = size_of::<N>();
		let mh = GpuMemRep {
			id,
			size: metadata.total_elements() * byte_size,
			byte_size,
			metadata: metadata.clone(),
		};
		Ok(mh)
	}

	pub fn allocate_empty(
		&self,
		metadata: &MetaData,
	) -> Result<GpuMemRep> {
		let cpu_handle = self.create_handle(metadata)?;
		let gpu_handle = self.client().empty(cpu_handle.size);
		{
			let mut allocations = self.allocations.lock().unwrap();
			allocations.insert(cpu_handle.clone(), gpu_handle.clone());
		}

		Ok(cpu_handle)
	}

	pub fn allocate_from_tensor(
		&self,
		tensor: &Tensor<N>,
	) -> Result<GpuMemRep> {
		let gpu_handle = self.client().create_tensor(
			tensor.data(),
			tensor.shape(),
			N::elem_size() as usize,
		);
		let cpu_handle = self.create_handle(&tensor.metadata)?;

		{
			let mut allocations = self.allocations.lock().unwrap();
			allocations.insert(cpu_handle.clone(), gpu_handle.handle);
		}

		Ok(cpu_handle)
	}

	pub fn get_handles(
		&self,
		mem_handle: &GpuMemRep,
	) -> Result<(GpuMemRep, Handle)> {
		let allocations = self.allocations.lock()?;
		match allocations.get(mem_handle) {
			Some(handle) => Ok((mem_handle.clone(), handle.clone())),
			None => Err(OpError::TensorNotAvitable),
		}
	}

	pub fn get_data(
		&self,
		mem_handle: &GpuMemRep,
	) -> Result<Vec<N>> {
		let allocations = self.allocations.lock()?;
		match allocations.get(mem_handle) {
			Some(handle) => {
				let res = self.client().read_one(handle.clone());
				let out: Vec<N> = match res.try_into_vec() {
					Ok(vec) => vec,
					Err(bytes) => bytemuck::cast_slice(&bytes).to_vec(),
				};
				Ok(out)
			},
			None => Err(OpError::TensorNotAvitable),
		}
	}

	pub fn client(&self) -> &ComputeClient<R::Server> {
		&self.client
	}

	pub fn device(&self) -> &R::Device {
		&self.dev
	}

	pub fn deallocate(
		&self,
		mem_rep: &GpuMemRep,
	) -> Result<()> {
		let mut allocations = self.allocations.lock().unwrap();
		if let Some(_handle) = allocations.remove(mem_rep) {
			Ok(())
		} else {
			Err(OpError::TensorNotAvitable)
		}
	}
}
