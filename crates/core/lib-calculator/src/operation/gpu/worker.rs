// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use super::kernels::{Kernel, KernelOrder};
use super::pool::GpuMemoryPool;
use super::GpuMemRep;
use crate::operation::error::Result;
use crate::{MetaData, Tensor};
use crossbeam::channel::{Receiver, Sender};
use cubecl::prelude::{CubeElement, Numeric, Runtime};
use cubecl_common::device::{Device, DeviceId};
use std::any::Any;

#[cfg(feature = "nccl")]
use cubecl_cuda::CudaRuntime;
#[cfg(feature = "nccl")]
use cubecl_runtime::ext::SupportsExt;

use cudarc::nccl::sys::ncclUniqueId;

pub(crate) enum GpuCommand<R: Runtime, N: Numeric + CubeElement> {
	CopyTensor {
		tensor: Tensor<N>,
		response: Sender<GpuResponse<N>>,
	},
	CreateEmpty {
		md: MetaData,
		response: Sender<GpuResponse<N>>,
	},
	GetData {
		mem_rep: GpuMemRep,
		response: Sender<GpuResponse<N>>,
	},
	ExecuteKernel {
		executor: Box<dyn FnOnce(&GpuMemoryPool<R, N>) -> Result<()> + Send>,
		response: Sender<GpuResponse<N>>,
	},
	Synchronize {
		response: Sender<GpuResponse<N>>,
	},
	DeallocateHandle {
		mem_rep: GpuMemRep,
		response: Sender<GpuResponse<N>>,
	},
	Shutdown,
}

#[derive(Clone, Debug)]
pub(crate) enum GpuResponse<N: Numeric + CubeElement> {
	HandleCreated { dev_id: DeviceId, handle: GpuMemRep },
	DataRetrieved { dev_id: DeviceId, data: Vec<N> },
	Executed { dev_id: DeviceId },
	Synchronized { dev_id: DeviceId },
	Deallocated { dev_id: DeviceId },
	Error { dev_id: DeviceId, message: String },
}

pub(crate) fn gpu_worker<R: Runtime, N: Numeric + CubeElement>(
	dev_id: DeviceId,
	rx: Receiver<GpuCommand<R, N>>,
	nccl_id: Option<ncclUniqueId>,
) -> Result<()> {
	let gpu_pool = GpuMemoryPool::<R, N>::new(dev_id)?;

	#[cfg(feature = "nccl")]
	{
		if let Some(uid) = nccl_id {
			let pool_any: &dyn Any = &gpu_pool;

			if let Some(cuda_pool) = pool_any.downcast_ref::<GpuMemoryPool<CudaRuntime, N>>() {
				let dev_count = <CudaRuntime as Runtime>::Device::device_count_total();
				let nccl_init = cubecl_cuda::NcclInit {
					id: dev_id.index_id as i32,
					dev_count: dev_count as i32,
					uid,
				};

				cuda_pool
					.client()
					.init_extention::<cubecl_cuda::NcclExtension>(nccl_init)
					.map_err(|e| {
						crate::operation::error::OpError::ExtensionError(format!("{:?}", e))
					})?;
			}
		}
	}
	while let Ok(command) = rx.recv() {
		match command {
			GpuCommand::CopyTensor { tensor, response } => {
				let result = gpu_pool.allocate_from_tensor(&tensor);
				let resp = match result {
					Ok(handle) => GpuResponse::HandleCreated { dev_id, handle },
					Err(e) => GpuResponse::Error {
						dev_id,
						message: format!("{:?}", e),
					},
				};
				let _ = response.send(resp);
			},
			GpuCommand::CreateEmpty { md, response } => {
				let result = gpu_pool.allocate_empty(&md);
				let resp = match result {
					Ok(handle) => GpuResponse::HandleCreated { dev_id, handle },
					Err(e) => GpuResponse::Error {
						dev_id,
						message: format!("{:?}", e),
					},
				};
				let _ = response.send(resp);
			},
			GpuCommand::GetData { mem_rep, response } => {
				let result = gpu_pool.get_data(&mem_rep);
				let resp = match result {
					Ok(data) => GpuResponse::DataRetrieved { dev_id, data },
					Err(e) => GpuResponse::Error {
						dev_id,
						message: format!("{:?}", e),
					},
				};
				let _ = response.send(resp);
			},
			GpuCommand::ExecuteKernel { executor, response } => {
				let result = executor(&gpu_pool);
				let resp = match result {
					Ok(_) => GpuResponse::Executed { dev_id },
					Err(e) => GpuResponse::Error {
						dev_id,
						message: format!("{:?}", e),
					},
				};
				let _ = response.send(resp);
			},
			GpuCommand::Synchronize { response } => {
				let _ = response.send(GpuResponse::Synchronized { dev_id });
			},
			GpuCommand::DeallocateHandle { mem_rep, response } => {
				let result = gpu_pool.deallocate(&mem_rep);
				let resp = match result {
					Ok(_) => GpuResponse::Deallocated { dev_id },
					Err(e) => GpuResponse::Error {
						dev_id,
						message: format!("{:?}", e),
					},
				};
				let _ = response.send(resp);
			},
			GpuCommand::Shutdown => break,
		}
	}

	Ok(())
}
