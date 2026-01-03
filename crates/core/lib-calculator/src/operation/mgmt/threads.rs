// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use super::error::{Result, ThreadError};
use crate::operation::gpu::*;
use crate::{MetaData, Tensor};
use core_affinity;
use crossbeam::channel::{bounded, Sender};
use cubecl_common::device::{Device, DeviceId};
use cubecl_core::prelude::{CubeElement, Numeric, Runtime};
use std::thread;

#[cfg(feature = "nccl")]
use cudarc::nccl::sys::ncclUniqueId;
#[cfg(feature = "nccl")]
use cudarc::nccl::Id as NcclId;

pub struct ManagingThread<R: Runtime, N: Numeric + CubeElement> {
	gpu_channels: Vec<Sender<GpuCommand<R, N>>>,
	worker_handles: Vec<thread::JoinHandle<()>>,
	gpu_mem: GpuMem<N>,
	dev_count: usize,
	#[cfg(feature = "nccl")]
	nccl_id: Option<ncclUniqueId>,
}

impl<R: Runtime, N: Numeric + CubeElement + Clone + 'static> ManagingThread<R, N> {
	pub fn init() -> Result<Self> {
		let dev_count = <R as Runtime>::Device::device_count_total();
		let mut gpu_channels = Vec::new();
		let mut worker_handles = Vec::new();

		// Generate NCCL unique ID for multi-GPU communication
		#[cfg(feature = "nccl")]
		let nccl_id = cudarc::nccl::result::get_uniqueid().unwrap();

		for gpu_rank in 0..dev_count {
			let (tx, rx) = bounded::<GpuCommand<R, N>>(100);
			gpu_channels.push(tx);

			let dev_id = DeviceId::new(0, gpu_rank as u32);

			// Copy NCCL ID for this worker (array copy)
			#[cfg(feature = "nccl")]
			let nccl_id_for_worker = nccl_id;

			let handle = thread::Builder::new()
				.name(format!("gpu-worker-{}", gpu_rank))
				.spawn(move || {
					// Set core affinity - Worker 0 on core 15, Worker 1 on core 31
					let core_ids = core_affinity::get_core_ids().unwrap_or_default();
					if !core_ids.is_empty() {
						let target_core_idx = if gpu_rank == 0 {
							15
						} else if gpu_rank == 1 {
							31
						} else {
							gpu_rank * 16 // Fallback for more GPUs
						};

						if target_core_idx < core_ids.len() {
							let target_core = core_ids[target_core_idx];
							if core_affinity::set_for_current(target_core) {
								eprintln!(
									"✅ GPU worker {} pinned to core {}",
									gpu_rank, target_core_idx
								);
							} else {
								eprintln!(
									"⚠️  GPU worker {} failed to pin to core {}",
									gpu_rank, target_core_idx
								);
							}
						}
					}

					#[cfg(feature = "nccl")]
					let result = gpu_worker::<R, N>(dev_id, rx, Some(nccl_id));

					#[cfg(not(feature = "nccl"))]
					let result = gpu_worker::<R, N>(dev_id, rx, None);

					if let Err(e) = result {
						eprintln!("GPU worker {} failed: {:?}", gpu_rank, e);
					}
				})
				.expect("Failed to spawn GPU worker thread");

			worker_handles.push(handle);
		}

		let gpu_mem = GpuMem::init()?;

		Ok(Self {
			gpu_channels,
			worker_handles,
			gpu_mem,
			dev_count,
			#[cfg(feature = "nccl")]
			nccl_id: Some(nccl_id),
		})
	}

	fn validate_dev_id(
		&self,
		dev_id: DeviceId,
	) -> Result<()> {
		if dev_id.index_id as usize >= self.dev_count {
			return Err(ThreadError::InvalidGpuId {
				id: dev_id.index_id as usize,
				max: self.dev_count - 1,
			});
		}
		Ok(())
	}

	fn send_command(
		&self,
		dev_id: DeviceId,
		command: GpuCommand<R, N>,
	) -> Result<()> {
		self.gpu_channels[dev_id.index_id as usize]
			.send(command)
			.map_err(|_| ThreadError::SendError)
	}

	/// Create empty tensor on single GPU
	pub fn tensor_empty(
		&self,
		dev_id: DeviceId,
		md: &MetaData,
	) -> Result<Interval> {
		self.validate_dev_id(dev_id)?;
		let interval = self.gpu_mem.interval_create()?;

		let (response_tx, response_rx) = bounded(1);
		let command = GpuCommand::CreateEmpty {
			md: md.clone(),
			response: response_tx,
		};

		self.send_command(dev_id, command)?;

		match response_rx.recv()? {
			GpuResponse::HandleCreated { handle, .. } => {
				self.gpu_mem
					.memory_handle_add(interval, dev_id.index_id as usize, handle)?;
				Ok(interval)
			},
			GpuResponse::Error { message, .. } => Err(ThreadError::WorkerError { message }),
			_ => Err(ThreadError::InvalidResponse),
		}
	}

	/// Create empty tensors on all GPUs
	pub fn tensor_empty_broadcast(
		&self,
		md: &MetaData,
	) -> Result<Interval> {
		let interval = self.gpu_mem.interval_create()?;
		let mut responses = Vec::new();

		// Send all commands first
		for dev_k in 0..self.dev_count {
			let (response_tx, response_rx) = bounded(1);
			let command = GpuCommand::CreateEmpty {
				md: md.clone(),
				response: response_tx,
			};
			let dev_id = DeviceId::new(0, dev_k as u32);
			self.send_command(dev_id, command)?;
			responses.push((dev_id, response_rx));
		}

		// Collect all responses
		for (dev_id, rx) in responses {
			match rx.recv()? {
				GpuResponse::HandleCreated { handle, .. } => {
					self.gpu_mem
						.memory_handle_add(interval, dev_id.index_id as usize, handle)?;
				},
				GpuResponse::Error { message, .. } => {
					return Err(ThreadError::WorkerError { message })
				},
				_ => return Err(ThreadError::InvalidResponse),
			}
		}

		Ok(interval)
	}

	/// Send tensor to single GPU
	pub fn tensor_send(
		&self,
		dev_id: DeviceId,
		tensor: &Tensor<N>,
	) -> Result<Interval> {
		self.validate_dev_id(dev_id)?;
		let interval = self.gpu_mem.interval_create()?;

		let (response_tx, response_rx) = bounded(1);
		let command = GpuCommand::CopyTensor {
			tensor: tensor.clone(),
			response: response_tx,
		};

		self.send_command(dev_id, command)?;

		match response_rx.recv()? {
			GpuResponse::HandleCreated { handle, .. } => {
				self.gpu_mem
					.memory_handle_add(interval, dev_id.index_id as usize, handle)?;
				Ok(interval)
			},
			GpuResponse::Error { message, .. } => Err(ThreadError::WorkerError { message }),
			_ => Err(ThreadError::InvalidResponse),
		}
	}

	/// Send different tensors to each GPU (distributed)
	pub fn tensor_send_distributed(
		&self,
		tensors: Vec<Tensor<N>>,
	) -> Result<Interval> {
		if tensors.len() != self.dev_count {
			return Err(ThreadError::TensorSizeMismatch {
				expected: self.dev_count,
				actual: tensors.len(),
			});
		}

		let interval = self.gpu_mem.interval_create()?;
		let mut responses = Vec::new();

		// Send all commands
		for (dev_id, tensor) in tensors.into_iter().enumerate() {
			let (response_tx, response_rx) = bounded(1);
			let command = GpuCommand::CopyTensor {
				tensor,
				response: response_tx,
			};
			let dev = DeviceId::new(0, dev_id as u32);
			self.send_command(dev, command)?;
			responses.push((dev_id, response_rx));
		}

		// Collect responses
		for (dev_id, rx) in responses {
			match rx.recv()? {
				GpuResponse::HandleCreated { handle, .. } => {
					self.gpu_mem.memory_handle_add(interval, dev_id, handle)?;
				},
				GpuResponse::Error { message, .. } => {
					return Err(ThreadError::WorkerError { message })
				},
				_ => return Err(ThreadError::InvalidResponse),
			}
		}

		Ok(interval)
	}

	/// Send same tensor to all GPUs (broadcast)
	pub fn tensor_send_broadcast(
		&self,
		tensor: Tensor<N>,
	) -> Result<Interval> {
		let interval = self.gpu_mem.interval_create()?;
		let mut responses = Vec::new();

		// Send all commands
		for dev_id in 0..self.dev_count {
			let (response_tx, response_rx) = bounded(1);
			let command = GpuCommand::CopyTensor {
				tensor: tensor.clone(),
				response: response_tx,
			};
			let dev = DeviceId::new(0, dev_id as u32);
			self.send_command(dev, command)?;
			responses.push((dev_id, response_rx));
		}

		// Collect responses
		for (dev_id, rx) in responses {
			match rx.recv()? {
				GpuResponse::HandleCreated { handle, .. } => {
					self.gpu_mem.memory_handle_add(interval, dev_id, handle)?;
				},
				GpuResponse::Error { message, .. } => {
					return Err(ThreadError::WorkerError { message })
				},
				_ => return Err(ThreadError::InvalidResponse),
			}
		}

		Ok(interval)
	}

	/// Execute kernel on single GPU using intervals
	pub fn exec_kernel_on_gpu<K, I, O>(
		&self,
		dev_id: DeviceId,
		input_intervals: I,
		output_intervals: O,
		kernel: K,
		config: K::Cfg,
	) -> Result<()>
	where
		K: Kernel<R, N>,
		I: IntervalTuple<N>,
		O: IntervalTuple<N>,
		I::Output: ExtractMemHandle<K::Input>,
		O::Output: ExtractMemHandle<K::Output>,
	{
		self.validate_dev_id(dev_id)?;

		// Use GpuMem for interval mapping
		let input_map = input_intervals.interval_map(&self.gpu_mem)?;
		let output_map = output_intervals.interval_map(&self.gpu_mem)?;

		// Extract handles for this GPU
		let input = input_map.extract_for_gpu(dev_id.index_id as usize)?;
		let output = output_map.extract_for_gpu(dev_id.index_id as usize)?;

		let order = KernelOrder::order(kernel, config, input, output);

		let (response_tx, response_rx) = bounded(1);
		let command = GpuCommand::ExecuteKernel {
			executor: Box::new(move |pool| order.op.exec(&order, pool)),
			response: response_tx,
		};

		self.send_command(dev_id, command)?;

		match response_rx.recv()? {
			GpuResponse::Executed { .. } => Ok(()),
			GpuResponse::Error { message, .. } => Err(ThreadError::WorkerError { message }),
			_ => Err(ThreadError::InvalidResponse),
		}
	}

	/// Execute kernel on all GPUs using intervals (parallel)
	pub fn exec_kernel_broadcast<K, I, O>(
		&self,
		input_intervals: &I,
		output_intervals: &O,
		kernel: K,
		config: K::Cfg,
	) -> Result<()>
	where
		K: Kernel<R, N> + Clone,
		K::Cfg: Clone,
		I: IntervalTuple<N>,
		O: IntervalTuple<N>,
		I::Output: ExtractMemHandle<K::Input>,
		O::Output: ExtractMemHandle<K::Output>,
	{
		// Get mapped handles using GpuMem
		let input_map = input_intervals.interval_map(&self.gpu_mem)?;
		let output_map = output_intervals.interval_map(&self.gpu_mem)?;

		let mut responses = Vec::new();

		// Send all kernel executions in parallel
		for dev_id in 0..self.dev_count {
			let dev = DeviceId::new(0, dev_id as u32);
			// Try to extract handles for this GPU
			match (
				input_map.extract_for_gpu(dev_id),
				output_map.extract_for_gpu(dev_id),
			) {
				(Ok(input), Ok(output)) => {
					let order = KernelOrder::order(
						kernel.clone(),
						config.clone(),
						input,
						output,
					);

					let (response_tx, response_rx) = bounded(1);
					let command = GpuCommand::ExecuteKernel {
						executor: Box::new(move |pool| order.op.exec(&order, pool)),
						response: response_tx,
					};

					self.send_command(dev, command)?;
					responses.push((dev_id, response_rx));
				},
				_ => {
					// Skip GPUs that don't have the required handles
					continue;
				},
			}
		}

		for (dev_id, rx) in responses {
			match rx.recv()? {
				GpuResponse::Executed { .. } => {},
				GpuResponse::Error { message, .. } => {
					return Err(ThreadError::WorkerError {
						message: format!("GPU {} failed: {}", dev_id, message),
					})
				},
				_ => return Err(ThreadError::InvalidResponse),
			}
		}

		Ok(())
	}

	pub fn exec_kernel_on_gpus<K, I, O>(
		&self,
		dev_ids: &[usize],
		input_intervals: I,
		output_intervals: O,
		kernel: K,
		config: K::Cfg,
	) -> Result<()>
	where
		K: Kernel<R, N> + Clone,
		K::Cfg: Clone,
		I: IntervalTuple<N>,
		O: IntervalTuple<N>,
		I::Output: ExtractMemHandle<K::Input>,
		O::Output: ExtractMemHandle<K::Output>,
	{
		let mut devs = Vec::new();

		for &dev_id in dev_ids {
			let dev = DeviceId::new(0, dev_id as u32);
			devs.push(dev);
			self.validate_dev_id(dev)?;
		}

		let input_map = input_intervals.interval_map(&self.gpu_mem)?;
		let output_map = output_intervals.interval_map(&self.gpu_mem)?;

		let mut responses = Vec::new();

		for dev_id in devs {
			let input = input_map.extract_for_gpu(dev_id.index_id as usize)?;
			let output = output_map.extract_for_gpu(dev_id.index_id as usize)?;

			let order = KernelOrder::order(
				kernel.clone(),
				config.clone(),
				input,
				output,
			);

			let (response_tx, response_rx) = bounded(1);
			let command = GpuCommand::ExecuteKernel {
				executor: Box::new(move |pool| order.op.exec(&order, pool)),
				response: response_tx,
			};

			self.send_command(dev_id, command)?;
			responses.push((dev_id, response_rx));
		}

		for (dev_id, rx) in responses {
			match rx.recv()? {
				GpuResponse::Executed { .. } => {},
				GpuResponse::Error { message, .. } => {
					return Err(ThreadError::WorkerError {
						message: format!("GPU {} failed: {}", dev_id, message),
					})
				},
				_ => return Err(ThreadError::InvalidResponse),
			}
		}

		Ok(())
	}

	pub fn tensor_get(
		&self,
		interval: &Interval,
		dev_id: DeviceId,
	) -> Result<Tensor<N>> {
		self.validate_dev_id(dev_id)?;

		let mem_rep = self
			.gpu_mem
			.memory_handle_specific(*interval, dev_id.index_id as usize)?;

		let (response_tx, response_rx) = bounded(1);
		let command = GpuCommand::GetData {
			mem_rep: mem_rep.clone(),
			response: response_tx,
		};

		self.send_command(dev_id, command)?;

		match response_rx.recv()? {
			GpuResponse::DataRetrieved { data, .. } => Ok(Tensor::new(
				data,
				mem_rep.metadata.clone(),
			)),
			GpuResponse::Error { message, .. } => Err(ThreadError::WorkerError { message }),
			_ => Err(ThreadError::InvalidResponse),
		}
	}

	pub fn tensors_get_all(
		&self,
		interval: &Interval,
	) -> Result<Vec<Tensor<N>>> {
		let handles = self.gpu_mem.memory_handles_interval(interval)?;
		let mut responses = Vec::new();

		for (&dev_id, mem_rep) in handles.iter() {
			let (response_tx, response_rx) = bounded(1);
			let command = GpuCommand::GetData {
				mem_rep: mem_rep.clone(),
				response: response_tx,
			};

			self.send_command(DeviceId::new(0, dev_id as u32), command)?;
			responses.push((
				dev_id,
				response_rx,
				mem_rep.metadata.clone(),
			));
		}

		let mut tensors = Vec::with_capacity(responses.len());
		for (dev_id, rx, metadata) in responses {
			match rx.recv()? {
				GpuResponse::DataRetrieved { data, .. } => {
					tensors.push(Tensor::new(data, metadata));
				},
				GpuResponse::Error { message, .. } => {
					return Err(ThreadError::WorkerError {
						message: format!("GPU {} failed: {}", dev_id, message),
					})
				},
				_ => return Err(ThreadError::InvalidResponse),
			}
		}

		Ok(tensors)
	}

	pub fn tensors_get_from_gpus(
		&self,
		interval: &Interval,
		dev_ids: &[usize],
	) -> Result<Vec<Tensor<N>>> {
		let mut tensors = Vec::new();
		let mut responses = Vec::new();

		for dev_id in dev_ids {
			let dev = DeviceId::new(0, dev_id.clone() as u32);
			self.validate_dev_id(dev)?;
			let mem_rep = self
				.gpu_mem
				.memory_handle_specific(*interval, dev_id.clone())?;

			let (response_tx, response_rx) = bounded(1);
			let command = GpuCommand::GetData {
				mem_rep: mem_rep.clone(),
				response: response_tx,
			};

			self.send_command(dev, command)?;
			responses.push((
				dev_id,
				response_rx,
				mem_rep.metadata.clone(),
			));
		}

		for (dev_id, rx, metadata) in responses {
			match rx.recv()? {
				GpuResponse::DataRetrieved { data, .. } => {
					tensors.push(Tensor::new(data, metadata));
				},
				GpuResponse::Error { message, .. } => {
					return Err(ThreadError::WorkerError {
						message: format!("GPU {} failed: {}", dev_id, message),
					})
				},
				_ => return Err(ThreadError::InvalidResponse),
			}
		}

		Ok(tensors)
	}

	pub fn interval_remove(
		&self,
		interval: Interval,
	) -> Result<()> {
		let handles = self.gpu_mem.interval_get_handles(interval)?;

		for (dev_id, mem_rep) in handles {
			let (tx, rx) = bounded(1);
			let command = GpuCommand::DeallocateHandle {
				mem_rep,
				response: tx,
			};

			self.send_command(dev_id, command)?;

			match rx.recv()? {
				GpuResponse::Deallocated { .. } => {},
				GpuResponse::Error { message, .. } => {
					return Err(ThreadError::WorkerError {
						message: format!(
							"Failed to deallocate on GPU {}: {}",
							dev_id.index_id, message
						),
					})
				},
				_ => return Err(ThreadError::InvalidResponse),
			}
		}

		self.gpu_mem.interval_remove(interval)?;
		Ok(())
	}

	pub fn interval_exists(
		&self,
		interval: Interval,
	) -> Result<bool> {
		let res = self.gpu_mem.interval_exists(interval)?;
		Ok(res)
	}

	pub fn intervals(&self) -> Result<Vec<Interval>> {
		let res = self.gpu_mem.intervals()?;
		Ok(res)
	}

	pub fn interval_size(
		&self,
		interval: Interval,
	) -> Result<usize> {
		let res = self.gpu_mem.interval_size(interval)?;
		Ok(res)
	}

	pub fn gpu_sync(
		&self,
		dev_id: DeviceId,
	) -> Result<()> {
		self.validate_dev_id(dev_id)?;

		let (response_tx, response_rx) = bounded(1);
		let command = GpuCommand::Synchronize {
			response: response_tx,
		};

		self.send_command(dev_id, command)?;

		match response_rx.recv()? {
			GpuResponse::Synchronized { .. } => Ok(()),
			GpuResponse::Error { message, .. } => Err(ThreadError::WorkerError { message }),
			_ => Err(ThreadError::InvalidResponse),
		}
	}

	pub fn gpu_sync_all(&self) -> Result<()> {
		let mut responses = Vec::new();

		for dev_id in 0..self.dev_count {
			let (response_tx, response_rx) = bounded(1);
			let command = GpuCommand::Synchronize {
				response: response_tx,
			};

			self.send_command(DeviceId::new(0, dev_id as u32), command)?;
			responses.push((dev_id, response_rx));
		}

		for (dev_id, rx) in responses {
			match rx.recv()? {
				GpuResponse::Synchronized { .. } => {},
				GpuResponse::Error { message, .. } => {
					return Err(ThreadError::WorkerError {
						message: format!("GPU {} sync failed: {}", dev_id, message),
					})
				},
				_ => return Err(ThreadError::InvalidResponse),
			}
		}

		Ok(())
	}

	pub fn dev_count(&self) -> usize {
		self.dev_count
	}

	pub fn shutdown(self) -> Result<()> {
		for tx in self.gpu_channels.iter() {
			let _ = tx.send(GpuCommand::Shutdown);
		}

		for handle in self.worker_handles {
			handle.join().map_err(|_| ThreadError::WorkerPanic)?;
		}

		Ok(())
	}
}
