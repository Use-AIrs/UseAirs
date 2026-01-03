// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use super::config::ManagerConfig;
use super::environment::{ComputeCapability, EnvironmentError, GpuDeviceInfo, GpuEnvironment};
use super::error::{Result, ThreadError};
use super::groups::{GpuGroup, GpuGroupManager, GroupStrategy};
use super::threads::ManagingThread;
use crate::operation::gpu::*;
use crate::{MetaData, Tensor};

use cubecl_common::device::{Device, DeviceId};
use cubecl_core::client::ComputeClient;
use cubecl_core::prelude::{CubeElement, Numeric, Runtime};
use std::collections::HashMap;
use std::marker::PhantomData;

pub struct GpuCoordinator<R: Runtime, N: Numeric + CubeElement> {
	thread: ManagingThread<R, N>,

	groups: GpuGroupManager,

	environment: GpuEnvironment,

	device_properties: HashMap<usize, DeviceHardwareInfo>,

	total_gpus: usize,

	_phantom: PhantomData<(R, N)>,
}

#[derive(Debug, Clone)]
pub struct DeviceHardwareInfo {
	pub index: usize,

	pub plane_size_min: u32,
	pub plane_size_max: u32,

	pub max_bindings: u32,

	pub max_shared_memory_size: usize,

	pub max_cube_count_x: u32,
	pub max_cube_count_y: u32,
	pub max_cube_count_z: u32,

	pub max_units_per_cube: u32,

	pub max_cube_dim_x: u32,
	pub max_cube_dim_y: u32,
	pub max_cube_dim_z: u32,

	pub num_streaming_multiprocessors: Option<u32>,

	pub num_tensor_cores: Option<u32>,

	pub min_tensor_cores_dim: Option<u32>,

	pub memory_alignment: u64,

	pub max_page_size: u64,
}

impl<R: Runtime, N: Numeric + CubeElement + Clone + 'static> GpuCoordinator<R, N> {
	/// Create coordinator builder
	pub fn builder() -> GpuCoordinatorBuilder<R, N> {
		GpuCoordinatorBuilder::new()
	}

	/// Initialize with detected environment and default configuration
	pub fn init() -> Result<Self> {
		Self::builder().detect_environment().build()
	}

	/// Initialize with custom configuration
	pub fn with_config(config: ManagerConfig) -> Result<Self> {
		Self::builder().from_config(config).build()
	}

	/// Get environment information
	pub fn environment(&self) -> &GpuEnvironment {
		&self.environment
	}

	/// Get group manager
	pub fn groups(&self) -> &GpuGroupManager {
		&self.groups
	}

	/// Get mutable group manager
	pub fn groups_mut(&mut self) -> &mut GpuGroupManager {
		&mut self.groups
	}

	/// Get managing thread
	pub fn thread(&self) -> &ManagingThread<R, N> {
		&self.thread
	}

	/// Get mutable managing thread
	pub fn thread_mut(&mut self) -> &mut ManagingThread<R, N> {
		&mut self.thread
	}

	/// Get hardware info for a specific GPU
	pub fn device_info(
		&self,
		gpu_index: usize,
	) -> Option<&DeviceHardwareInfo> {
		self.device_properties.get(&gpu_index)
	}

	/// Get all device hardware info
	pub fn all_device_info(&self) -> &HashMap<usize, DeviceHardwareInfo> {
		&self.device_properties
	}

	/// Execute operation on a specific GPU group
	///
	/// # Example
	/// ```ignore
	/// coordinator.exec_on_group("training", |thread, gpu_indices| {
	///     // Send data to GPUs in training group
	///     let interval = thread.tensor_send_broadcast(data)?;
	///
	///     // Execute kernel on all GPUs in group
	///     thread.exec_kernel_on_gpus(gpu_indices, ...)?;
	///     Ok(())
	/// })?;
	/// ```
	pub fn exec_on_group<F, T>(
		&self,
		group_name: &str,
		f: F,
	) -> Result<T>
	where
		F: FnOnce(&ManagingThread<R, N>, &[usize]) -> Result<T>,
	{
		let group = self
			.groups
			.group(group_name)
			.ok_or_else(|| ThreadError::GroupNotFound {
				name: group_name.to_string(),
			})?;

		if !group.enabled {
			return Err(ThreadError::WorkerError {
				message: format!("Group '{}' is disabled", group_name),
			});
		}

		f(&self.thread, &group.gpu_indices)
	}

	/// Execute operation on default group
	pub fn exec_on_default<F, T>(
		&self,
		f: F,
	) -> Result<T>
	where
		F: FnOnce(&ManagingThread<R, N>, &[usize]) -> Result<T>,
	{
		let group = self
			.groups
			.default_group()
			.ok_or_else(|| ThreadError::WorkerError {
				message: "No default group configured".to_string(),
			})?;

		if !group.enabled {
			return Err(ThreadError::WorkerError {
				message: "Default group is disabled".to_string(),
			});
		}

		f(&self.thread, &group.gpu_indices)
	}

	/// Execute kernel on specific group with automatic strategy application
	///
	/// This method respects the group's configured strategy:

	pub fn exec_kernel_on_group<K, I, O>(
		&self,
		group_name: &str,
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
		let group = self
			.groups
			.group(group_name)
			.ok_or_else(|| ThreadError::GroupNotFound {
				name: group_name.to_string(),
			})?;

		if !group.enabled {
			return Err(ThreadError::WorkerError {
				message: format!("Group '{}' is disabled", group_name),
			});
		}

		match group.strategy {
			GroupStrategy::SingleGpu => {
				if let Some(&first_gpu) = group.gpu_indices.first() {
					let dev_id = DeviceId::new(0, first_gpu as u32);
					self.thread.exec_kernel_on_gpu(
						dev_id,
						input_intervals,
						output_intervals,
						kernel,
						config,
					)
				} else {
					Err(ThreadError::WorkerError {
						message: format!("Group '{}' has no GPUs", group_name),
					})
				}
			},

			GroupStrategy::DataParallel | GroupStrategy::TensorParallel => {
				self.thread.exec_kernel_on_gpus(
					&group.gpu_indices,
					input_intervals,
					output_intervals,
					kernel,
					config,
				)
			},

			GroupStrategy::PipelineParallel => self.thread.exec_kernel_on_gpus(
				&group.gpu_indices,
				input_intervals,
				output_intervals,
				kernel,
				config,
			),

			GroupStrategy::RoundRobin => {
				if let Some(&first_gpu) = group.gpu_indices.first() {
					let dev_id = DeviceId::new(0, first_gpu as u32);
					self.thread.exec_kernel_on_gpu(
						dev_id,
						input_intervals,
						output_intervals,
						kernel,
						config,
					)
				} else {
					Err(ThreadError::WorkerError {
						message: format!("Group '{}' has no GPUs", group_name),
					})
				}
			},

			GroupStrategy::Custom => Err(ThreadError::WorkerError {
				message: "Custom strategies must be executed manually via exec_on_group"
					.to_string(),
			}),
		}
	}

	pub fn group_optimal_line_size(
		&self,
		group_name: &str,
	) -> Result<u32> {
		let group = self
			.groups
			.group(group_name)
			.ok_or_else(|| ThreadError::GroupNotFound {
				name: group_name.to_string(),
			})?;

		let min_plane_size = group
			.gpu_indices
			.iter()
			.filter_map(|&idx| self.device_properties.get(&idx))
			.map(|info| info.plane_size_min)
			.min()
			.unwrap_or(32);

		Ok(min_plane_size)
	}

	pub fn group_supports_tensor_cores(
		&self,
		group_name: &str,
	) -> Result<bool> {
		let group = self
			.groups
			.group(group_name)
			.ok_or_else(|| ThreadError::GroupNotFound {
				name: group_name.to_string(),
			})?;

		let all_support = group
			.gpu_indices
			.iter()
			.filter_map(|&idx| self.device_properties.get(&idx))
			.all(|info| info.num_tensor_cores.is_some() && info.num_tensor_cores.unwrap() > 0);

		Ok(all_support)
	}

	pub fn group_min_shared_memory(
		&self,
		group_name: &str,
	) -> Result<usize> {
		let group = self
			.groups
			.group(group_name)
			.ok_or_else(|| ThreadError::GroupNotFound {
				name: group_name.to_string(),
			})?;

		let min_shared = group
			.gpu_indices
			.iter()
			.filter_map(|&idx| self.device_properties.get(&idx))
			.map(|info| info.max_shared_memory_size)
			.min()
			.unwrap_or(48 * 1024);

		Ok(min_shared)
	}

	pub fn report(&self) -> String {
		let mut report = String::new();

		report.push_str("╔═══════════════════════════════════════════════════════════════╗\n");
		report.push_str("║           GPU COORDINATOR SYSTEM REPORT                       ║\n");
		report.push_str("╚═══════════════════════════════════════════════════════════════╝\n\n");

		report.push_str(&self.environment.report());
		report.push_str("\n");

		report.push_str("╔═══════════════════════════════════════════════════════════════╗\n");
		report.push_str("║           HARDWARE PROPERTIES (from cubecl)                   ║\n");
		report.push_str("╚═══════════════════════════════════════════════════════════════╝\n\n");

		for gpu_idx in 0..self.total_gpus {
			if let Some(hw_info) = self.device_properties.get(&gpu_idx) {
				report.push_str(&format!("GPU {} Hardware:\n", gpu_idx));
				report.push_str(&format!(
					"  Plane Size: {} - {}\n",
					hw_info.plane_size_min, hw_info.plane_size_max
				));
				report.push_str(&format!(
					"  Max Bindings: {}\n",
					hw_info.max_bindings
				));
				report.push_str(&format!(
					"  Shared Memory: {:.1} KB\n",
					hw_info.max_shared_memory_size as f64 / 1024.0
				));
				report.push_str(&format!(
					"  Max Threads/Block: {}\n",
					hw_info.max_units_per_cube
				));
				report.push_str(&format!(
					"  Max Grid: ({}, {}, {})\n",
					hw_info.max_cube_count_x, hw_info.max_cube_count_y, hw_info.max_cube_count_z
				));

				if let Some(sms) = hw_info.num_streaming_multiprocessors {
					report.push_str(&format!("  SMs/CUs: {}\n", sms));
				}

				if let Some(tc) = hw_info.num_tensor_cores {
					report.push_str(&format!("  Tensor Cores/SM: {}\n", tc));
				}

				report.push_str("\n");
			}
		}

		report.push_str(&self.groups.report());

		report
	}

	pub fn shutdown(self) -> Result<()> {
		self.thread.shutdown()
	}
}

pub struct GpuCoordinatorBuilder<R: Runtime, N: Numeric + CubeElement> {
	config: ManagerConfig,
	_phantom: PhantomData<(R, N)>,
}

impl<R: Runtime, N: Numeric + CubeElement + Clone + 'static> GpuCoordinatorBuilder<R, N> {
	fn new() -> Self {
		Self {
			config: ManagerConfig::builder(),
			_phantom: PhantomData,
		}
	}

	/// Detect GPU environment automatically
	pub fn detect_environment(mut self) -> Self {
		self.config = self.config.detect_environment::<R>();
		self
	}

	/// Set GPU count manually
	pub fn with_gpu_count(
		mut self,
		count: usize,
	) -> Self {
		self.config = self.config.with_gpu_count(count);
		self
	}

	/// Add a GPU group
	pub fn with_group(
		mut self,
		name: &str,
		gpu_indices: &[usize],
		strategy: GroupStrategy,
	) -> Self {
		self.config = self.config.with_group(name, gpu_indices, strategy);
		self
	}

	/// Set default group
	pub fn with_default_group(
		mut self,
		name: &str,
	) -> Self {
		self.config = self.config.with_default_group(name);
		self
	}

	/// Enable environment report printing
	pub fn print_report(mut self) -> Self {
		self.config = self.config.print_environment_report();
		self
	}

	/// Build from existing config
	pub fn from_config(
		mut self,
		config: ManagerConfig,
	) -> Self {
		self.config = config;
		self
	}

	/// Build the coordinator
	pub fn build(self) -> Result<GpuCoordinator<R, N>> {
		// Detect environment if not already done
		let environment = if self.config.environment().is_some() {
			self.config.environment().unwrap().clone()
		} else {
			GpuEnvironment::detect::<R>().map_err(|e| ThreadError::WorkerError {
				message: format!("Environment detection failed: {}", e),
			})?
		};

		let total_gpus = environment.total_devices;

		// Query hardware properties for each GPU
		let mut device_properties = HashMap::new();
		for gpu_idx in 0..total_gpus {
			let hw_info = query_device_hardware::<R>(gpu_idx)?;
			device_properties.insert(gpu_idx, hw_info);
		}

		// Initialize managing thread (launches workers)
		let thread = ManagingThread::<R, N>::init()?;

		// Get group manager from config
		let groups = self.config.group_manager().clone();

		let coordinator = GpuCoordinator {
			thread,
			groups,
			environment,
			device_properties,
			total_gpus,
			_phantom: PhantomData,
		};

		Ok(coordinator)
	}
}

/// Query hardware properties from a specific GPU using cubecl
fn query_device_hardware<R: Runtime>(gpu_index: usize) -> Result<DeviceHardwareInfo> {
	let dev_id = DeviceId::new(0, gpu_index as u32);
	let device = <R as Runtime>::Device::from_id(dev_id);
	let client = R::client(&device);

	// Get properties from the client
	let props = client.properties();

	// Extract max_cube_count dimensions (CubeCount is an enum)
	let (max_cube_count_x, max_cube_count_y, max_cube_count_z) = R::max_cube_count();

	// Extract max_cube_dim dimensions
	let max_cube_dim = props.hardware.max_cube_dim;

	let hw_info = DeviceHardwareInfo {
		index: gpu_index,
		plane_size_min: props.hardware.plane_size_min,
		plane_size_max: props.hardware.plane_size_max,
		max_bindings: props.hardware.max_bindings,
		max_shared_memory_size: props.hardware.max_shared_memory_size,
		max_cube_count_x,
		max_cube_count_y,
		max_cube_count_z,
		max_units_per_cube: props.hardware.max_units_per_cube,
		max_cube_dim_x: max_cube_dim.x,
		max_cube_dim_y: max_cube_dim.y,
		max_cube_dim_z: max_cube_dim.z,
		num_streaming_multiprocessors: props.hardware.num_streaming_multiprocessors,
		num_tensor_cores: props.hardware.num_tensor_cores,
		min_tensor_cores_dim: props.hardware.min_tensor_cores_dim,
		memory_alignment: props.memory.alignment,
		max_page_size: props.memory.max_page_size,
	};

	Ok(hw_info)
}

// ============================================================================
// Future Enhancement: Group-Level Threading
// ============================================================================
//
// TODO: Implement per-group thread management
//
// Architecture:
// - Each GpuGroup gets its own thread pool
// - Groups can autonomously decide strategies
// - Inter-group communication via message passing
// - Load balancing between groups
//
// Benefits:
// - True isolation between workloads
// - Independent failure handling
// - Dynamic strategy adaptation
// - Better resource utilization
//
// Example structure:
// ```
// pub struct GroupThread<R: Runtime, N: Numeric + CubeElement> {
//     group: GpuGroup,
//     worker_channels: Vec<Sender<GpuCommand<R, N>>>,
//     strategy_selector: Box<dyn StrategySelector>,
//     load_balancer: GroupLoadBalancer,
// }
//
// impl GroupThread {
//     pub fn auto_select_strategy(&mut self, workload: &Workload) {
//         // Autonomously decide best strategy based on:
//         // - Workload characteristics
//         // - Current GPU utilization
//         // - Memory availability
//         // - Historical performance
//     }
//
//     pub fn execute(&self, task: Task) -> Result<()> {
//         let strategy = self.strategy_selector.select(&task);
//         match strategy {
//             GroupStrategy::DataParallel => self.exec_data_parallel(task),
//             GroupStrategy::PipelineParallel => self.exec_pipeline(task),
//             // ...
//         }
//     }
// }
// ```

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_device_hardware_info_structure() {
		// Test that the structure can be created
		let info = DeviceHardwareInfo {
			index: 0,
			plane_size_min: 32,
			plane_size_max: 32,
			max_bindings: 16,
			max_shared_memory_size: 48 * 1024,
			max_cube_count_x: 2147483647,
			max_cube_count_y: 65535,
			max_cube_count_z: 65535,
			max_units_per_cube: 1024,
			max_cube_dim_x: 1024,
			max_cube_dim_y: 1024,
			max_cube_dim_z: 64,
			num_streaming_multiprocessors: Some(84),
			num_tensor_cores: Some(4),
			min_tensor_cores_dim: Some(16),
			memory_alignment: 256,
			max_page_size: 4096,
		};

		assert_eq!(info.index, 0);
		assert_eq!(info.plane_size_min, 32);
		assert!(info.num_streaming_multiprocessors.is_some());
	}
}
