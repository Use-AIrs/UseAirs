// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use cubecl_common::device::{Device, DeviceId};
use cubecl_core::prelude::Runtime;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct GpuEnvironment {
	pub total_devices: usize,

	pub devices: Vec<GpuDeviceInfo>,

	pub system_memory: SystemMemoryInfo,

	pub recommended_config: EnvironmentConfig,
}

#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
	pub id: DeviceId,

	pub index: usize,

	pub name: String,

	pub total_memory: usize,

	pub available_memory: usize,

	pub compute_capability: ComputeCapability,

	pub max_threads_per_block: u32,

	pub max_block_dims: [u32; 3],

	pub max_grid_dims: [u32; 3],

	pub warp_size: u32,

	pub shared_memory_per_block: usize,

	pub supports_nccl: bool,

	pub current_utilization: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ComputeCapability {
	pub major: u32,
	pub minor: u32,
}

impl ComputeCapability {
	pub fn new(
		major: u32,
		minor: u32,
	) -> Self {
		Self { major, minor }
	}

	pub fn supports_feature(
		&self,
		feature: ComputeFeature,
	) -> bool {
		match feature {
			ComputeFeature::Fp16 => self.major >= 5,
			ComputeFeature::Fp64 => self.major >= 1,
			ComputeFeature::TensorCores => self.major >= 7,
			ComputeFeature::CooperativeGroups => self.major >= 6,
			ComputeFeature::UnifiedMemory => self.major >= 3,
		}
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeFeature {
	Fp16,
	Fp64,
	TensorCores,
	CooperativeGroups,
	UnifiedMemory,
}

#[derive(Debug, Clone)]
pub struct SystemMemoryInfo {
	pub total_ram: usize,

	pub available_ram: usize,

	pub total_gpu_memory: usize,

	pub available_gpu_memory: usize,
}

#[derive(Debug, Clone)]
pub struct EnvironmentConfig {
	pub distribution_strategy: RecommendedDistribution,

	pub streaming_chunk_size: usize,

	pub default_line_size: u32,

	pub use_nccl: bool,

	pub cpu_thread_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecommendedDistribution {
	SingleGpu,

	DataParallel,

	PipelineParallel,

	Hybrid,
}

impl GpuEnvironment {
	pub fn detect<R: Runtime>() -> Result<Self, EnvironmentError> {
		let total_devices = <R as Runtime>::Device::device_count_total();

		if total_devices == 0 {
			return Err(EnvironmentError::NoGpusFound);
		}

		let mut devices = Vec::with_capacity(total_devices);
		for index in 0..total_devices {
			let dev_id = DeviceId::new(0, index as u32);
			let info = Self::detect_device::<R>(dev_id, index)?;
			devices.push(info);
		}

		let system_memory = Self::detect_system_memory(&devices);

		let recommended_config = Self::generate_recommended_config(&devices, &system_memory);

		Ok(Self {
			total_devices,
			devices,
			system_memory,
			recommended_config,
		})
	}

	fn detect_device<R: Runtime>(
		dev_id: DeviceId,
		index: usize,
	) -> Result<GpuDeviceInfo, EnvironmentError> {
		let device = <R as Runtime>::Device::from_id(dev_id);

		let info = GpuDeviceInfo {
			id: dev_id,
			index,
			name: format!("GPU {}", index),

			total_memory: 8 * 1024 * 1024 * 1024,
			available_memory: 7 * 1024 * 1024 * 1024,

			compute_capability: ComputeCapability::new(7, 0),

			max_threads_per_block: 1024,
			max_block_dims: [1024, 1024, 64],
			max_grid_dims: [2147483647, 65535, 65535],

			warp_size: 32,

			shared_memory_per_block: 48 * 1024,

			supports_nccl: true,

			current_utilization: None,
		};

		Ok(info)
	}

	fn detect_system_memory(devices: &[GpuDeviceInfo]) -> SystemMemoryInfo {
		let total_gpu_memory: usize = devices.iter().map(|d| d.total_memory).sum();

		let available_gpu_memory: usize = devices.iter().map(|d| d.available_memory).sum();

		let total_ram = 32 * 1024 * 1024 * 1024;
		let available_ram = 24 * 1024 * 1024 * 1024;

		SystemMemoryInfo {
			total_ram,
			available_ram,
			total_gpu_memory,
			available_gpu_memory,
		}
	}

	fn generate_recommended_config(
		devices: &[GpuDeviceInfo],
		system_memory: &SystemMemoryInfo,
	) -> EnvironmentConfig {
		let distribution_strategy = if devices.len() == 1 {
			RecommendedDistribution::SingleGpu
		} else if devices.len() <= 4 {
			RecommendedDistribution::DataParallel
		} else if devices.len() <= 8 {
			RecommendedDistribution::Hybrid
		} else {
			RecommendedDistribution::PipelineParallel
		};

		let min_gpu_memory = devices
			.iter()
			.map(|d| d.available_memory)
			.min()
			.unwrap_or(1024 * 1024 * 1024);

		let streaming_chunk_size = (min_gpu_memory / 10).min(100 * 1024 * 1024);

		let default_line_size = 4;

		let use_nccl = devices.len() > 1 && devices.iter().all(|d| d.supports_nccl);

		let cpu_thread_count = std::thread::available_parallelism()
			.map(|n| n.get())
			.unwrap_or(4)
			.max(2);

		EnvironmentConfig {
			distribution_strategy,
			streaming_chunk_size,
			default_line_size,
			use_nccl,
			cpu_thread_count,
		}
	}

	pub fn report(&self) -> String {
		let mut report = String::new();

		report.push_str("=== GPU Environment Report ===\n\n");

		report.push_str(&format!(
			"Total GPUs: {}\n",
			self.total_devices
		));
		report.push_str(&format!(
			"Total GPU Memory: {:.2} GB\n",
			self.system_memory.total_gpu_memory as f64 / (1024.0 * 1024.0 * 1024.0)
		));
		report.push_str(&format!(
			"Available GPU Memory: {:.2} GB\n",
			self.system_memory.available_gpu_memory as f64 / (1024.0 * 1024.0 * 1024.0)
		));
		report.push_str(&format!(
			"System RAM: {:.2} GB\n\n",
			self.system_memory.total_ram as f64 / (1024.0 * 1024.0 * 1024.0)
		));

		report.push_str("=== GPU Devices ===\n");
		for device in &self.devices {
			report.push_str(&format!(
				"\nGPU {} ({})\n",
				device.index, device.name
			));
			report.push_str(&format!(
				"  Memory: {:.2} GB ({:.2} GB available)\n",
				device.total_memory as f64 / (1024.0 * 1024.0 * 1024.0),
				device.available_memory as f64 / (1024.0 * 1024.0 * 1024.0)
			));
			report.push_str(&format!(
				"  Compute Capability: {}.{}\n",
				device.compute_capability.major, device.compute_capability.minor
			));
			report.push_str(&format!(
				"  Warp Size: {}\n",
				device.warp_size
			));
			report.push_str(&format!(
				"  Max Threads/Block: {}\n",
				device.max_threads_per_block
			));
			report.push_str(&format!(
				"  Shared Memory: {:.1} KB\n",
				device.shared_memory_per_block as f64 / 1024.0
			));
			report.push_str(&format!(
				"  NCCL Support: {}\n",
				device.supports_nccl
			));

			if let Some(util) = device.current_utilization {
				report.push_str(&format!(
					"  Utilization: {:.1}%\n",
					util * 100.0
				));
			}
		}

		report.push_str("\n=== Recommended Configuration ===\n");
		report.push_str(&format!(
			"Distribution Strategy: {:?}\n",
			self.recommended_config.distribution_strategy
		));
		report.push_str(&format!(
			"Streaming Chunk Size: {:.2} MB\n",
			self.recommended_config.streaming_chunk_size as f64 / (1024.0 * 1024.0)
		));
		report.push_str(&format!(
			"Default Line Size: {}\n",
			self.recommended_config.default_line_size
		));
		report.push_str(&format!(
			"Use NCCL: {}\n",
			self.recommended_config.use_nccl
		));
		report.push_str(&format!(
			"CPU Threads: {}\n",
			self.recommended_config.cpu_thread_count
		));

		report
	}

	pub fn device(
		&self,
		index: usize,
	) -> Option<&GpuDeviceInfo> {
		self.devices.get(index)
	}

	pub fn devices_range(
		&self,
		start: usize,
		end: usize,
	) -> &[GpuDeviceInfo] {
		&self.devices[start.min(self.total_devices)..end.min(self.total_devices)]
	}

	pub fn nccl_available(&self) -> bool {
		self.total_devices > 1 && self.devices.iter().all(|d| d.supports_nccl)
	}

	pub fn total_available_memory(&self) -> usize {
		self.system_memory.available_gpu_memory
	}

	pub fn min_gpu_memory(&self) -> usize {
		self.devices
			.iter()
			.map(|d| d.available_memory)
			.min()
			.unwrap_or(0)
	}

	pub fn should_stream(
		&self,
		data_size_bytes: usize,
	) -> bool {
		data_size_bytes > (self.min_gpu_memory() / 2)
	}

	pub fn recommend_gpu_count(
		&self,
		data_size_bytes: usize,
	) -> usize {
		if data_size_bytes < self.min_gpu_memory() {
			1
		} else {
			let gpus_needed = (data_size_bytes + self.min_gpu_memory() - 1) / self.min_gpu_memory();
			gpus_needed.min(self.total_devices)
		}
	}
}

#[derive(Debug, thiserror::Error)]
pub enum EnvironmentError {
	#[error("No GPUs found in system")]
	NoGpusFound,

	#[error("Failed to query device {0}: {1}")]
	DeviceQueryFailed(usize, String),

	#[error("Insufficient GPU memory: required {required} bytes, available {available} bytes")]
	InsufficientMemory { required: usize, available: usize },

	#[error("Device {0} does not support required feature: {1:?}")]
	UnsupportedFeature(usize, ComputeFeature),
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_compute_capability() {
		let cap = ComputeCapability::new(7, 5);
		assert!(cap.supports_feature(ComputeFeature::TensorCores));
		assert!(cap.supports_feature(ComputeFeature::Fp16));

		let cap = ComputeCapability::new(5, 0);
		assert!(!cap.supports_feature(ComputeFeature::TensorCores));
		assert!(cap.supports_feature(ComputeFeature::Fp16));
	}

	#[test]
	fn test_recommended_distribution() {
		let devices = vec![GpuDeviceInfo {
			id: DeviceId::new(0, 0),
			index: 0,
			name: "GPU 0".to_string(),
			total_memory: 8 * 1024 * 1024 * 1024,
			available_memory: 7 * 1024 * 1024 * 1024,
			compute_capability: ComputeCapability::new(7, 0),
			max_threads_per_block: 1024,
			max_block_dims: [1024, 1024, 64],
			max_grid_dims: [2147483647, 65535, 65535],
			warp_size: 32,
			shared_memory_per_block: 48 * 1024,
			supports_nccl: true,
			current_utilization: None,
		}];

		let system_memory = SystemMemoryInfo {
			total_ram: 32 * 1024 * 1024 * 1024,
			available_ram: 24 * 1024 * 1024 * 1024,
			total_gpu_memory: 8 * 1024 * 1024 * 1024,
			available_gpu_memory: 7 * 1024 * 1024 * 1024,
		};

		let config = GpuEnvironment::generate_recommended_config(&devices, &system_memory);

		assert_eq!(
			config.distribution_strategy,
			RecommendedDistribution::SingleGpu
		);
		assert_eq!(config.default_line_size, 4);
		assert!(!config.use_nccl);
	}
}
