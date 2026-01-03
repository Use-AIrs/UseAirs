// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use super::environment::{EnvironmentError, GpuEnvironment};
use super::error::{Result, ThreadError};
use super::groups::{GpuGroupManager, GroupStrategy};
use super::threads::ManagingThread;
use cubecl_common::device::Device;
use cubecl_core::prelude::{CubeElement, Numeric, Runtime};
use std::marker::PhantomData;

#[derive(Debug, Clone)]
pub struct ManagerConfig {
	environment: Option<GpuEnvironment>,

	group_manager: GpuGroupManager,

	auto_detect_env: bool,

	print_env_report: bool,

	validate_on_build: bool,

	total_gpus: Option<usize>,
}

impl Default for ManagerConfig {
	fn default() -> Self {
		Self {
			environment: None,
			group_manager: GpuGroupManager::new(0),
			auto_detect_env: true,
			print_env_report: false,
			validate_on_build: true,
			total_gpus: None,
		}
	}
}

impl ManagerConfig {
	pub fn builder() -> Self {
		Self::default()
	}

	pub fn detect_environment<R: Runtime>(mut self) -> Self {
		match GpuEnvironment::detect::<R>() {
			Ok(env) => {
				self.total_gpus = Some(env.total_devices);
				self.group_manager = GpuGroupManager::from_environment(&env);
				self.environment = Some(env);
			},
			Err(e) => {
				eprintln!(
					"Warning: Failed to detect environment: {}",
					e
				);

				self.total_gpus = Some(<R as Runtime>::Device::device_count_total());
				self.group_manager = GpuGroupManager::new(self.total_gpus.unwrap_or(0));
			},
		}
		self
	}

	pub fn with_gpu_count(
		mut self,
		count: usize,
	) -> Self {
		self.total_gpus = Some(count);
		self.group_manager = GpuGroupManager::new(count);
		self.auto_detect_env = false;
		self
	}

	pub fn print_environment_report(mut self) -> Self {
		self.print_env_report = true;
		self
	}

	pub fn with_group(
		mut self,
		name: &str,
		gpu_indices: &[usize],
		strategy: GroupStrategy,
	) -> Self {
		if let Err(e) = self
			.group_manager
			.create_group(name, gpu_indices.to_vec(), strategy, 5)
		{
			eprintln!(
				"Warning: Failed to create group '{}': {}",
				name, e
			);
		}
		self
	}

	pub fn with_group_priority(
		mut self,
		name: &str,
		gpu_indices: &[usize],
		strategy: GroupStrategy,
		priority: u8,
	) -> Self {
		if let Err(e) = self.group_manager.create_group(
			name,
			gpu_indices.to_vec(),
			strategy,
			priority,
		) {
			eprintln!(
				"Warning: Failed to create group '{}': {}",
				name, e
			);
		}
		self
	}

	pub fn with_default_group(
		mut self,
		group_name: &str,
	) -> Self {
		if let Err(e) = self.group_manager.set_default_group(group_name) {
			eprintln!(
				"Warning: Failed to set default group: {}",
				e
			);
		}
		self
	}

	pub fn with_group_tag(
		mut self,
		group_name: &str,
		tag: &str,
	) -> Self {
		if let Err(e) = self.group_manager.add_group_tag(group_name, tag) {
			eprintln!(
				"Warning: Failed to add tag to group: {}",
				e
			);
		}
		self
	}

	pub fn skip_validation(mut self) -> Self {
		self.validate_on_build = false;
		self
	}

	pub fn auto_detect(
		mut self,
		enable: bool,
	) -> Self {
		self.auto_detect_env = enable;
		self
	}

	pub fn environment(&self) -> Option<&GpuEnvironment> {
		self.environment.as_ref()
	}

	pub fn group_manager(&self) -> &GpuGroupManager {
		&self.group_manager
	}

	pub fn group_manager_mut(&mut self) -> &mut GpuGroupManager {
		&mut self.group_manager
	}

	pub fn validate(&self) -> Result<()> {
		let gpu_count = self
			.total_gpus
			.ok_or_else(|| ThreadError::GpuMemCreationError)?;
		if gpu_count == 0 {
			return Err(ThreadError::InvalidGpuId { id: 0, max: 0 });
		}

		self.group_manager.validate()?;

		if self.group_manager.group_names().is_empty() {
			eprintln!("Warning: No groups defined, creating default group");
		}

		Ok(())
	}

	pub fn build<R: Runtime, N: Numeric + CubeElement + Clone + 'static>(
		self
	) -> Result<ConfiguredManager<R, N>> {
		// Validate if enabled
		if self.validate_on_build {
			self.validate()?;
		}

		// Print environment report if requested
		if self.print_env_report {
			if let Some(env) = &self.environment {
				println!("{}", env.report());
				println!("{}", self.group_manager.report());
			} else {
				println!("Environment not detected. Using manual configuration.");
				println!("{}", self.group_manager.report());
			}
		}

		// Initialize managing thread
		let managing_thread = ManagingThread::<R, N>::init()?;

		Ok(ConfiguredManager {
			thread: managing_thread,
			group_manager: self.group_manager,
			environment: self.environment,
		})
	}
}

/// Manager with applied configuration
///
/// This is the result of building a ManagerConfig.
/// It combines the ManagingThread with group management capabilities.
pub struct ConfiguredManager<R: Runtime, N: Numeric + CubeElement> {
	/// The managing thread handling GPU workers
	pub thread: ManagingThread<R, N>,

	/// GPU group manager
	pub group_manager: GpuGroupManager,

	/// Environment information (if detected)
	pub environment: Option<GpuEnvironment>,
}

impl<R: Runtime, N: Numeric + CubeElement + Clone + 'static> ConfiguredManager<R, N> {
	pub fn thread(&self) -> &ManagingThread<R, N> {
		&self.thread
	}

	pub fn thread_mut(&mut self) -> &mut ManagingThread<R, N> {
		&mut self.thread
	}

	pub fn groups(&self) -> &GpuGroupManager {
		&self.group_manager
	}

	pub fn groups_mut(&mut self) -> &mut GpuGroupManager {
		&mut self.group_manager
	}

	pub fn environment(&self) -> Option<&GpuEnvironment> {
		self.environment.as_ref()
	}

	pub fn exec_on_group<F, T>(
		&mut self,
		group_name: &str,
		f: F,
	) -> Result<T>
	where
		F: FnOnce(&mut ManagingThread<R, N>, &[usize]) -> Result<T>,
	{
		let gpu_indices = self.group_manager.group_gpus(group_name)?.to_vec();

		f(&mut self.thread, &gpu_indices)
	}

	pub fn exec_on_default<F, T>(
		&mut self,
		f: F,
	) -> Result<T>
	where
		F: FnOnce(&mut ManagingThread<R, N>, &[usize]) -> Result<T>,
	{
		let group =
			self.group_manager
				.default_group()
				.ok_or_else(|| ThreadError::GroupNotFound {
					name: "default".to_string(),
				})?;

		let gpu_indices = group.gpu_indices.clone();
		f(&mut self.thread, &gpu_indices)
	}

	pub fn shutdown(self) -> Result<()> {
		self.thread.shutdown()
	}
}

impl ManagerConfig {
	pub fn single_gpu() -> Self {
		Self::builder()
			.with_gpu_count(1)
			.with_group("default", &[0], GroupStrategy::SingleGpu)
			.with_default_group("default")
	}

	pub fn data_parallel<R: Runtime>() -> Self {
		let count = <R as Runtime>::Device::device_count_total();
		let gpus: Vec<usize> = (0..count).collect();

		Self::builder()
			.with_gpu_count(count)
			.with_group(
				"default",
				&gpus,
				GroupStrategy::DataParallel,
			)
			.with_default_group("default")
	}

	pub fn training_inference_split<R: Runtime>() -> Self {
		let count = <R as Runtime>::Device::device_count_total();

		if count < 2 {
			return Self::single_gpu();
		}

		let training_count = (count * 3 / 4).max(1);
		let training_gpus: Vec<usize> = (0..training_count).collect();
		let inference_gpus: Vec<usize> = (training_count..count).collect();

		Self::builder()
			.with_gpu_count(count)
			.with_group_priority(
				"training",
				&training_gpus,
				GroupStrategy::DataParallel,
				8,
			)
			.with_group_tag("training", "high-memory")
			.with_group_priority(
				"inference",
				&inference_gpus,
				GroupStrategy::RoundRobin,
				5,
			)
			.with_default_group("training")
	}

	pub fn pipeline_parallel<R: Runtime>() -> Self {
		let count = <R as Runtime>::Device::device_count_total();
		let gpus: Vec<usize> = (0..count).collect();

		Self::builder()
			.with_gpu_count(count)
			.with_group(
				"pipeline",
				&gpus,
				GroupStrategy::PipelineParallel,
			)
			.with_default_group("pipeline")
	}

	pub fn auto_configure<R: Runtime>() -> Self {
		Self::builder()
			.detect_environment::<R>()
			.print_environment_report()
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_builder_pattern() {
		let config = ManagerConfig::builder()
			.with_gpu_count(4)
			.with_group(
				"test",
				&[0, 1],
				GroupStrategy::DataParallel,
			)
			.with_default_group("test");

		assert_eq!(config.total_gpus, Some(4));
		assert!(config.group_manager.group("test").is_some());
	}

	#[test]
	fn test_validation() {
		let config = ManagerConfig::builder().with_gpu_count(2).with_group(
			"group1",
			&[0, 1],
			GroupStrategy::DataParallel,
		);

		assert!(config.validate().is_ok());
	}

	#[test]
	fn test_preset_single_gpu() {
		let config = ManagerConfig::single_gpu();
		assert_eq!(config.total_gpus, Some(1));
		assert!(config.group_manager.group("default").is_some());
	}
}
