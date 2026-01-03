// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use super::coordinator::DeviceHardwareInfo;
use super::error::{Result, ThreadError};
use super::groups::{GpuGroup, GroupStrategy};
use super::strategy::{
	ComputeIntensity, GroupMetrics, HeuristicSelector, MemoryPattern, MetricsTracker,
	OperationType, StrategySelector, WorkloadProfile,
};
use super::threads::ManagingThread;
use crate::operation::gpu::*;
use crate::{MetaData, Tensor};

use cubecl_common::device::DeviceId;
use cubecl_core::prelude::{CubeElement, Numeric, Runtime};
use std::collections::{HashMap, VecDeque};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

pub struct GroupThread<R: Runtime, N: Numeric + CubeElement> {
	group: GpuGroup,

	strategy_selector: Box<dyn StrategySelector>,

	hardware_info: Vec<DeviceHardwareInfo>,

	metrics: Arc<RwLock<GroupMetrics>>,

	load_balancer: GroupLoadBalancer,

	task_queue: Arc<Mutex<VecDeque<GroupTask>>>,

	active: Arc<Mutex<bool>>,

	config: GroupThreadConfig,

	_phantom: PhantomData<(R, N)>,
}

#[derive(Debug, Clone)]
pub struct GroupThreadConfig {
	pub auto_select_strategy: bool,

	pub max_queue_depth: usize,

	pub metrics_interval: Duration,

	pub adaptive_strategies: bool,

	pub strategy_switch_threshold: f32,

	pub enable_load_balancing: bool,
}

impl Default for GroupThreadConfig {
	fn default() -> Self {
		Self {
			auto_select_strategy: true,
			max_queue_depth: 100,
			metrics_interval: Duration::from_secs(5),
			adaptive_strategies: true,
			strategy_switch_threshold: 0.7,
			enable_load_balancing: true,
		}
	}
}

#[derive(Debug)]
pub struct GroupTask {
	pub id: u64,

	pub workload: WorkloadProfile,

	pub suggested_strategy: Option<GroupStrategy>,

	pub priority: u8,

	pub created_at: Instant,

	pub timeout: Option<Duration>,
}

#[derive(Debug, Clone)]
pub struct GroupLoadBalancer {
	gpu_utilization: HashMap<usize, f32>,

	gpu_task_counts: HashMap<usize, u64>,

	last_balance: Instant,

	balance_interval: Duration,

	strategy: LoadBalanceStrategy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadBalanceStrategy {
	RoundRobin,

	LeastLoaded,

	Weighted,

	Random,
}

impl GroupLoadBalancer {
	pub fn new(
		gpu_indices: &[usize],
		strategy: LoadBalanceStrategy,
	) -> Self {
		let mut gpu_utilization = HashMap::new();
		let mut gpu_task_counts = HashMap::new();

		for &idx in gpu_indices {
			gpu_utilization.insert(idx, 0.0);
			gpu_task_counts.insert(idx, 0);
		}

		Self {
			gpu_utilization,
			gpu_task_counts,
			last_balance: Instant::now(),
			balance_interval: Duration::from_secs(1),
			strategy,
		}
	}

	pub fn select_gpu(
		&mut self,
		gpu_indices: &[usize],
	) -> Option<usize> {
		if gpu_indices.is_empty() {
			return None;
		}

		match self.strategy {
			LoadBalanceStrategy::RoundRobin => {
				let min_tasks = self
					.gpu_task_counts
					.iter()
					.filter(|(idx, _)| gpu_indices.contains(idx))
					.map(|(_, count)| count)
					.min()
					.copied()
					.unwrap_or(0);

				gpu_indices
					.iter()
					.find(|&&idx| self.gpu_task_counts.get(&idx).copied().unwrap_or(0) == min_tasks)
					.copied()
			},

			LoadBalanceStrategy::LeastLoaded => gpu_indices
				.iter()
				.min_by(|&&a, &&b| {
					let util_a = self.gpu_utilization.get(&a).copied().unwrap_or(0.0);
					let util_b = self.gpu_utilization.get(&b).copied().unwrap_or(0.0);
					util_a
						.partial_cmp(&util_b)
						.unwrap_or(std::cmp::Ordering::Equal)
				})
				.copied(),

			LoadBalanceStrategy::Weighted | LoadBalanceStrategy::Random => {
				gpu_indices.first().copied()
			},
		}
	}

	pub fn update_utilization(
		&mut self,
		gpu_idx: usize,
		utilization: f32,
	) {
		self.gpu_utilization
			.insert(gpu_idx, utilization.clamp(0.0, 1.0));
	}

	pub fn increment_task_count(
		&mut self,
		gpu_idx: usize,
	) {
		*self.gpu_task_counts.entry(gpu_idx).or_insert(0) += 1;
	}

	pub fn reset_counts(&mut self) {
		for count in self.gpu_task_counts.values_mut() {
			*count = 0;
		}
	}

	pub fn should_rebalance(&self) -> bool {
		self.last_balance.elapsed() >= self.balance_interval
	}
}

impl<R: Runtime, N: Numeric + CubeElement + Clone + 'static> GroupThread<R, N> {
	/// Create new group thread
	pub fn new(
		group: GpuGroup,
		hardware_info: Vec<DeviceHardwareInfo>,
		config: GroupThreadConfig,
	) -> Self {
		let strategy_selector: Box<dyn StrategySelector> = if config.adaptive_strategies {
			Box::new(super::strategy::AdaptiveSelector::new())
		} else {
			Box::new(HeuristicSelector::new())
		};

		let metrics = Arc::new(RwLock::new(GroupMetrics {
			group_name: group.name.clone(),
			total_executions: 0,
			successful_executions: 0,
			failed_executions: 0,
			avg_execution_time: Duration::from_secs(0),
			min_execution_time: Duration::from_secs(u64::MAX),
			max_execution_time: Duration::from_secs(0),
			total_data_processed: 0,
			current_utilization: 0.0,
			last_execution: None,
			strategy_performance: HashMap::new(),
		}));

		let load_balancer = GroupLoadBalancer::new(
			&group.gpu_indices,
			LoadBalanceStrategy::LeastLoaded,
		);

		Self {
			group,
			strategy_selector,
			hardware_info,
			metrics,
			load_balancer,
			task_queue: Arc::new(Mutex::new(VecDeque::new())),
			active: Arc::new(Mutex::new(true)),
			config,
			_phantom: PhantomData,
		}
	}

	/// Get group reference
	pub fn group(&self) -> &GpuGroup {
		&self.group
	}

	/// Get metrics
	pub fn metrics(&self) -> Arc<RwLock<GroupMetrics>> {
		self.metrics.clone()
	}

	/// Check if group is active
	pub fn is_active(&self) -> bool {
		*self.active.lock().unwrap()
	}

	/// Deactivate group thread
	pub fn deactivate(&self) {
		*self.active.lock().unwrap() = false;
	}

	/// Activate group thread
	pub fn activate(&self) {
		*self.active.lock().unwrap() = true;
	}

	/// Autonomously select strategy for workload
	///
	/// Uses the configured strategy selector to choose optimal strategy
	/// based on workload characteristics and historical performance.
	pub fn select_strategy(
		&self,
		workload: &WorkloadProfile,
	) -> GroupStrategy {
		if !self.config.auto_select_strategy {
			return self.group.strategy;
		}

		let metrics = self.metrics.read().unwrap();

		self.strategy_selector.select_strategy(
			workload,
			&self.group,
			&metrics,
			&self.hardware_info,
		)
	}

	/// Execute kernel with autonomous strategy selection
	///
	/// This is the main entry point for group execution. It will:
	/// 1. Analyze the workload
	/// 2. Select optimal strategy
	/// 3. Execute using selected strategy
	/// 4. Record metrics
	/// 5. Update strategy selector
	pub fn exec_kernel_autonomous<K, I, O>(
		&mut self,
		thread: &ManagingThread<R, N>,
		workload: WorkloadProfile,
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
		if !self.is_active() {
			return Err(ThreadError::WorkerError {
				message: format!(
					"Group '{}' is not active",
					self.group.name
				),
			});
		}

		// Select strategy
		let strategy = self.select_strategy(&workload);

		// Record start time
		let start = Instant::now();
		let data_size = workload.data_size;

		// Execute based on selected strategy
		let result = self.exec_with_strategy(
			thread,
			strategy,
			input_intervals,
			output_intervals,
			kernel,
			config,
		);

		// Record execution metrics
		let execution_time = start.elapsed();
		let success = result.is_ok();

		self.record_execution(
			strategy,
			data_size,
			execution_time,
			success,
		);

		result
	}

	/// Execute with specific strategy
	fn exec_with_strategy<K, I, O>(
		&mut self,
		thread: &ManagingThread<R, N>,
		strategy: GroupStrategy,
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
		match strategy {
			GroupStrategy::SingleGpu => self.exec_single_gpu(
				thread,
				input_intervals,
				output_intervals,
				kernel,
				config,
			),

			GroupStrategy::DataParallel => self.exec_data_parallel(
				thread,
				input_intervals,
				output_intervals,
				kernel,
				config,
			),

			GroupStrategy::RoundRobin => self.exec_round_robin(
				thread,
				input_intervals,
				output_intervals,
				kernel,
				config,
			),

			GroupStrategy::PipelineParallel => {
				// TODO: Implement pipeline parallelism
				self.exec_data_parallel(
					thread,
					input_intervals,
					output_intervals,
					kernel,
					config,
				)
			},

			GroupStrategy::TensorParallel => {
				// TODO: Implement tensor parallelism
				self.exec_data_parallel(
					thread,
					input_intervals,
					output_intervals,
					kernel,
					config,
				)
			},

			GroupStrategy::Custom => Err(ThreadError::WorkerError {
				message: "Custom strategies must be implemented externally".to_string(),
			}),
		}
	}

	/// Execute on single GPU (first in group)
	fn exec_single_gpu<K, I, O>(
		&self,
		thread: &ManagingThread<R, N>,
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
		let gpu_idx = self
			.group
			.gpu_indices
			.first()
			.ok_or_else(|| ThreadError::WorkerError {
				message: format!("Group '{}' has no GPUs", self.group.name),
			})?;

		let dev_id = DeviceId::new(0, *gpu_idx as u32);
		thread.exec_kernel_on_gpu(
			dev_id,
			input_intervals,
			output_intervals,
			kernel,
			config,
		)
	}

	/// Execute with data parallelism across all GPUs
	fn exec_data_parallel<K, I, O>(
		&self,
		thread: &ManagingThread<R, N>,
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
		thread.exec_kernel_on_gpus(
			&self.group.gpu_indices,
			input_intervals,
			output_intervals,
			kernel,
			config,
		)
	}

	/// Execute with round-robin GPU selection
	fn exec_round_robin<K, I, O>(
		&mut self,
		thread: &ManagingThread<R, N>,
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
		let gpu_idx = self
			.load_balancer
			.select_gpu(&self.group.gpu_indices)
			.ok_or_else(|| ThreadError::WorkerError {
				message: format!("Group '{}' has no GPUs", self.group.name),
			})?;

		self.load_balancer.increment_task_count(gpu_idx);

		let dev_id = DeviceId::new(0, gpu_idx as u32);
		thread.exec_kernel_on_gpu(
			dev_id,
			input_intervals,
			output_intervals,
			kernel,
			config,
		)
	}

	/// Record execution metrics
	fn record_execution(
		&mut self,
		strategy: GroupStrategy,
		data_size: usize,
		execution_time: Duration,
		success: bool,
	) {
		let mut metrics = self.metrics.write().unwrap();

		// Update overall metrics
		metrics.total_executions += 1;
		if success {
			metrics.successful_executions += 1;
		} else {
			metrics.failed_executions += 1;
		}

		metrics.total_data_processed += data_size;
		metrics.last_execution = Some(Instant::now());

		// Update timing
		if execution_time < metrics.min_execution_time {
			metrics.min_execution_time = execution_time;
		}
		if execution_time > metrics.max_execution_time {
			metrics.max_execution_time = execution_time;
		}

		let prev_avg = metrics.avg_execution_time.as_secs_f64();
		let new_time = execution_time.as_secs_f64();
		let count = metrics.total_executions as f64;
		let new_avg = (prev_avg * (count - 1.0) + new_time) / count;
		metrics.avg_execution_time = Duration::from_secs_f64(new_avg);

		// Update strategy-specific metrics
		let strategy_metrics = metrics
			.strategy_performance
			.entry(strategy)
			.or_insert_with(|| super::strategy::StrategyMetrics {
				usage_count: 0,
				success_rate: 0.0,
				avg_time: Duration::from_secs(0),
				throughput: 0.0,
				last_used: None,
			});

		strategy_metrics.usage_count += 1;
		strategy_metrics.last_used = Some(Instant::now());

		let prev_success = strategy_metrics.success_rate;
		let new_success = if success { 1.0 } else { 0.0 };
		strategy_metrics.success_rate = (prev_success * (strategy_metrics.usage_count - 1) as f32
			+ new_success)
			/ strategy_metrics.usage_count as f32;

		let prev_time = strategy_metrics.avg_time.as_secs_f64();
		let new_avg_time = (prev_time * (strategy_metrics.usage_count - 1) as f64 + new_time)
			/ strategy_metrics.usage_count as f64;
		strategy_metrics.avg_time = Duration::from_secs_f64(new_avg_time);

		strategy_metrics.throughput = data_size as f64 / execution_time.as_secs_f64();

		// Update strategy selector with result
		drop(metrics); // Release lock before updating selector
		self.strategy_selector.update_from_result(
			&WorkloadProfile {
				data_size,
				batch_size: 1,
				operation_type: OperationType::Mixed,
				memory_pattern: MemoryPattern::Mixed,
				compute_intensity: ComputeIntensity::Balanced,
				independent_operations: true,
				estimated_duration: Some(execution_time),
				priority: 5,
			},
			strategy,
			success,
			execution_time,
		);
	}

	/// Generate performance report
	pub fn report(&self) -> String {
		let metrics = self.metrics.read().unwrap();
		let mut report = String::new();

		report.push_str(&format!(
			"╔═══════════════════════════════════════════════════════════════╗\n"
		));
		report.push_str(&format!(
			"║  Group Thread: {:<48}║\n",
			self.group.name
		));
		report.push_str(&format!(
			"╚═══════════════════════════════════════════════════════════════╝\n\n"
		));

		report.push_str(&format!(
			"GPUs: {:?}\n",
			self.group.gpu_indices
		));
		report.push_str(&format!(
			"Strategy: {:?}\n",
			self.group.strategy
		));
		report.push_str(&format!("Active: {}\n", self.is_active()));
		report.push_str(&format!(
			"Priority: {}\n\n",
			self.group.priority
		));

		report.push_str("Performance Metrics:\n");
		report.push_str(&format!(
			"  Total Executions: {}\n",
			metrics.total_executions
		));
		report.push_str(&format!(
			"  Success Rate: {:.1}%\n",
			(metrics.successful_executions as f64 / metrics.total_executions.max(1) as f64) * 100.0
		));
		report.push_str(&format!(
			"  Avg Execution Time: {:.3}s\n",
			metrics.avg_execution_time.as_secs_f64()
		));
		report.push_str(&format!(
			"  Data Processed: {:.2} GB\n\n",
			metrics.total_data_processed as f64 / (1024.0 * 1024.0 * 1024.0)
		));

		if !metrics.strategy_performance.is_empty() {
			report.push_str("Strategy Performance:\n");
			for (strategy, perf) in &metrics.strategy_performance {
				report.push_str(&format!("  {:?}:\n", strategy));
				report.push_str(&format!(
					"    Usage: {}\n",
					perf.usage_count
				));
				report.push_str(&format!(
					"    Success: {:.1}%\n",
					perf.success_rate * 100.0
				));
				report.push_str(&format!(
					"    Avg Time: {:.3}s\n",
					perf.avg_time.as_secs_f64()
				));
				report.push_str(&format!(
					"    Throughput: {:.2} GB/s\n",
					perf.throughput / (1024.0 * 1024.0 * 1024.0)
				));
			}
		}

		report
	}
}

// ============================================================================
// FUTURE: Message-Based Threading System
// ============================================================================
//
// When moving to separate threads, use this architecture:
//
// pub enum GroupThreadMessage {
//     ExecuteKernel { workload: WorkloadProfile, ... },
//     UpdateStrategy { strategy: GroupStrategy },
//     GetMetrics { response: Sender<GroupMetrics> },
//     Shutdown,
// }
//
// pub fn spawn_group_thread<R, N>(
//     group: GpuGroup,
//     hardware: Vec<DeviceHardwareInfo>,
// ) -> (Sender<GroupThreadMessage>, JoinHandle<()>) {
//     let (tx, rx) = channel();
//     let handle = thread::spawn(move || {
//         let mut group_thread = GroupThread::new(...);
//         group_thread.run_loop(rx);
//     });
//     (tx, handle)
// }

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_load_balancer_round_robin() {
		let mut balancer = GroupLoadBalancer::new(
			&[0, 1, 2],
			LoadBalanceStrategy::RoundRobin,
		);

		let gpu1 = balancer.select_gpu(&[0, 1, 2]);
		assert!(gpu1.is_some());

		balancer.increment_task_count(gpu1.unwrap());
		let gpu2 = balancer.select_gpu(&[0, 1, 2]);

		// Should select different GPU due to round-robin
		assert!(gpu2.is_some());
	}

	#[test]
	fn test_load_balancer_least_loaded() {
		let mut balancer = GroupLoadBalancer::new(
			&[0, 1, 2],
			LoadBalanceStrategy::LeastLoaded,
		);

		balancer.update_utilization(0, 0.9);
		balancer.update_utilization(1, 0.1);
		balancer.update_utilization(2, 0.5);

		let selected = balancer.select_gpu(&[0, 1, 2]);
		assert_eq!(selected, Some(1)); // Should select GPU 1 (lowest utilization)
	}

	#[test]
	fn test_group_thread_config() {
		let config = GroupThreadConfig::default();
		assert!(config.auto_select_strategy);
		assert!(config.adaptive_strategies);
		assert_eq!(config.max_queue_depth, 100);
	}
}
