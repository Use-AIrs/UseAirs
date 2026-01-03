// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use super::coordinator::DeviceHardwareInfo;
use super::environment::{GpuDeviceInfo, GpuEnvironment};
use super::groups::{GpuGroup, GroupStrategy};
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct WorkloadProfile {
	pub data_size: usize,

	pub batch_size: usize,

	pub operation_type: OperationType,

	pub memory_pattern: MemoryPattern,

	pub compute_intensity: ComputeIntensity,

	pub independent_operations: bool,

	pub estimated_duration: Option<Duration>,

	pub priority: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationType {
	MatrixCompute,

	ElementWise,

	Reduction,

	DataTransfer,

	Pipeline,

	Mixed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPattern {
	Sequential,

	Strided,

	Random,

	Mixed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ComputeIntensity {
	MemoryBound,

	Balanced,

	ComputeBound,
}

#[derive(Debug, Clone)]
pub struct GroupMetrics {
	pub group_name: String,

	pub total_executions: u64,

	pub successful_executions: u64,

	pub failed_executions: u64,

	pub avg_execution_time: Duration,

	pub min_execution_time: Duration,
	pub max_execution_time: Duration,

	pub total_data_processed: usize,

	pub current_utilization: f32,

	pub last_execution: Option<Instant>,

	pub strategy_performance: HashMap<GroupStrategy, StrategyMetrics>,
}

#[derive(Debug, Clone)]
pub struct StrategyMetrics {
	pub usage_count: u64,

	pub success_rate: f32,

	pub avg_time: Duration,

	pub throughput: f64,

	pub last_used: Option<Instant>,
}

pub trait StrategySelector: Send + Sync {
	fn select_strategy(
		&self,
		workload: &WorkloadProfile,
		group: &GpuGroup,
		metrics: &GroupMetrics,
		hardware: &[DeviceHardwareInfo],
	) -> GroupStrategy;

	fn update_from_result(
		&mut self,
		workload: &WorkloadProfile,
		strategy: GroupStrategy,
		success: bool,
		execution_time: Duration,
	);

	fn name(&self) -> &str;
}

#[derive(Debug, Clone)]
pub struct HeuristicSelector {
	large_data_threshold: usize,

	min_batch_for_parallel: usize,

	pipeline_stage_threshold: usize,
}

impl Default for HeuristicSelector {
	fn default() -> Self {
		Self {
			large_data_threshold: 100 * 1024 * 1024,
			min_batch_for_parallel: 8,
			pipeline_stage_threshold: 4,
		}
	}
}

impl HeuristicSelector {
	pub fn new() -> Self {
		Self::default()
	}

	pub fn with_thresholds(
		large_data_mb: usize,
		min_batch: usize,
		pipeline_stages: usize,
	) -> Self {
		Self {
			large_data_threshold: large_data_mb * 1024 * 1024,
			min_batch_for_parallel: min_batch,
			pipeline_stage_threshold: pipeline_stages,
		}
	}
}

impl StrategySelector for HeuristicSelector {
	fn select_strategy(
		&self,
		workload: &WorkloadProfile,
		group: &GpuGroup,
		metrics: &GroupMetrics,
		hardware: &[DeviceHardwareInfo],
	) -> GroupStrategy {
		let group_size = group.gpu_indices.len();

		if group_size == 1 {
			return GroupStrategy::SingleGpu;
		}

		if let Some(current_metrics) = metrics.strategy_performance.get(&group.strategy) {
			if current_metrics.success_rate > 0.95 && current_metrics.usage_count > 10 {
				return group.strategy;
			}
		}

		match workload.operation_type {
			OperationType::MatrixCompute => {
				if workload.batch_size >= self.min_batch_for_parallel {
					GroupStrategy::DataParallel
				} else if workload.data_size > self.large_data_threshold {
					GroupStrategy::TensorParallel
				} else {
					GroupStrategy::SingleGpu
				}
			},

			OperationType::ElementWise => {
				if workload.independent_operations && workload.batch_size > 1 {
					GroupStrategy::DataParallel
				} else {
					GroupStrategy::SingleGpu
				}
			},

			OperationType::Reduction => {
				if group_size <= 4 && workload.data_size > self.large_data_threshold {
					GroupStrategy::DataParallel
				} else {
					GroupStrategy::SingleGpu
				}
			},

			OperationType::DataTransfer => {
				if workload.memory_pattern == MemoryPattern::Sequential {
					GroupStrategy::DataParallel
				} else {
					GroupStrategy::SingleGpu
				}
			},

			OperationType::Pipeline => {
				if group_size >= self.pipeline_stage_threshold {
					GroupStrategy::PipelineParallel
				} else {
					GroupStrategy::DataParallel
				}
			},

			OperationType::Mixed => {
				if metrics.current_utilization < 0.5 {
					GroupStrategy::DataParallel
				} else {
					group.strategy
				}
			},
		}
	}

	fn update_from_result(
		&mut self,
		_workload: &WorkloadProfile,
		_strategy: GroupStrategy,
		_success: bool,
		_execution_time: Duration,
	) {
	}

	fn name(&self) -> &str {
		"HeuristicSelector"
	}
}

#[derive(Debug, Clone)]
pub struct AdaptiveSelector {
	heuristic: HeuristicSelector,

	epsilon: f32,

	learning_rate: f32,

	strategy_scores: HashMap<GroupStrategy, f32>,

	min_samples_for_adaptation: u64,
}

impl Default for AdaptiveSelector {
	fn default() -> Self {
		let mut strategy_scores = HashMap::new();
		strategy_scores.insert(GroupStrategy::SingleGpu, 1.0);
		strategy_scores.insert(GroupStrategy::DataParallel, 1.0);
		strategy_scores.insert(GroupStrategy::PipelineParallel, 1.0);
		strategy_scores.insert(GroupStrategy::TensorParallel, 1.0);
		strategy_scores.insert(GroupStrategy::RoundRobin, 1.0);

		Self {
			heuristic: HeuristicSelector::new(),
			epsilon: 0.1,
			learning_rate: 0.1,
			strategy_scores,
			min_samples_for_adaptation: 20,
		}
	}
}

impl AdaptiveSelector {
	pub fn new() -> Self {
		Self::default()
	}

	pub fn with_exploration_rate(
		mut self,
		epsilon: f32,
	) -> Self {
		self.epsilon = epsilon.clamp(0.0, 1.0);
		self
	}

	pub fn with_learning_rate(
		mut self,
		rate: f32,
	) -> Self {
		self.learning_rate = rate.clamp(0.01, 1.0);
		self
	}

	fn best_strategy(
		&self,
		metrics: &GroupMetrics,
	) -> Option<GroupStrategy> {
		if metrics.total_executions < self.min_samples_for_adaptation {
			return None;
		}

		metrics
			.strategy_performance
			.iter()
			.max_by(|(_, a), (_, b)| {
				let score_a = self.compute_score(a);
				let score_b = self.compute_score(b);
				score_a
					.partial_cmp(&score_b)
					.unwrap_or(std::cmp::Ordering::Equal)
			})
			.map(|(strategy, _)| *strategy)
	}

	fn compute_score(
		&self,
		metrics: &StrategyMetrics,
	) -> f32 {
		if metrics.usage_count == 0 {
			return 0.0;
		}

		let success_weight = 0.6;
		let throughput_weight = 0.4;

		let normalized_throughput = (metrics.throughput / 1e9).min(1.0) as f32;

		(metrics.success_rate * success_weight) + (normalized_throughput * throughput_weight)
	}

	fn should_explore(&self) -> bool {
		use rand::Rng;
		let mut rng = rand::thread_rng();
		rng.gen::<f32>() < self.epsilon
	}
}

impl StrategySelector for AdaptiveSelector {
	fn select_strategy(
		&self,
		workload: &WorkloadProfile,
		group: &GpuGroup,
		metrics: &GroupMetrics,
		hardware: &[DeviceHardwareInfo],
	) -> GroupStrategy {
		if self.should_explore() {
			self.heuristic
				.select_strategy(workload, group, metrics, hardware)
		} else {
			self.best_strategy(metrics).unwrap_or_else(|| {
				self.heuristic
					.select_strategy(workload, group, metrics, hardware)
			})
		}
	}

	fn update_from_result(
		&mut self,
		workload: &WorkloadProfile,
		strategy: GroupStrategy,
		success: bool,
		execution_time: Duration,
	) {
		let current_score = self.strategy_scores.get(&strategy).copied().unwrap_or(1.0);

		let reward = if success {
			let throughput = workload.data_size as f64 / execution_time.as_secs_f64();
			let normalized_reward = (throughput / 1e9).min(1.0) as f32;
			normalized_reward
		} else {
			-0.5
		};

		let new_score = current_score + self.learning_rate * reward;
		self.strategy_scores.insert(strategy, new_score.max(0.0));
	}

	fn name(&self) -> &str {
		"AdaptiveSelector"
	}
}

pub struct MetricsTracker {
	group_metrics: HashMap<String, GroupMetrics>,
}

impl MetricsTracker {
	pub fn new() -> Self {
		Self {
			group_metrics: HashMap::new(),
		}
	}

	pub fn init_group(
		&mut self,
		group_name: String,
	) {
		self.group_metrics.insert(
			group_name.clone(),
			GroupMetrics {
				group_name,
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
			},
		);
	}

	pub fn record_execution(
		&mut self,
		group_name: &str,
		strategy: GroupStrategy,
		data_size: usize,
		execution_time: Duration,
		success: bool,
	) {
		let metrics = self
			.group_metrics
			.entry(group_name.to_string())
			.or_insert_with(|| GroupMetrics {
				group_name: group_name.to_string(),
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
			});

		metrics.total_executions += 1;
		if success {
			metrics.successful_executions += 1;
		} else {
			metrics.failed_executions += 1;
		}

		metrics.total_data_processed += data_size;
		metrics.last_execution = Some(Instant::now());

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

		let strategy_metrics = metrics
			.strategy_performance
			.entry(strategy)
			.or_insert_with(|| StrategyMetrics {
				usage_count: 0,
				success_rate: 0.0,
				avg_time: Duration::from_secs(0),
				throughput: 0.0,
				last_used: None,
			});

		strategy_metrics.usage_count += 1;
		strategy_metrics.last_used = Some(Instant::now());

		let prev_success_rate = strategy_metrics.success_rate;
		let new_success = if success { 1.0 } else { 0.0 };
		strategy_metrics.success_rate =
			(prev_success_rate * (strategy_metrics.usage_count - 1) as f32 + new_success)
				/ strategy_metrics.usage_count as f32;

		let prev_time = strategy_metrics.avg_time.as_secs_f64();
		let new_avg_time = (prev_time * (strategy_metrics.usage_count - 1) as f64 + new_time)
			/ strategy_metrics.usage_count as f64;
		strategy_metrics.avg_time = Duration::from_secs_f64(new_avg_time);

		strategy_metrics.throughput = data_size as f64 / execution_time.as_secs_f64();
	}

	pub fn get_metrics(
		&self,
		group_name: &str,
	) -> Option<&GroupMetrics> {
		self.group_metrics.get(group_name)
	}

	pub fn get_metrics_mut(
		&mut self,
		group_name: &str,
	) -> Option<&mut GroupMetrics> {
		self.group_metrics.get_mut(group_name)
	}

	pub fn report(&self) -> String {
		let mut report = String::new();

		report.push_str("╔═══════════════════════════════════════════════════════════════╗\n");
		report.push_str("║           GROUP PERFORMANCE METRICS                           ║\n");
		report.push_str("╚═══════════════════════════════════════════════════════════════╝\n\n");

		for (group_name, metrics) in &self.group_metrics {
			report.push_str(&format!("Group: {}\n", group_name));
			report.push_str(&format!(
				"  Total Executions: {}\n",
				metrics.total_executions
			));
			report.push_str(&format!(
				"  Success Rate: {:.1}%\n",
				(metrics.successful_executions as f64 / metrics.total_executions as f64) * 100.0
			));
			report.push_str(&format!(
				"  Avg Execution Time: {:.3}s\n",
				metrics.avg_execution_time.as_secs_f64()
			));
			report.push_str(&format!(
				"  Data Processed: {:.2} GB\n",
				metrics.total_data_processed as f64 / (1024.0 * 1024.0 * 1024.0)
			));

			if !metrics.strategy_performance.is_empty() {
				report.push_str("  Strategy Performance:\n");
				for (strategy, perf) in &metrics.strategy_performance {
					report.push_str(&format!("    {:?}:\n", strategy));
					report.push_str(&format!(
						"      Usage: {} times\n",
						perf.usage_count
					));
					report.push_str(&format!(
						"      Success Rate: {:.1}%\n",
						perf.success_rate * 100.0
					));
					report.push_str(&format!(
						"      Avg Time: {:.3}s\n",
						perf.avg_time.as_secs_f64()
					));
					report.push_str(&format!(
						"      Throughput: {:.2} GB/s\n",
						perf.throughput / (1024.0 * 1024.0 * 1024.0)
					));
				}
			}

			report.push_str("\n");
		}

		report
	}
}

impl Default for MetricsTracker {
	fn default() -> Self {
		Self::new()
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_workload_profile() {
		let profile = WorkloadProfile {
			data_size: 1024 * 1024 * 100,
			batch_size: 32,
			operation_type: OperationType::MatrixCompute,
			memory_pattern: MemoryPattern::Sequential,
			compute_intensity: ComputeIntensity::ComputeBound,
			independent_operations: true,
			estimated_duration: None,
			priority: 5,
		};

		assert_eq!(profile.batch_size, 32);
		assert_eq!(
			profile.operation_type,
			OperationType::MatrixCompute
		);
	}

	#[test]
	fn test_metrics_tracker() {
		let mut tracker = MetricsTracker::new();
		tracker.init_group("test_group".to_string());

		tracker.record_execution(
			"test_group",
			GroupStrategy::DataParallel,
			1024 * 1024,
			Duration::from_millis(100),
			true,
		);

		let metrics = tracker.get_metrics("test_group").unwrap();
		assert_eq!(metrics.total_executions, 1);
		assert_eq!(metrics.successful_executions, 1);
	}

	#[test]
	fn test_heuristic_selector() {
		let selector = HeuristicSelector::new();
		assert_eq!(selector.name(), "HeuristicSelector");
	}

	#[test]
	fn test_adaptive_selector() {
		let selector = AdaptiveSelector::new()
			.with_exploration_rate(0.2)
			.with_learning_rate(0.15);

		assert_eq!(selector.name(), "AdaptiveSelector");
		assert_eq!(selector.epsilon, 0.2);
	}
}
