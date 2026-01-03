// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use super::environment::GpuEnvironment;
use super::error::{Result, ThreadError};
use cubecl_common::device::DeviceId;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct GpuGroupManager {
	total_gpus: usize,

	groups: HashMap<String, GpuGroup>,

	gpu_to_group: HashMap<usize, String>,

	default_group: Option<String>,
}

#[derive(Debug, Clone)]
pub struct GpuGroup {
	pub name: String,

	pub gpu_indices: Vec<usize>,

	pub strategy: GroupStrategy,

	pub priority: u8,

	pub enabled: bool,

	pub tags: HashSet<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GroupStrategy {
	SingleGpu,

	DataParallel,

	PipelineParallel,

	TensorParallel,

	RoundRobin,

	Custom,
}

impl GpuGroupManager {
	pub fn new(total_gpus: usize) -> Self {
		let mut manager = Self {
			total_gpus,
			groups: HashMap::new(),
			gpu_to_group: HashMap::new(),
			default_group: None,
		};

		if total_gpus > 0 {
			let all_gpus: Vec<usize> = (0..total_gpus).collect();
			let _ = manager.create_group(
				"default",
				all_gpus,
				GroupStrategy::DataParallel,
				5,
			);
			manager.default_group = Some("default".to_string());
		}

		manager
	}

	pub fn from_environment(env: &GpuEnvironment) -> Self {
		let mut manager = Self::new(env.total_devices);

		if let Some(group) = manager.groups.get_mut("default") {
			group.strategy = match env.recommended_config.distribution_strategy {
				super::environment::RecommendedDistribution::SingleGpu => GroupStrategy::SingleGpu,
				super::environment::RecommendedDistribution::DataParallel => {
					GroupStrategy::DataParallel
				},
				super::environment::RecommendedDistribution::PipelineParallel => {
					GroupStrategy::PipelineParallel
				},
				super::environment::RecommendedDistribution::Hybrid => GroupStrategy::DataParallel,
			};
		}

		manager
	}

	pub fn create_group(
		&mut self,
		name: &str,
		gpu_indices: Vec<usize>,
		strategy: GroupStrategy,
		priority: u8,
	) -> Result<()> {
		for &idx in &gpu_indices {
			if idx >= self.total_gpus {
				return Err(ThreadError::InvalidGpuId {
					id: idx,
					max: self.total_gpus - 1,
				});
			}

			if let Some(existing_group) = self.gpu_to_group.get(&idx) {
				if existing_group != name {
					return Err(ThreadError::GpuAlreadyAssigned {
						gpu_id: idx,
						group: existing_group.clone(),
					});
				}
			}
		}

		let group = GpuGroup {
			name: name.to_string(),
			gpu_indices: gpu_indices.clone(),
			strategy,
			priority,
			enabled: true,
			tags: HashSet::new(),
		};

		for &idx in &gpu_indices {
			self.gpu_to_group.insert(idx, name.to_string());
		}

		self.groups.insert(name.to_string(), group);

		Ok(())
	}

	pub fn group(
		&self,
		name: &str,
	) -> Option<&GpuGroup> {
		self.groups.get(name)
	}

	pub fn group_mut(
		&mut self,
		name: &str,
	) -> Option<&mut GpuGroup> {
		self.groups.get_mut(name)
	}

	pub fn remove_group(
		&mut self,
		name: &str,
	) -> Result<()> {
		if let Some(group) = self.groups.remove(name) {
			for idx in &group.gpu_indices {
				self.gpu_to_group.remove(idx);
			}

			if self.default_group.as_deref() == Some(name) {
				self.default_group = None;
			}

			Ok(())
		} else {
			Err(ThreadError::GroupNotFound {
				name: name.to_string(),
			})
		}
	}

	pub fn set_default_group(
		&mut self,
		name: &str,
	) -> Result<()> {
		if self.groups.contains_key(name) {
			self.default_group = Some(name.to_string());
			Ok(())
		} else {
			Err(ThreadError::GroupNotFound {
				name: name.to_string(),
			})
		}
	}

	pub fn default_group(&self) -> Option<&GpuGroup> {
		self.default_group
			.as_ref()
			.and_then(|name| self.groups.get(name))
	}

	pub fn group_gpus(
		&self,
		name: &str,
	) -> Result<&[usize]> {
		self.groups
			.get(name)
			.map(|g| g.gpu_indices.as_slice())
			.ok_or_else(|| ThreadError::GroupNotFound {
				name: name.to_string(),
			})
	}

	pub fn gpu_group(
		&self,
		gpu_idx: usize,
	) -> Option<&str> {
		self.gpu_to_group.get(&gpu_idx).map(|s| s.as_str())
	}

	pub fn group_names(&self) -> Vec<String> {
		self.groups.keys().cloned().collect()
	}

	pub fn enable_group(
		&mut self,
		name: &str,
	) -> Result<()> {
		self.groups
			.get_mut(name)
			.map(|g| {
				g.enabled = true;
				()
			})
			.ok_or_else(|| ThreadError::GroupNotFound {
				name: name.to_string(),
			})
	}

	pub fn disable_group(
		&mut self,
		name: &str,
	) -> Result<()> {
		self.groups
			.get_mut(name)
			.map(|g| {
				g.enabled = false;
				()
			})
			.ok_or_else(|| ThreadError::GroupNotFound {
				name: name.to_string(),
			})
	}

	pub fn is_group_enabled(
		&self,
		name: &str,
	) -> bool {
		self.groups.get(name).map(|g| g.enabled).unwrap_or(false)
	}

	pub fn add_group_tag(
		&mut self,
		name: &str,
		tag: &str,
	) -> Result<()> {
		self.groups
			.get_mut(name)
			.map(|g| {
				g.tags.insert(tag.to_string());
				()
			})
			.ok_or_else(|| ThreadError::GroupNotFound {
				name: name.to_string(),
			})
	}

	pub fn groups_with_tag(
		&self,
		tag: &str,
	) -> Vec<&GpuGroup> {
		self.groups
			.values()
			.filter(|g| g.tags.contains(tag))
			.collect()
	}

	pub fn set_group_strategy(
		&mut self,
		name: &str,
		strategy: GroupStrategy,
	) -> Result<()> {
		self.groups
			.get_mut(name)
			.map(|g| {
				g.strategy = strategy;
				()
			})
			.ok_or_else(|| ThreadError::GroupNotFound {
				name: name.to_string(),
			})
	}

	pub fn set_group_priority(
		&mut self,
		name: &str,
		priority: u8,
	) -> Result<()> {
		self.groups
			.get_mut(name)
			.map(|g| {
				g.priority = priority;
				()
			})
			.ok_or_else(|| ThreadError::GroupNotFound {
				name: name.to_string(),
			})
	}

	pub fn groups_by_priority(&self) -> Vec<&GpuGroup> {
		let mut groups: Vec<&GpuGroup> = self.groups.values().collect();
		groups.sort_by(|a, b| b.priority.cmp(&a.priority));
		groups
	}

	pub fn validate(&self) -> Result<()> {
		let mut assigned = HashSet::new();

		for group in self.groups.values() {
			for &idx in &group.gpu_indices {
				if assigned.contains(&idx) {
					return Err(ThreadError::GpuAlreadyAssigned {
						gpu_id: idx,
						group: self.gpu_to_group[&idx].clone(),
					});
				}
				assigned.insert(idx);
			}
		}

		Ok(())
	}

	pub fn report(&self) -> String {
		let mut report = String::new();

		report.push_str("=== GPU Group Configuration ===\n\n");
		report.push_str(&format!(
			"Total GPUs: {}\n",
			self.total_gpus
		));
		report.push_str(&format!(
			"Total Groups: {}\n",
			self.groups.len()
		));

		if let Some(default) = &self.default_group {
			report.push_str(&format!("Default Group: {}\n", default));
		}

		report.push_str("\n=== Groups ===\n");

		for group in self.groups_by_priority() {
			report.push_str(&format!(
				"\n[{}] (Priority: {})\n",
				group.name, group.priority
			));
			report.push_str(&format!(
				"  GPUs: {:?}\n",
				group.gpu_indices
			));
			report.push_str(&format!(
				"  Strategy: {:?}\n",
				group.strategy
			));
			report.push_str(&format!("  Enabled: {}\n", group.enabled));

			if !group.tags.is_empty() {
				report.push_str(&format!("  Tags: {:?}\n", group.tags));
			}
		}

		report
	}
}

impl GpuGroup {
	pub fn contains_gpu(
		&self,
		gpu_idx: usize,
	) -> bool {
		self.gpu_indices.contains(&gpu_idx)
	}

	pub fn size(&self) -> usize {
		self.gpu_indices.len()
	}

	pub fn is_empty(&self) -> bool {
		self.gpu_indices.is_empty()
	}

	pub fn first_gpu(&self) -> Option<usize> {
		self.gpu_indices.first().copied()
	}

	pub fn device_ids(&self) -> Vec<DeviceId> {
		self.gpu_indices
			.iter()
			.map(|&idx| DeviceId::new(0, idx as u32))
			.collect()
	}

	pub fn add_tag(
		&mut self,
		tag: impl Into<String>,
	) {
		self.tags.insert(tag.into());
	}

	pub fn has_tag(
		&self,
		tag: &str,
	) -> bool {
		self.tags.contains(tag)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_create_groups() {
		let mut manager = GpuGroupManager::new(8);

		manager
			.create_group(
				"training",
				vec![0, 1, 2, 3],
				GroupStrategy::DataParallel,
				10,
			)
			.unwrap();

		manager
			.create_group(
				"inference",
				vec![4, 5],
				GroupStrategy::RoundRobin,
				5,
			)
			.unwrap();

		manager
			.create_group(
				"debug",
				vec![6, 7],
				GroupStrategy::SingleGpu,
				1,
			)
			.unwrap();

		assert_eq!(manager.groups.len(), 4);
		assert_eq!(
			manager.group_gpus("training").unwrap(),
			&[0, 1, 2, 3]
		);
		assert_eq!(
			manager.group_gpus("inference").unwrap(),
			&[4, 5]
		);
	}

	#[test]
	fn test_gpu_already_assigned() {
		let mut manager = GpuGroupManager::new(4);

		manager
			.create_group(
				"group1",
				vec![0, 1],
				GroupStrategy::DataParallel,
				5,
			)
			.unwrap();

		let result = manager.create_group(
			"group2",
			vec![0, 2],
			GroupStrategy::DataParallel,
			5,
		);

		assert!(result.is_err());
	}

	#[test]
	fn test_group_priority() {
		let mut manager = GpuGroupManager::new(4);

		manager
			.create_group(
				"low",
				vec![0],
				GroupStrategy::SingleGpu,
				1,
			)
			.unwrap();
		manager
			.create_group(
				"high",
				vec![1],
				GroupStrategy::SingleGpu,
				10,
			)
			.unwrap();
		manager
			.create_group(
				"medium",
				vec![2],
				GroupStrategy::SingleGpu,
				5,
			)
			.unwrap();

		let groups = manager.groups_by_priority();
		assert_eq!(groups[0].name, "high");
		assert_eq!(groups[1].name, "medium");
		assert_eq!(groups[2].name, "low");
	}

	#[test]
	fn test_enable_disable() {
		let mut manager = GpuGroupManager::new(2);

		manager
			.create_group(
				"test",
				vec![0, 1],
				GroupStrategy::DataParallel,
				5,
			)
			.unwrap();

		assert!(manager.is_group_enabled("test"));

		manager.disable_group("test").unwrap();
		assert!(!manager.is_group_enabled("test"));

		manager.enable_group("test").unwrap();
		assert!(manager.is_group_enabled("test"));
	}

	#[test]
	fn test_tags() {
		let mut manager = GpuGroupManager::new(4);

		manager
			.create_group(
				"gpu1",
				vec![0],
				GroupStrategy::SingleGpu,
				5,
			)
			.unwrap();
		manager
			.create_group(
				"gpu2",
				vec![1],
				GroupStrategy::SingleGpu,
				5,
			)
			.unwrap();

		manager.add_group_tag("gpu1", "production").unwrap();
		manager.add_group_tag("gpu2", "production").unwrap();
		manager.add_group_tag("gpu1", "high-memory").unwrap();

		let prod_groups = manager.groups_with_tag("production");
		assert_eq!(prod_groups.len(), 2);

		let high_mem_groups = manager.groups_with_tag("high-memory");
		assert_eq!(high_mem_groups.len(), 1);
	}
}
