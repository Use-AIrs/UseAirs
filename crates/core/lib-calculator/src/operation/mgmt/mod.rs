// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

mod config;
mod coordinator;
mod environment;
mod error;
mod group_thread;
mod groups;
mod strategy;
mod threads;

pub use config::{ConfiguredManager, ManagerConfig};
pub use coordinator::{DeviceHardwareInfo, GpuCoordinator, GpuCoordinatorBuilder};
pub use environment::{
	ComputeCapability, ComputeFeature, EnvironmentConfig, EnvironmentError, GpuDeviceInfo,
	GpuEnvironment, RecommendedDistribution, SystemMemoryInfo,
};
pub use group_thread::{
	GroupLoadBalancer, GroupTask, GroupThread, GroupThreadConfig, LoadBalanceStrategy,
};
pub use groups::{GpuGroup, GpuGroupManager, GroupStrategy};
pub use strategy::{
	AdaptiveSelector, ComputeIntensity, GroupMetrics, HeuristicSelector, MemoryPattern,
	MetricsTracker, OperationType, StrategyMetrics, StrategySelector, WorkloadProfile,
};
pub use threads::ManagingThread;
