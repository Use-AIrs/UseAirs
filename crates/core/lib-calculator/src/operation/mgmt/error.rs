// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use cubecl::Runtime;
use cubecl_common::device::DeviceId;
use thiserror::Error;

pub type Result<T> = core::result::Result<T, ThreadError>;

#[derive(Debug, Error)]
pub enum ThreadError {
	#[error("Failed to lock mutex")]
	LockError,

	#[error("Invalid GPU ID: {id}. Available GPUs: 0-{max}")]
	InvalidGpuId { id: usize, max: usize },

	#[error("Channel closed unexpectedly")]
	ChannelClosed,

	#[error("Failed to send message via channel")]
	SendError,

	#[error("Failed to receive message via channel")]
	RecvError,

	#[error("Invalid response received from GPU worker")]
	InvalidResponse,

	#[error("GPU worker returned error: {message}")]
	WorkerError { message: String },

	#[error("GPU error: {0}")]
	GpuError(#[from] crate::operation::gpu::GpuError),

	#[error("Memory handle not found on GPU {dev_id}")]
	HandleNotFound { dev_id: DeviceId },

	#[error("Interval not found")]
	IntervalNotFound,

	#[error("Tensor count mismatch: expected {expected}, got {actual}")]
	TensorSizeMismatch { expected: usize, actual: usize },

	#[error("Worker thread panicked")]
	WorkerPanic,

	#[error("Failed to create GpuMem wrapper")]
	GpuMemCreationError,

	#[error("GPU {gpu_id} is already assigned to group '{group}'")]
	GpuAlreadyAssigned { gpu_id: usize, group: String },

	#[error("Group '{name}' not found")]
	GroupNotFound { name: String },
}

impl<R: Runtime, N> From<crossbeam::channel::SendError<crate::operation::gpu::GpuCommand<R, N>>>
	for ThreadError
where
	N: cubecl::prelude::Numeric + cubecl::prelude::CubeElement,
{
	fn from(_: crossbeam::channel::SendError<crate::operation::gpu::GpuCommand<R, N>>) -> Self {
		ThreadError::SendError
	}
}

impl From<crossbeam::channel::RecvError> for ThreadError {
	fn from(_: crossbeam::channel::RecvError) -> Self {
		ThreadError::RecvError
	}
}
