// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use thiserror::Error;

pub type Result<T> = core::result::Result<T, GpuError>;

#[derive(Debug, Error)]
pub enum GpuError {
	#[error("GpuMemRep Mutex lock dosnt work")]
	LockError,
	#[error("Interval dosnt exist")]
	InvalidInterval,
	#[error("Couldn't get handle")]
	InvalidHandle,
	#[error("Couldn't get handle")]
	HandleNotFound,
	#[error("Explicit tensors have to be same size as GPU copunt")]
	TensorCountMismatch,
}
