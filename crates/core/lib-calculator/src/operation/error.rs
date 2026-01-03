// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use std::sync::PoisonError;

use cubecl_runtime::ext::ExtensionError;
use thiserror::Error;

pub type Result<T> = core::result::Result<T, OpError>;

#[derive(Debug, Error)]
pub enum OpError {
	#[error("Operation Error")]
	OpError,

	#[error("Couldn't get handle")]
	TensorNotAvitable,

	#[error("Poisoned lock: {0}")]
	PoisonError(String),

	#[error("InvalidConfiguration")]
	InvalidConfiguration,
	#[error("CommunicationError")]
	CommunicationError,
	#[error("InvalidGpuId")]
	InvalidGpuId,
	#[error("Execution Error")]
	ExecutionError,
	#[error("HandleNotFound")]
	HandleNotFound,
	#[error("IntervalNotFound")]
	IntervalNotFound,
	#[error("IntervalNotFound")]
	UnexpectedResponse,
	#[error("IntervalNotFound")]
	InvalidDataCount,
	#[error("TensorCountMismatch")]
	TensorCountMismatch,
	#[error("IntervalAlreadyExists")]
	IntervalAlreadyExists,
	#[error("Error locking Mutex")]
	LockError,
	#[error("Interval not found")]
	InvalidInterval,
	#[error("GpuMem not found")]
	InvalidHandle,
	#[error("InvalidResponse")]
	InvalidResponse,
	#[error("{err}")]
	GpuError { err: String },
	#[error("InvalidResponse")]
	CtxError,
	#[error("Extension Error: {0}")]
	ExtensionError(String),
}

impl<T> From<PoisonError<T>> for OpError {
	fn from(err: PoisonError<T>) -> Self {
		OpError::PoisonError(err.to_string())
	}
}

impl From<ExtensionError> for OpError {
	fn from(err: ExtensionError) -> Self {
		OpError::ExtensionError(format!("{:?}", err))
	}
}
