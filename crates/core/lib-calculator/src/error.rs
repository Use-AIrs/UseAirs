// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use thiserror::Error;

pub type Result<T> = core::result::Result<T, CalcError>;

#[derive(Debug, Error)]
pub enum CalcError {
	#[error("Gpu Error")]
	GpuError,
	#[error("transparent")]
	OperationError,
}
