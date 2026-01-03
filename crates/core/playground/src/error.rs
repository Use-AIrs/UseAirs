// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use std::convert::Infallible;
use std::num::{ParseFloatError, ParseIntError};
pub use thiserror_impl::*;

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
	#[error("GpuMem not found")]
	InvalidHandle(#[from] lib_calculator::operation::error::OpError),
	#[error("Time Error")]
	TimeError(#[from] std::io::Error),
}
