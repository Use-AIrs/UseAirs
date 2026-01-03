// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

mod cfg;
pub(crate) mod dyn_mem;
mod error;
pub(crate) mod io;
pub mod kernel_args;
pub mod virtual_tensor;

pub(crate) use cfg::*;
pub(crate) use dyn_mem::*;
pub(crate) use error::*;
pub(crate) use io::*;
pub(crate) use kernel_args::*;
pub(crate) use virtual_tensor::*;

pub(crate) trait GpuOp {
	type InTensors;
	type OutTensors;
	type AccItem;
	type Accumulator;
	fn level_warp() {}
}
