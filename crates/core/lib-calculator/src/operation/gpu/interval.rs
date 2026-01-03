// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use cubecl_common::device::DeviceId;
use cubecl_core::prelude::Numeric;
use cubecl_core::CubeElement;

use super::error::{GpuError, Result};
use super::GpuMemRep;
use super::*;
use std::collections::HashMap;
use std::ops::{Deref, Drop};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Interval {
	_id: usize,
}

impl Interval {
	pub(crate) fn first() -> Self {
		Interval { _id: 0 }
	}
	pub(crate) fn next(&self) -> Self {
		let id = self._id + 1;
		Interval { _id: id }
	}
}

impl Deref for Interval {
	type Target = usize;

	fn deref(&self) -> &Self::Target {
		&self._id
	}
}

impl From<usize> for Interval {
	fn from(id: usize) -> Self {
		Interval { _id: id }
	}
}

pub trait IntervalTuple<N: Numeric + CubeElement> {
	type Output;

	fn interval_map(
		&self,
		mem_rep: &GpuMem<N>,
	) -> Result<Self::Output>;
}

impl<N: Numeric + CubeElement> IntervalTuple<N> for Interval {
	type Output = Arc<HashMap<usize, GpuMemRep>>;

	fn interval_map(
		&self,
		mem_rep: &GpuMem<N>,
	) -> Result<Self::Output> {
		let storage = mem_rep.memory_handles_interval(self).unwrap();
		Ok(Arc::new(storage))
	}
}

impl<N: Numeric + CubeElement> IntervalTuple<N> for (Interval, Interval) {
	type Output = Arc<(
		HashMap<usize, GpuMemRep>,
		HashMap<usize, GpuMemRep>,
	)>;

	fn interval_map(
		&self,
		mem_rep: &GpuMem<N>,
	) -> Result<Self::Output> {
		let storage_a = mem_rep.memory_handles_interval(&self.0)?;
		let storage_b = mem_rep.memory_handles_interval(&self.1)?;

		Ok(Arc::new((storage_a, storage_b)))
	}
}

impl<N: Numeric + CubeElement> IntervalTuple<N> for (Interval, Interval, Interval) {
	type Output = Arc<(
		HashMap<usize, GpuMemRep>,
		HashMap<usize, GpuMemRep>,
		HashMap<usize, GpuMemRep>,
	)>;

	fn interval_map(
		&self,
		mem_rep: &GpuMem<N>,
	) -> Result<Self::Output> {
		let storage_a = mem_rep.memory_handles_interval(&self.0)?;
		let storage_b = mem_rep.memory_handles_interval(&self.1)?;
		let storage_c = mem_rep.memory_handles_interval(&self.2)?;

		Ok(Arc::new((
			storage_a, storage_b, storage_c,
		)))
	}
}

pub trait ExtractMemHandle<T> {
	fn extract_for_gpu(
		&self,
		dev_id: usize,
	) -> Result<T>;
}

impl ExtractMemHandle<GpuMemRep> for Arc<HashMap<usize, GpuMemRep>> {
	fn extract_for_gpu(
		&self,
		dev_id: usize,
	) -> Result<GpuMemRep> {
		self.get(&dev_id).cloned().ok_or(GpuError::HandleNotFound)
	}
}

impl ExtractMemHandle<(GpuMemRep, GpuMemRep)>
	for Arc<(
		HashMap<usize, GpuMemRep>,
		HashMap<usize, GpuMemRep>,
	)>
{
	fn extract_for_gpu(
		&self,
		dev_id: usize,
	) -> Result<(GpuMemRep, GpuMemRep)> {
		let handle_a = self
			.0
			.get(&dev_id)
			.cloned()
			.ok_or(GpuError::HandleNotFound)?;
		let handle_b = self
			.1
			.get(&dev_id)
			.cloned()
			.ok_or(GpuError::HandleNotFound)?;
		Ok((handle_a, handle_b))
	}
}

impl ExtractMemHandle<(GpuMemRep, GpuMemRep, GpuMemRep)>
	for Arc<(
		HashMap<usize, GpuMemRep>,
		HashMap<usize, GpuMemRep>,
		HashMap<usize, GpuMemRep>,
	)>
{
	fn extract_for_gpu(
		&self,
		dev_id: usize,
	) -> Result<(GpuMemRep, GpuMemRep, GpuMemRep)> {
		let handle_a = self
			.0
			.get(&dev_id)
			.cloned()
			.ok_or(GpuError::HandleNotFound)?;
		let handle_b = self
			.1
			.get(&dev_id)
			.cloned()
			.ok_or(GpuError::HandleNotFound)?;
		let handle_c = self
			.2
			.get(&dev_id)
			.cloned()
			.ok_or(GpuError::HandleNotFound)?;
		Ok((handle_a, handle_b, handle_c))
	}
}
