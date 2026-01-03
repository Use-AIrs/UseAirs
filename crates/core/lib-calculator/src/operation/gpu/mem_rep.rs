// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use super::error::{GpuError, Result};
use super::Interval;
use crate::MetaData;
use cubecl_common::device::DeviceId;
use cubecl_core::prelude::*;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct GpuMemRep {
	pub id: usize,
	pub size: usize,
	pub byte_size: usize,
	pub metadata: crate::MetaData,
}

impl GpuMemRep {
	pub fn needs_col_padding(&self) -> bool {
		let cols = self.metadata.shape[self.metadata.shape.len() - 1];
		!cols.is_power_of_two()
	}

	pub fn padded_cols(&self) -> usize {
		let current_cols = self.metadata.shape[self.metadata.shape.len() - 1];
		current_cols.next_power_of_two()
	}

	pub fn is_bitonic_compatible(&self) -> bool {
		self.metadata.shape.len() == 2
	}

	pub fn with_row_padding(&self) -> Self {
		let current_rows = self.metadata.shape[0];
		let padded_rows = current_rows.next_power_of_two();
		let mut new_shape = self.metadata.shape.to_vec();
		new_shape[0] = padded_rows;
		let new_stride = MetaData::row_strides(&new_shape);
		Self {
			id: self.id,
			size: self.size,
			byte_size: self.byte_size,
			metadata: MetaData {
				stride: new_stride,
				shape: new_shape.into_boxed_slice(),
			},
		}
	}

	pub fn with_col_padding(&self) -> Self {
		let current_cols = self.metadata.shape[self.metadata.shape.len() - 1];
		let padded_cols = current_cols.next_power_of_two();
		let mut new_shape = self.metadata.shape.to_vec();
		let n = new_shape.len() - 1;
		new_shape[n] = padded_cols;
		let new_stride = MetaData::row_strides(&new_shape);
		Self {
			id: self.id,
			size: self.size,
			byte_size: self.byte_size,
			metadata: MetaData {
				stride: new_stride,
				shape: new_shape.into_boxed_slice(),
			},
		}
	}

	pub fn row_padding_bytes<N: Numeric>(&self) -> usize {
		let current_rows = self.metadata.shape[0];
		let padded_rows = current_rows.next_power_of_two();
		let padding_rows = padded_rows - current_rows;
		let cols = if self.metadata.shape.len() > 1 {
			self.metadata.shape[1]
		} else {
			1
		};
		padding_rows * cols * std::mem::size_of::<N>()
	}

	pub fn col_padding_bytes<N: Numeric>(&self) -> usize {
		let current_cols = self.metadata.shape[self.metadata.shape.len() - 1];
		let padded_cols = current_cols.next_power_of_two();
		let padding_cols = padded_cols - current_cols;
		let rows = if self.metadata.shape.len() > 1 {
			self.metadata.shape[0]
		} else {
			1
		};
		rows * padding_cols * std::mem::size_of::<N>()
	}
}

pub struct GpuMem<N: Numeric + CubeElement> {
	storage: Arc<Mutex<HashMap<Interval, HashMap<usize, GpuMemRep>>>>,
	next_interval: Arc<Mutex<Interval>>,
	_pd: PhantomData<N>,
}

impl<N: Numeric + CubeElement> GpuMem<N> {
	pub(crate) fn init() -> Result<GpuMem<N>> {
		let storage = Arc::new(Mutex::new(HashMap::new()));
		let next_interval = Arc::new(Mutex::new(Interval::first()));
		Ok(Self {
			storage,
			next_interval,
			_pd: PhantomData::<N>,
		})
	}

	pub(crate) fn interval_create(&self) -> Result<Interval> {
		let mut next = self.next_interval.lock().map_err(|_| GpuError::LockError)?;
		let interval = *next;
		*next = interval.next();

		let mut storage = self.storage.lock().map_err(|_| GpuError::LockError)?;
		storage.insert(interval, HashMap::new());
		Ok(interval)
	}

	pub(crate) fn interval_remove(
		&self,
		interval: Interval,
	) -> Result<()> {
		let mut storage = self.storage.lock().map_err(|_| GpuError::LockError)?;

		if !storage.contains_key(&interval) {
			return Err(GpuError::InvalidInterval);
		}

		storage.remove(&interval);
		Ok(())
	}

	pub(crate) fn interval_exists(
		&self,
		interval: Interval,
	) -> Result<bool> {
		let storage = self.storage.lock().map_err(|_| GpuError::LockError)?;
		Ok(storage.contains_key(&interval))
	}

	pub(crate) fn intervals(&self) -> Result<Vec<Interval>> {
		let storage = self.storage.lock().map_err(|_| GpuError::LockError)?;
		Ok(storage.keys().cloned().collect())
	}

	pub(crate) fn interval_size(
		&self,
		interval: Interval,
	) -> Result<usize> {
		let storage = self.storage.lock().map_err(|_| GpuError::LockError)?;

		let interval_map = storage.get(&interval).ok_or(GpuError::InvalidInterval)?;

		Ok(interval_map.len())
	}

	pub(crate) fn interval_get_handles(
		&self,
		interval: Interval,
	) -> Result<Vec<(DeviceId, GpuMemRep)>> {
		let storage = self.storage.lock().map_err(|_| GpuError::LockError)?;

		let interval_map = storage.get(&interval).ok_or(GpuError::InvalidInterval)?;

		let mut handles = Vec::new();
		for (dev_id, mem_rep) in interval_map {
			handles.push((
				DeviceId::new(0, *dev_id as u32),
				mem_rep.clone(),
			));
		}

		Ok(handles)
	}

	pub(crate) fn interval_clear(
		&self,
		interval: Interval,
	) -> Result<()> {
		let mut storage = self.storage.lock().map_err(|_| GpuError::LockError)?;

		let interval_map = storage
			.get_mut(&interval)
			.ok_or(GpuError::InvalidInterval)?;

		interval_map.clear();
		Ok(())
	}

	pub(crate) fn intervals_total(&self) -> Result<usize> {
		let storage = self.storage.lock().map_err(|_| GpuError::LockError)?;
		Ok(storage.len())
	}

	pub(crate) fn interval_with<F, R>(
		&self,
		interval: Interval,
		f: F,
	) -> Result<R>
	where
		F: FnOnce(&mut HashMap<usize, GpuMemRep>) -> R,
	{
		let mut storage = self.storage.lock().map_err(|_| GpuError::LockError)?;

		let interval_map = storage
			.get_mut(&interval)
			.ok_or(GpuError::InvalidInterval)?;

		Ok(f(interval_map))
	}

	pub(crate) fn memory_handle_add(
		&self,
		interval: Interval,
		handle_id: usize,
		handle: GpuMemRep,
	) -> Result<()> {
		let mut storage = self.storage.lock().map_err(|_| GpuError::LockError)?;

		let interval_map = storage
			.get_mut(&interval)
			.ok_or(GpuError::InvalidInterval)?;

		interval_map.insert(handle_id, handle);
		Ok(())
	}

	pub(crate) fn memory_handle_remove(
		&self,
		interval: Interval,
		handle_id: usize,
	) -> Result<GpuMemRep> {
		let mut storage = self.storage.lock().map_err(|_| GpuError::LockError)?;

		let interval_map = storage
			.get_mut(&interval)
			.ok_or(GpuError::InvalidInterval)?;

		interval_map
			.remove(&handle_id)
			.ok_or(GpuError::InvalidHandle)
	}

	pub(crate) fn memory_handle_specific(
		&self,
		interval: Interval,
		handle_id: usize,
	) -> Result<GpuMemRep> {
		let storage = self.storage.lock().map_err(|_| GpuError::LockError)?;

		let interval_map = storage.get(&interval).ok_or(GpuError::InvalidInterval)?;

		interval_map
			.get(&handle_id)
			.cloned()
			.ok_or(GpuError::InvalidHandle)
	}

	pub(crate) fn memory_handles_interval(
		&self,
		interval: &Interval,
	) -> Result<HashMap<usize, GpuMemRep>> {
		let storage = self.storage.lock().map_err(|_| GpuError::LockError)?;

		let interval_map = storage.get(&interval).ok_or(GpuError::InvalidInterval)?;

		Ok(interval_map.clone())
	}
}
