// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use cubecl::prelude::*;
use cubecl_std::CubeOption;

use super::io::ArgLine;

#[cube]
pub trait SharedAccumulator: CubeType + Send + Sync + 'static {
	type Item: CubeType;

	fn allocate(
		#[comptime] length: u32,
		#[comptime] line_size: u32,
		#[comptime] _coordinate: bool,
	) -> Self;

	fn read(
		accumulator: &Self,
		index: u32,
	) -> Self::Item;

	fn write(
		accumulator: &mut Self,
		index: u32,
		item: Self::Item,
	);
}

#[cube]
impl<In: Numeric> SharedAccumulator for SharedMemory<Line<In>> {
	type Item = Line<In>;

	fn allocate(
		#[comptime] length: u32,
		#[comptime] line_size: u32,
		#[comptime] _coordinate: bool,
	) -> Self {
		SharedMemory::new_lined(length, line_size)
	}

	fn read(
		accumulator: &Self,
		index: u32,
	) -> Self::Item {
		accumulator[index]
	}

	fn write(
		accumulator: &mut Self,
		index: u32,
		item: Self::Item,
	) {
		accumulator[index] = item;
	}
}

/// A pair of shared memory used for [`ArgMax`](super::ArgMax) and [`ArgMin`](super::ArgMin).
#[derive(CubeType)]
pub struct ArgAccumulator<N: Numeric> {
	pub elements: SharedMemory<Line<N>>,
	pub args: SharedMemory<Line<u32>>,
}

#[cube]
impl<In: Numeric> SharedAccumulator for ArgAccumulator<In> {
	type Item = (Line<In>, Line<u32>);

	fn allocate(
		#[comptime] length: u32,
		#[comptime] line_size: u32,
		#[comptime] _coordinate: bool,
	) -> Self {
		ArgAccumulator::<In> {
			elements: SharedMemory::new_lined(length, line_size),
			args: SharedMemory::new_lined(length, line_size),
		}
	}

	fn read(
		accumulator: &Self,
		index: u32,
	) -> Self::Item {
		(
			accumulator.elements[index],
			accumulator.args[index],
		)
	}

	fn write(
		accumulator: &mut Self,
		index: u32,
		item: Self::Item,
	) {
		accumulator.elements[index] = item.0;
		accumulator.args[index] = item.1;
	}
}
