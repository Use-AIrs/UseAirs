// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use super::super::base::*;
use cubecl::{
	prelude::*, tensor_line_size, tensor_line_size_parallel, tensor_line_size_perpendicular,
};
use std::mem::size_of;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ComptimeCfg {
	pub line_size: u32,
	pub row_size: u32,
	pub warp_size: u32,
	pub shmem_len: u32,
	pub cycles_per_axis: u32,
}

impl ComptimeCfg {
	pub fn new<R: Runtime, N: Numeric>(
		client: &ComputeClient<R::Server>,
		input: &TensorHandleRef<R>,
		output: &TensorHandleRef<R>,
		axis: usize,
	) -> (Self, CubeCount) {
		let props = client.properties();
		let warp_size = props.hardware.plane_size_max;
		let shmem_available = props.hardware.max_shared_memory_size as u32;

		let row_size = input.shape[axis];
		let num_rows: usize = input
			.shape
			.iter()
			.enumerate()
			.filter(|(i, _)| *i != axis)
			.map(|(_, &s)| s)
			.product::<usize>()
			.max(1);

		let line_size = Self::ln_perp::<R, N>(input, output, axis);
		let row_size_padded = (row_size as u32).next_power_of_two();

		let argline_size = line_size * (size_of::<N>() as u32 + size_of::<u32>() as u32);
		let max_elements_by_shmem = shmem_available / argline_size;
		let max_threads: u32 = 512;
		let max_elements = Ord::min(max_elements_by_shmem, max_threads);

		let (shmem_len, cycles_per_axis) = if row_size_padded <= max_elements {
			let shmem_len = Ord::max(row_size_padded, warp_size);
			(shmem_len, 1)
		} else {
			let shmem_len: u32 = if max_elements.is_power_of_two() {
				max_elements
			} else {
				max_elements.next_power_of_two() / 2
			};
			let cycles = (row_size_padded + shmem_len - 1) / shmem_len;
			(shmem_len, cycles)
		};

		let cfg = Self {
			line_size,
			row_size: row_size as u32,
			warp_size,
			shmem_len,
			cycles_per_axis,
		};

		let num_rows_f = num_rows as f32;
		let cube_count_x = f32::ceil(f32::sqrt(num_rows_f)) as u32;
		let cube_count_y = f32::ceil(num_rows_f / cube_count_x as f32) as u32;
		let cube_count = CubeCount::Static(cube_count_x, cube_count_y, 1);

		(cfg, cube_count)
	}

	fn ln_perp<R: Runtime, N: Numeric>(
		input: &TensorHandleRef<R>,
		output: &TensorHandleRef<R>,
		axis: usize,
	) -> u32 {
		let supported_line_sizes = R::io_optimized_line_sizes_unchecked(size_of::<N>());

		let mut input_axis_and_strides: Vec<_> = input.strides.iter().enumerate().collect();
		input_axis_and_strides.sort_by_key(|(_, stride)| *stride);
		let input_sorted_axis = input_axis_and_strides
			.into_iter()
			.map(|(a, _)| a)
			.take_while(|a| *a != axis);

		let mut output_axis_and_strides: Vec<_> = output.strides.iter().enumerate().collect();
		output_axis_and_strides.sort_by_key(|(_, stride)| *stride);
		let output_sorted_axis = output_axis_and_strides
			.into_iter()
			.filter_map(|(a, _)| (a != axis).then_some(a));

		let max_line_size: usize = input_sorted_axis
			.zip(output_sorted_axis)
			.filter_map(|(i, o)| (i == o).then_some(output.shape[i]))
			.product::<usize>()
			.max(1);

		tensor_line_size_perpendicular(
			supported_line_sizes.filter(|size| {
				(*size as usize) <= max_line_size && max_line_size % (*size as usize) == 0
			}),
			input.shape,
			input.strides,
			axis,
		) as u32
	}
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum SortStrategy {
	WarpOnly {
		warp_steps: u32,
		line_size: u32,
	},
	SharedMem {
		warp_steps: u32,
		shmem_steps: u32,
		log_shmem_len: u32,
		line_size: u32,
	},
}

impl SortStrategy {
	pub fn from_cfg(cfg: &ComptimeCfg) -> Self {
		let warp_steps = cfg.warp_size.ilog2();
		let warp_capacity = cfg.warp_size * cfg.line_size;

		let line_size = if cfg.row_size < cfg.warp_size {
			1
		} else if cfg.row_size <= 2 * cfg.warp_size {
			2
		} else {
			4
		};

		if cfg.row_size <= warp_capacity {
			SortStrategy::WarpOnly {
				warp_steps,
				line_size,
			}
		} else {
			let shmem_steps = cfg.shmem_len.ilog2();
			let log_shmem_len = cfg.shmem_len.ilog2();
			SortStrategy::SharedMem {
				warp_steps,
				shmem_steps,
				log_shmem_len,
				line_size,
			}
		}
	}
}
