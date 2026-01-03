// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use cubecl::prelude::*;

use crate::operation::gpu::kernels::sort::cfg::ComptimeCfg;
use crate::operation::gpu::kernels::sort::range::{k0_ranger, K0Range, SortPhase};

#[cube]
pub fn sort_arged<N: Numeric>(
	input: &Slice<Line<N>>,
	output_data: &mut Slice<Line<N>, ReadWrite>,
	output_args: &mut Slice<Line<u32>, ReadWrite>,
	#[comptime] cfg: ComptimeCfg,
	#[comptime] ascending: u32,
) {
	let range = k0_ranger(SortPhase::IntroSort, cfg, ascending);

	let shmem_len = comptime!(cfg.shmem_len);
	let line_size = comptime!(cfg.line_size);
	let mut shmem_data = SharedMemory::<N>::new_lined(shmem_len, line_size);
	let mut shmem_args = SharedMemory::<u32>::new_lined(shmem_len, line_size);

	load_chunk_to_shmem_arged::<N>(
		input,
		&mut shmem_data,
		&mut shmem_args,
		&range,
		cfg,
	);
	sync_cube();

	sort_shmem_bitonic_arged::<N>(
		&mut shmem_data,
		&mut shmem_args,
		&range,
		cfg,
	);
	sync_cube();

	write_shmem_to_chunk_arged::<N>(
		&shmem_data,
		&shmem_args,
		output_data,
		output_args,
		&range,
		cfg,
	);
}

#[cube]
fn load_chunk_to_shmem_arged<N: Numeric>(
	input: &Slice<Line<N>>,
	shmem_data: &mut SharedMemory<Line<N>>,
	shmem_args: &mut SharedMemory<Line<u32>>,
	range: &K0Range,
	#[comptime] cfg: ComptimeCfg,
) {
	let shmem_len = comptime!(cfg.shmem_len);
	let line_size = comptime!(cfg.line_size);
	let total_lines = comptime!(cfg.row_size);

	let thread_id = UNIT_POS_X;
	let global_idx = range.chunk_offset + thread_id;

	let sentinel = select(
		range.ascending(),
		N::max_value(),
		N::min_value(),
	);
	let sentinel_line = Line::<N>::new(sentinel);

	if thread_id < shmem_len {
		let valid = global_idx < total_lines;
		let safe_idx = global_idx * u32::cast_from(valid);

		shmem_data[thread_id] = select(valid, input[safe_idx], sentinel_line);

		let base_idx = global_idx * line_size;
		let mut idx_line = Line::<u32>::empty(4u32);
		#[unroll]
		for i in 0..line_size {
			idx_line[i] = base_idx + i;
		}
		shmem_args[thread_id] = idx_line;
	}
}

#[cube]
fn write_shmem_to_chunk_arged<N: Numeric>(
	shmem_data: &SharedMemory<Line<N>>,
	shmem_args: &SharedMemory<Line<u32>>,
	output_data: &mut Slice<Line<N>, ReadWrite>,
	output_args: &mut Slice<Line<u32>, ReadWrite>,
	range: &K0Range,
	#[comptime] cfg: ComptimeCfg,
) {
	let shmem_len = comptime!(cfg.shmem_len);
	let total_lines = comptime!(cfg.row_size);

	let thread_id = UNIT_POS_X;
	let global_idx = range.chunk_offset + thread_id;

	if thread_id < shmem_len && global_idx < total_lines {
		output_data[global_idx] = shmem_data[thread_id];
		output_args[global_idx] = shmem_args[thread_id];
	}
}

#[cube]
fn sort_shmem_bitonic_arged<N: Numeric>(
	shmem_data: &mut SharedMemory<Line<N>>,
	shmem_args: &mut SharedMemory<Line<u32>>,
	range: &K0Range,
	#[comptime] cfg: ComptimeCfg,
) {
	let shmem_len = comptime!(cfg.shmem_len);
	let line_size = comptime!(cfg.line_size);
	let num_steps = comptime!(shmem_len.ilog2());

	let thread_id = UNIT_POS_X;

	if thread_id < shmem_len {
		intra_line_sort_arged::<N>(
			shmem_data,
			shmem_args,
			thread_id,
			range.is_ascending,
			line_size,
		);
	}
	sync_cube();

	#[unroll]
	for stage in 0..num_steps {
		#[unroll]
		for substep in 0..num_steps {
			if comptime!(substep <= stage) {
				let merge_step = comptime!(stage - substep);
				bitonic_merge_step_arged::<N>(
					shmem_data,
					shmem_args,
					thread_id,
					stage,
					merge_step,
					range.is_ascending,
					cfg,
				);
				sync_cube();
			}
		}
	}

	if thread_id < shmem_len {
		intra_line_sort_arged::<N>(
			shmem_data,
			shmem_args,
			thread_id,
			range.is_ascending,
			line_size,
		);
	}
}

#[cube]
fn bitonic_merge_step_arged<N: Numeric>(
	shmem_data: &mut SharedMemory<Line<N>>,
	shmem_args: &mut SharedMemory<Line<u32>>,
	thread_id: u32,
	#[comptime] stage: u32,
	#[comptime] merge_step: u32,
	chunk_ascending: u32,
	#[comptime] cfg: ComptimeCfg,
) {
	let shmem_len = comptime!(cfg.shmem_len);
	let stride = comptime!(1u32 << merge_step);
	let block_size = comptime!(1u32 << (stage + 1));

	let partner = thread_id ^ stride;

	if partner < shmem_len && thread_id < partner {
		let block_id = thread_id / block_size;

		let block_ascending = select(
			chunk_ascending == 1,
			(block_id % 2u32) == 0u32,
			(block_id % 2u32) != 0u32,
		);

		let my_data = shmem_data[thread_id];
		let partner_data = shmem_data[partner];
		let my_args = shmem_args[thread_id];
		let partner_args = shmem_args[partner];

		let my_key = my_data[0u32];
		let partner_key = partner_data[0u32];

		let should_swap = select(
			block_ascending,
			my_key > partner_key,
			my_key < partner_key,
		);

		if should_swap {
			shmem_data[thread_id] = partner_data;
			shmem_data[partner] = my_data;
			shmem_args[thread_id] = partner_args;
			shmem_args[partner] = my_args;
		}
	}
}

#[cube]
fn intra_line_sort_arged<N: Numeric>(
	shmem_data: &mut SharedMemory<Line<N>>,
	shmem_args: &mut SharedMemory<Line<u32>>,
	thread_id: u32,
	ascending: u32,
	#[comptime] line_size: u32,
) {
	let mut data = shmem_data[thread_id];
	let mut args = shmem_args[thread_id];

	let mut a = data[0u32];
	let mut b = data[1u32];
	let mut c = data[2u32];
	let mut d = data[3u32];

	let mut ia = args[0u32];
	let mut ib = args[1u32];
	let mut ic = args[2u32];
	let mut id = args[3u32];

	if ascending == 1 {
		if a > b {
			let t = a;
			a = b;
			b = t;
			let ti = ia;
			ia = ib;
			ib = ti;
		}
		if c > d {
			let t = c;
			c = d;
			d = t;
			let ti = ic;
			ic = id;
			id = ti;
		}

		if a > c {
			let t = a;
			a = c;
			c = t;
			let ti = ia;
			ia = ic;
			ic = ti;
		}
		if b > d {
			let t = b;
			b = d;
			d = t;
			let ti = ib;
			ib = id;
			id = ti;
		}

		if b > c {
			let t = b;
			b = c;
			c = t;
			let ti = ib;
			ib = ic;
			ic = ti;
		}
	} else {
		if a < b {
			let t = a;
			a = b;
			b = t;
			let ti = ia;
			ia = ib;
			ib = ti;
		}
		if c < d {
			let t = c;
			c = d;
			d = t;
			let ti = ic;
			ic = id;
			id = ti;
		}

		if a < c {
			let t = a;
			a = c;
			c = t;
			let ti = ia;
			ia = ic;
			ic = ti;
		}
		if b < d {
			let t = b;
			b = d;
			d = t;
			let ti = ib;
			ib = id;
			id = ti;
		}

		if b < c {
			let t = b;
			b = c;
			c = t;
			let ti = ib;
			ib = ic;
			ic = ti;
		}
	}

	data[0u32] = a;
	data[1u32] = b;
	data[2u32] = c;
	data[3u32] = d;

	args[0u32] = ia;
	args[1u32] = ib;
	args[2u32] = ic;
	args[3u32] = id;

	shmem_data[thread_id] = data;
	shmem_args[thread_id] = args;
}

#[cube]
pub fn to_output_single_arged<N: Numeric>(
	shmem_data: &SharedMemory<Line<N>>,
	shmem_args: &SharedMemory<Line<u32>>,
	output_data: &mut Slice<Line<N>, ReadWrite>,
	output_args: &mut Slice<Line<u32>, ReadWrite>,
	range: &K0Range,
	#[comptime] cfg: ComptimeCfg,
) {
	let shmem_len = comptime!(cfg.shmem_len);
	let total_lines = comptime!(cfg.row_size);

	let thread_id = UNIT_POS_X;
	let global_idx = range.chunk_offset + thread_id;

	if thread_id < shmem_len && global_idx < total_lines {
		let mut data = shmem_data[thread_id];
		let mut args = shmem_args[thread_id];

		let mut a = data[0u32];
		let mut b = data[1u32];
		let mut c = data[2u32];
		let mut d = data[3u32];

		let mut ia = args[0u32];
		let mut ib = args[1u32];
		let mut ic = args[2u32];
		let mut id = args[3u32];

		if range.ascending() {
			if a > b {
				let t = a;
				a = b;
				b = t;
				let ti = ia;
				ia = ib;
				ib = ti;
			}
			if c > d {
				let t = c;
				c = d;
				d = t;
				let ti = ic;
				ic = id;
				id = ti;
			}
			if a > c {
				let t = a;
				a = c;
				c = t;
				let ti = ia;
				ia = ic;
				ic = ti;
			}
			if b > d {
				let t = b;
				b = d;
				d = t;
				let ti = ib;
				ib = id;
				id = ti;
			}
			if b > c {
				let t = b;
				b = c;
				c = t;
				let ti = ib;
				ib = ic;
				ic = ti;
			}
		} else {
			if a < b {
				let t = a;
				a = b;
				b = t;
				let ti = ia;
				ia = ib;
				ib = ti;
			}
			if c < d {
				let t = c;
				c = d;
				d = t;
				let ti = ic;
				ic = id;
				id = ti;
			}
			if a < c {
				let t = a;
				a = c;
				c = t;
				let ti = ia;
				ia = ic;
				ic = ti;
			}
			if b < d {
				let t = b;
				b = d;
				d = t;
				let ti = ib;
				ib = id;
				id = ti;
			}
			if b < c {
				let t = b;
				b = c;
				c = t;
				let ti = ib;
				ib = ic;
				ic = ti;
			}
		}

		let mut out_data = Line::<N>::empty(4u32);
		let mut out_args = Line::<u32>::empty(4u32);
		out_data[0u32] = a;
		out_data[1u32] = b;
		out_data[2u32] = c;
		out_data[3u32] = d;
		out_args[0u32] = ia;
		out_args[1u32] = ib;
		out_args[2u32] = ic;
		out_args[3u32] = id;

		output_data[global_idx] = out_data;
		output_args[global_idx] = out_args;
	}
}

#[cube]
pub fn to_output_pingpong_arged<N: Numeric>(
	shmem_data: &SharedMemory<Line<N>>,
	shmem_args: &SharedMemory<Line<u32>>,
	output_data: &mut Slice<Line<N>, ReadWrite>,
	output_args: &mut Slice<Line<u32>, ReadWrite>,
	range: &K0Range,
	#[comptime] cfg: ComptimeCfg,
) {
	let shmem_len = comptime!(cfg.shmem_len);
	let total_lines = comptime!(cfg.row_size);

	let thread_id = UNIT_POS_X;
	let global_idx = range.chunk_offset + thread_id;

	if thread_id < shmem_len && global_idx < total_lines {
		let mut data = shmem_data[thread_id];
		let mut args = shmem_args[thread_id];

		let mut a = data[0u32];
		let mut b = data[1u32];
		let mut c = data[2u32];
		let mut d = data[3u32];

		let mut ia = args[0u32];
		let mut ib = args[1u32];
		let mut ic = args[2u32];
		let mut id = args[3u32];

		if range.ascending() {
			if a > b {
				let t = a;
				a = b;
				b = t;
				let ti = ia;
				ia = ib;
				ib = ti;
			}
			if c > d {
				let t = c;
				c = d;
				d = t;
				let ti = ic;
				ic = id;
				id = ti;
			}
			if a > c {
				let t = a;
				a = c;
				c = t;
				let ti = ia;
				ia = ic;
				ic = ti;
			}
			if b > d {
				let t = b;
				b = d;
				d = t;
				let ti = ib;
				ib = id;
				id = ti;
			}
			if b > c {
				let t = b;
				b = c;
				c = t;
				let ti = ib;
				ib = ic;
				ic = ti;
			}
		} else {
			if a < b {
				let t = a;
				a = b;
				b = t;
				let ti = ia;
				ia = ib;
				ib = ti;
			}
			if c < d {
				let t = c;
				c = d;
				d = t;
				let ti = ic;
				ic = id;
				id = ti;
			}
			if a < c {
				let t = a;
				a = c;
				c = t;
				let ti = ia;
				ia = ic;
				ic = ti;
			}
			if b < d {
				let t = b;
				b = d;
				d = t;
				let ti = ib;
				ib = id;
				id = ti;
			}
			if b < c {
				let t = b;
				b = c;
				c = t;
				let ti = ib;
				ib = ic;
				ic = ti;
			}
		}

		let mut out_data = Line::<N>::empty(4u32);
		let mut out_args = Line::<u32>::empty(4u32);
		out_data[0u32] = a;
		out_data[1u32] = b;
		out_data[2u32] = c;
		out_data[3u32] = d;
		out_args[0u32] = ia;
		out_args[1u32] = ib;
		out_args[2u32] = ic;
		out_args[3u32] = id;

		output_data[global_idx] = out_data;
		output_args[global_idx] = out_args;
	}
}
