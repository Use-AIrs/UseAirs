// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use cubecl_core::calculate_cube_count_elemwise;

use super::base::*;

#[derive(Clone, Copy)]
pub struct Padding;

impl<R: Runtime, N: Numeric + CubeElement> Kernel<R, N> for Padding {
	type Cfg = usize;
	type Input = GpuMemRep;
	type Output = GpuMemRep;

	fn exec(
		&self,
		order: &KernelOrder<R, N, Self>,
		pool: &GpuMemoryPool<R, N>,
	) -> Result<()> {
		let input = &order.input;
		let output = &order.output;

		let (rep, handle) = pool.get_handles(input)?;
		let padding_bytes = rep.row_padding_bytes::<N>();
		let padded_handle = handle.clone().offset_end(padding_bytes as u64);
		let padded_mem = rep.with_row_padding();
		let input_tuple = (padded_mem.clone(), padded_handle);
		let output_tuple = pool.get_handles(output)?;
		let md = padded_mem.metadata;
		let dim = CubeDim::new_2d(md.shape[0] as u32, md.shape[1] as u32);
		let count = calculate_cube_count_elemwise(padded_mem.size, dim);
		println!("{:?}", count);

		let tref_in = to_tref(&input_tuple);
		let tref_out = to_tref(&output_tuple);
		unsafe {
			padding::launch_unchecked::<N, R>(
				pool.client(),
				count,
				CubeDim::new_2d(32, 8),
				tref_in.as_tensor_arg(4),
				tref_out.as_tensor_arg(4),
				ScalarArg::new(padding_bytes as u32 / size_of::<N>() as u32),
			);
		}

		Ok(())
	}
}

#[cube(launch_unchecked)]
pub fn padding<N: Numeric>(
	input: &Tensor<Line<N>>,
	output: &mut Tensor<Line<N>>,
	padding: u32,
) {
	let offset_start = input.len() - padding;
	let id = UNIT_POS_X;
	if id < offset_start {
		output[id] = input[id];
	} else {
		output[id] = input[id].fill(N::max_value())
	}
}
