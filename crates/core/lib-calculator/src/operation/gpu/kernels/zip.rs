// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use super::base::*;

#[derive(Clone, Copy)]
pub struct Zip;

impl<R: Runtime, N: Numeric + CubeElement> Kernel<R, N> for Zip {
	type Cfg = ();
	type Input = (GpuMemRep, GpuMemRep);
	type Output = GpuMemRep;

	fn exec(
		&self,
		order: &KernelOrder<R, N, Self>,
		pool: &GpuMemoryPool<R, N>,
	) -> Result<()> {
		let input = &order.input;
		let output = &order.output;

		let input_a_tuple = pool.get_handles(&input.0)?;
		let input_b_tuple = pool.get_handles(&input.1)?;
		let output_tuple = pool.get_handles(output)?;

		let tref_a = to_tref(&input_a_tuple);
		let tref_b = to_tref(&input_b_tuple);
		let tref_out = to_tref(&output_tuple);

		unsafe {
			zip::launch_unchecked::<N, R>(
				pool.client(),
				CubeCount::Static(1, 1, 1),
				CubeDim::new_2d(32, 8),
				tref_a.as_tensor_arg(4),
				tref_b.as_tensor_arg(4),
				tref_out.as_tensor_arg(4),
			);
		}

		Ok(())
	}
}

#[cube(launch_unchecked)]
pub fn zip<N: Numeric>(
	input_a: &Tensor<N>,
	input_b: &Tensor<N>,
	output: &mut Tensor<N>,
) {
	let id = ABSOLUTE_POS;
	let cut = input_a.len();

	if id < output.len() {
		if id < cut {
			output[id] = input_a[id];
		} else {
			output[id] = input_b[id - cut];
		}
	}
}
