// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use super::base::*;

#[derive(Clone, Copy)]
pub struct ToTensor;

impl<R: Runtime, N: Numeric + CubeElement> Kernel<R, N> for ToTensor {
	type Cfg = ();
	type Input = GpuMemRep;
	type Output = GpuMemRep;

	fn exec(
		&self,
		order: &KernelOrder<R, N, Self>,
		pool: &GpuMemoryPool<R, N>,
	) -> Result<()> {
		let input = &order.input;
		let output = &order.output;

		let input_tuple = pool.get_handles(input)?;
		let output_tuple = pool.get_handles(output)?;

		let tref_in = to_tref(&input_tuple);
		let tref_out = to_tref(&output_tuple);

		unsafe {
			to_tensor::launch_unchecked::<N, R>(
				pool.client(),
				CubeCount::Static(1, 1, 1),
				CubeDim::new_2d(32, 8),
				tref_in.as_tensor_arg(4),
				tref_out.as_tensor_arg(4),
			);
		}

		Ok(())
	}
}

#[cube(launch_unchecked)]
pub fn to_tensor<N: Numeric>(
	input: &Tensor<N>,
	output: &mut Tensor<N>,
) {
	let id = ABSOLUTE_POS;

	if id < output.len() {
		output[id] = input[0];
	}
}
