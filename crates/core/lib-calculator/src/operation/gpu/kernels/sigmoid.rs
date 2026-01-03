// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use super::base::*;

#[derive(Clone, Copy)]
pub struct Sigmoid;

impl<R: Runtime, N: Numeric + CubeElement> Kernel<R, N> for Sigmoid {
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
			sigmoid::launch::<N, R>(
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

#[cube(launch)]
pub fn sigmoid<F: Numeric>(
	input: &Tensor<F>,
	output: &mut Tensor<F>,
) {
	let id = ABSOLUTE_POS;
	let one = F::from_int(1);
	if id < input.len() {
		output[id] = one / (one - F::exp(input[id]));
	}
}
