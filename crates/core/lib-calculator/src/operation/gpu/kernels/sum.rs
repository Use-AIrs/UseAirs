// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use super::base::*;

#[derive(Clone, Copy)]
pub struct Sum;

impl<R: Runtime, N: Numeric + CubeElement> Kernel<R, N> for Sum {
	type Cfg = usize;
	type Input = GpuMemRep;
	type Output = GpuMemRep;

	fn exec(
		&self,
		order: &KernelOrder<R, N, Self>,
		pool: &GpuMemoryPool<R, N>,
	) -> Result<()> {
		let cfg = &order.config;
		let input = &order.input;
		let output = &order.output;

		let input_tuple = pool.get_handles(input)?;
		let output_tuple = pool.get_handles(output)?;

		let tref_in = to_tref(&input_tuple);
		let tref_out = to_tref(&output_tuple);

		reduce::<R, (N, N), N, cubecl_reduce::instructions::Sum>(
			pool.client(),
			tref_in,
			tref_out,
			*cfg,
			None,
			(),
		)
		.map_err(|_| crate::operation::error::OpError::ExecutionError)?;

		Ok(())
	}
}
