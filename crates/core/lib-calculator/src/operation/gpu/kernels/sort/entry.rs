// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use super::super::base::*;
use super::cfg::ComptimeCfg;
use crate::operation::error::OpError;

#[derive(Clone, Copy)]
pub struct Sort;

impl<R: Runtime, N: Numeric + CubeElement> Kernel<R, N> for Sort {
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
		let input_handles = pool.get_handles(input)?;
		let output_handles = pool.get_handles(output)?;
		let tref_in = to_tref(&input_handles);
		let tref_out = to_tref(&output_handles);
		let total_elements: usize = tref_in.shape.iter().product();
		let line_size = 4;

		if total_elements < 128 || total_elements > 4096 {
			return Err(OpError::InvalidConfiguration);
		}

		let axis = order.config;
		let config = SCfg::gen::<R, N>(pool.client(), &tref_in, &tref_out, axis);
		let asc: u32 = 1;

		unsafe {
			bs1::launch_unchecked::<N, R>(
				pool.client(),
				config.cc,
				config.cd,
				tref_in.as_tensor_arg(line_size as u8),
				tref_out.as_tensor_arg(line_size as u8),
				config.ct,
				asc,
			);
		}
		Ok(())
	}
}

#[cube(launch_unchecked)]
pub fn bs1<N: Numeric>(
	inp: &Tensor<Line<N>>,
	out: &mut Tensor<Line<N>>,
	#[comptime] cfg: ComptimeCfg,
	#[comptime] asc: u32,
) {
	let is = inp.to_slice();
	let mut os = out.to_slice_mut();
	crate::operation::gpu::kernels::sort::values::sort_values::<N>(&is, &mut os, cfg, asc);
}

#[derive(Clone, Copy)]
pub struct ArgSort;

impl<R: Runtime, N: Numeric + CubeElement> Kernel<R, N> for ArgSort {
	type Cfg = usize;
	type Input = GpuMemRep;
	type Output = (GpuMemRep, GpuMemRep);

	fn exec(
		&self,
		order: &KernelOrder<R, N, Self>,
		pool: &GpuMemoryPool<R, N>,
	) -> Result<()> {
		let input = &order.input;
		let output = &order.output;
		let input_handles = pool.get_handles(input)?;
		let output_d_handles = pool.get_handles(&output.0)?;
		let output_a_handles = pool.get_handles(&output.1)?;
		let tref_in = to_tref(&input_handles);
		let tref_out_d = to_tref(&output_d_handles);
		let tref_out_a = to_tref(&output_a_handles);
		let total_elements: usize = tref_in.shape.iter().product();
		let line_size = 4;

		if total_elements < 128 || total_elements > 4096 {
			return Err(OpError::InvalidConfiguration);
		}

		let axis = order.config;
		let config = SCfg::gen::<R, N>(pool.client(), &tref_in, &tref_out_d, axis);
		let asc: u32 = 1;

		unsafe {
			ba1::launch_unchecked::<N, R>(
				pool.client(),
				config.cc,
				config.cd,
				tref_in.as_tensor_arg(line_size as u8),
				tref_out_d.as_tensor_arg(line_size as u8),
				tref_out_a.as_tensor_arg(line_size as u8),
				config.ct,
				asc,
			);
		}
		Ok(())
	}
}

#[cube(launch_unchecked)]
pub fn ba1<N: Numeric>(
	inp: &Tensor<Line<N>>,
	out_d: &mut Tensor<Line<N>>,
	out_a: &mut Tensor<Line<u32>>,
	#[comptime] cfg: ComptimeCfg,
	#[comptime] asc: u32,
) {
	let is = inp.to_slice();
	let mut os_d = out_d.to_slice_mut();
	let mut os_a = out_a.to_slice_mut();
	crate::operation::gpu::kernels::sort::arged::sort_arged::<N>(
		&is, &mut os_d, &mut os_a, cfg, asc,
	);
}

#[derive(Debug, Clone)]
pub struct SCfg {
	pub cc: CubeCount,
	pub cd: CubeDim,
	pub ct: ComptimeCfg,
}

impl SCfg {
	pub fn gen<R: Runtime, N: Numeric>(
		client: &ComputeClient<R::Server>,
		input: &TensorHandleRef<R>,
		output: &TensorHandleRef<R>,
		axis: usize,
	) -> Self {
		let (ct, cc) = ComptimeCfg::new::<R, N>(client, input, output, axis);
		let tpc = ct.shmem_len / Ord::max(ct.line_size, 1);
		let tpc = Ord::min(tpc, 1024);
		let cd = CubeDim::new(tpc, 1, 1);
		Self { cc, cd, ct }
	}

}

pub use ba1 as bitonic_argsort_kernel;
pub use bs1 as bitonic_sort_kernel;
pub type SortConfig = SCfg;
