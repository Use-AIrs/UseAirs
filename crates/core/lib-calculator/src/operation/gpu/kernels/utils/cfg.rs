// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use cubecl_core::{
	prelude::*, server::ComputeServer, tensor_line_size_parallel, tensor_line_size_perpendicular,
};
use cubecl_runtime::DeviceProperties;
use cubecl_std::tensor::is_contiguous;
use ndarray::Axis;
use std::mem::size_of;

use crate::Kernel;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]

pub enum BoundChecksInner {
	None,

	Mask,

	Branch,
}

#[derive(Debug, Clone)]
pub struct Config {
	pub cube_count: CubeCount,
	pub cube_dim: CubeDim,
	pub line_size: u32,
	pub bound_checks: bool,
	pub bound_checks_inner: BoundChecksInner,
	pub warp_size: u32,
	pub shared_size: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ComptimeCfg {
	pub line_size_input: u32,
	pub line_size_output: u32,
	pub warp_size: u32,
	pub shared_size: u32,
}

impl From<Config> for ComptimeCfg {
	fn from(value: Config) -> Self {
		Self {
			line_size_input: value.line_size,
			line_size_output: value.line_size,
			warp_size: value.warp_size,
			shared_size: value.shared_size,
		}
	}
}

impl Config {
	pub fn new<R: Runtime>(client: &ComputeClient<R::Server>) -> Self {
		let prop = client.properties();

		Self {
			cube_count: CubeCount::new_single(),
			cube_dim: CubeDim::new_single(),
			line_size: R::max_global_line_size() as u32,
			bound_checks: true,
			bound_checks_inner: BoundChecksInner::Mask,
			warp_size: prop.hardware.plane_size_max,
			shared_size: prop.hardware.max_shared_memory_size as u32,
		}
	}

	pub fn axis_cubes<R: Runtime>(
		mut self,
		client: &ComputeClient<R::Server>,
		input: TensorHandleRef<R>,
		axis: usize,
		pow2_axis: bool,
	) -> Self {
		let hprop = client.properties().hardware.clone();
		let mut max_units = hprop.max_units_per_cube;
		let mut tshape = input.shape.to_vec();
		let mut axis_size = tshape.split_off(axis)[0] as u32;
		let axis_cnt: usize = tshape.iter().product();
		let axis_count = axis_cnt as u32;
		if pow2_axis {
			axis_size = axis_size.next_power_of_two();
		}
		let (xdim, xcnt) = if axis_size < max_units {
			(axis_size, axis_count)
		} else {
			let xcount = (axis_size as f64 / max_units as f64).ceil() as u32;
			(max_units, xcount)
		};
		self.cube_dim = CubeDim::new(xdim, 1, 1);
		self.cube_count = CubeCount::new_1d(xcnt);
		self
	}
}

pub(crate) struct CubeEnv {
	pub cubes: CubeCount,
	pub dim: CubeDim,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct AxisOpCfg {
	left_cubes: (u32, u32, u32),
	left_dim: (u32, u32, u32),
	left_dim_size: u32,
	left_sh: u32,
	plane_size_max: u32,

	pub line_size: u32,
	pub axis_size: u32,
	pub axis_count: u32,
	pub axis_per_sh: u32,
	pub warps: u32,
	pub cycles_per_axis: u32,
}

impl AxisOpCfg {
	pub fn new(props: &DeviceProperties) -> Self {
		let max_cube = &props.hardware.max_cube_count;
		let max_dim = props.hardware.max_cube_dim;

		let (cube_x, cube_y, cube_z) = match max_cube {
			CubeCount::Static(x, y, z) => (*x, *y, *z),
			CubeCount::Dynamic(_) => (u32::MAX, u32::MAX, u32::MAX),
		};
		let left_cubes = (cube_x, cube_y, cube_z);
		let left_dim = (max_dim.x, max_dim.y, max_dim.z);
		let left_dim_size = props.hardware.max_units_per_cube;
		let left_sh = ((props.hardware.max_shared_memory_size as f64) * 0.9) as u32;
		let plane_size_max = props.hardware.plane_size_max;

		AxisOpCfg {
			left_cubes,
			left_dim,
			left_dim_size,
			left_sh,
			plane_size_max,

			line_size: 1,
			axis_size: 0,
			axis_count: 0,
			axis_per_sh: 1,
			warps: 0,
			cycles_per_axis: 1,
		}
	}

	pub fn padded_axis<R: Runtime>(
		mut self,
		in_tref: TensorHandleRef<R>,
		axis: usize,
	) -> (Self, CubeEnv) {
		let target_axis = axis.saturating_sub(1) as u32;
		let mut current_axis = 0u32;
		let (axis_size, axis_count) = in_tref.shape.iter().fold(
			(0u32, 1u32),
			|(axis_size, axis_count), &x| {
				let result = if current_axis == target_axis {
					(axis_size + x as u32, axis_count)
				} else {
					(axis_size, axis_count * x as u32)
				};
				current_axis += 1;
				result
			},
		);
		let axis_padded = axis_size.next_power_of_two();
		self.line_size = R::max_global_line_size() as u32;
		self.axis_size = axis_size;
		self.axis_count = axis_count;

		let cube_factor = self.left_dim_size as f32 / axis_padded as f32;

		let strat = if cube_factor < 0.5f32 {
			let axis_per_cube = (1.0 / cube_factor).ceil() as u32;
			let x = (axis_count as f64 / axis_per_cube as f64).floor() as u32;
			self.left_cubes.0 = self.left_cubes.0.saturating_sub(x);
			self.left_dim_size = self
				.left_dim_size
				.saturating_sub(axis_per_cube * axis_padded);
			self.left_dim.0 = self.left_dim.0.saturating_sub(axis_per_cube);
			self.left_dim.1 = self.left_dim.1.saturating_sub(axis_padded);
			(
				CubeDim::new_3d(axis_per_cube, axis_padded, 1),
				CubeCount::Static(x, 1, 1),
			)
		} else {
			let all = cube_factor.ceil() as u32;
			self.left_cubes.0 = self.left_cubes.0.saturating_sub(all);
			self.left_dim_size = 0;

			let forward = (
				CubeDim::new_3d(1, axis_padded, 1),
				CubeCount::Static(all, 1, 1),
			);
			self.left_dim.0 = 0;
			self.left_dim.1 = 0;
			self.left_dim.2 = 0;
			forward
		};
		let env = CubeEnv {
			cubes: strat.1,
			dim: strat.0,
		};
		(self, env)
	}
}
