// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use cubecl::prelude::*;

use crate::operation::gpu::kernels::base::Lined;

#[derive(CubeType, Clone, Copy)]
pub struct ArgLine<N: Numeric + CubePrimitive> {
	pub data: Line<N>,
	pub args: Line<u32>,
}

#[cube]
impl<N: Numeric> ArgLine<N> {
	pub fn new(
		data: Line<N>,
		args: Line<u32>,
	) -> Self {
		ArgLine::<N> { data, args }
	}
}

#[cube]
impl<N: Numeric> Lined for ArgLine<N> {}

impl<N: CubePrimitive + Numeric> LinedExpand for ArgLineExpand<N> {
	fn line_size(&self) -> u32 {
		self.data.line_size()
	}
}

#[derive(CubeType)]
pub struct TupledOutputIdxIO<In: Numeric, Out: Numeric> {
	input: *const Tensor<Line<In>>,
	output: *mut Tensor<Line<Out>>,
	output_idx: *mut Tensor<Line<u32>>,
}

#[cube]
impl<In: Numeric, Out: Numeric> TupledOutputIdxIO<In, Out> {
	pub fn init(
		input: &Tensor<Line<In>>,
		output: &mut Tensor<Line<Out>>,
		output_idx: &mut Tensor<Line<u32>>,
	) -> Self {
		TupledOutputIdxIO::<In, Out> {
			input,
			output,
			output_idx,
		}
	}
}
