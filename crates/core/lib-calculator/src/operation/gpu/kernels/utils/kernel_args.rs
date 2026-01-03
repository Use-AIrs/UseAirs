// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use cubecl_core::prelude::*;
use std::marker::PhantomData;

pub trait KernelDType {
	type In: Numeric;
	type Out: Numeric;
}

impl<In: Numeric, Out: Numeric> KernelDType for (In, Out) {
	type In = In;
	type Out = Out;
}

#[cube]
pub trait KernelArgs: Send + Sync + 'static + Clone {
	type Input<In: Numeric>: LaunchArg + CubeType;
	type Output<Out: Numeric>: LaunchArg + CubeType;
	type State<P: KernelDType>: CubeType;

	fn init_state<P: KernelDType>(
		input: &Self::Input<P::In>,
		output: &mut Self::Output<P::Out>,
	) -> Self::State<P>;

	fn read_input<P: KernelDType>(
		state: &Self::State<P>,
		index: u32,
	) -> Line<P::In>;
	fn read_output<P: KernelDType>(
		state: &Self::State<P>,
		index: u32,
	) -> Line<P::Out>;

	fn write_output<P: KernelDType>(
		state: &mut Self::State<P>,
		index: u32,
		value: Line<P::Out>,
	);

	fn shape_input<P: KernelDType>(
		state: &Self::State<P>,
		dim: u32,
	) -> u32;
	fn shape_output<P: KernelDType>(
		state: &Self::State<P>,
		dim: u32,
	) -> u32;

	fn stride_input<P: KernelDType>(
		state: &Self::State<P>,
		dim: u32,
	) -> u32;
	fn stride_output<P: KernelDType>(
		state: &Self::State<P>,
		dim: u32,
	) -> u32;

	fn len_input<P: KernelDType>(state: &Self::State<P>) -> u32;
	fn len_output<P: KernelDType>(state: &Self::State<P>) -> u32;

	fn buffer_len_input<P: KernelDType>(state: &Self::State<P>) -> u32;
	fn buffer_len_output<P: KernelDType>(state: &Self::State<P>) -> u32;

	fn rank_input<P: KernelDType>(state: &Self::State<P>) -> u32;
	fn rank_output<P: KernelDType>(state: &Self::State<P>) -> u32;

	fn line_size_input<P: KernelDType>(state: &Self::State<P>) -> comptime_type!(u32);
	fn line_size_output<P: KernelDType>(state: &Self::State<P>) -> comptime_type!(u32);
}

#[derive(CubeType, Clone)]
pub struct In1Out1;

#[derive(CubeType)]
pub struct StateIO<P: KernelDType> {
	i: *const Tensor<Line<P::In>>,
	o: *mut Tensor<Line<P::Out>>,
}

#[cube]
impl KernelArgs for In1Out1 {
	type Input<E: Numeric> = Tensor<Line<E>>;
	type Output<E: Numeric> = Tensor<Line<E>>;

	type State<P: KernelDType> = StateIO<P>;

	fn init_state<P: KernelDType>(
		input: &Self::Input<P::In>,
		output: &mut Self::Output<P::Out>,
	) -> Self::State<P> {
		StateIO::<P> {
			i: input,
			o: output,
		}
	}

	fn read_input<P: KernelDType>(
		state: &Self::State<P>,
		index: u32,
	) -> Line<P::In> {
		unsafe { (*state.i)[index] }
	}

	fn read_output<P: KernelDType>(
		state: &Self::State<P>,
		index: u32,
	) -> Line<P::Out> {
		unsafe { (*state.o)[index] }
	}

	fn write_output<P: KernelDType>(
		state: &mut Self::State<P>,
		index: u32,
		value: Line<P::Out>,
	) {
		unsafe { (*state.o)[index] = value }
	}

	fn len_input<P: KernelDType>(state: &Self::State<P>) -> u32 {
		unsafe { (*state.i).len() }
	}

	fn len_output<P: KernelDType>(state: &Self::State<P>) -> u32 {
		unsafe { (*state.o).len() }
	}

	fn buffer_len_input<P: KernelDType>(state: &Self::State<P>) -> u32 {
		unsafe { (*state.i).buffer_len() }
	}

	fn buffer_len_output<P: KernelDType>(state: &Self::State<P>) -> u32 {
		unsafe { (*state.o).buffer_len() }
	}

	fn rank_input<P: KernelDType>(state: &Self::State<P>) -> u32 {
		unsafe { (*state.i).rank() }
	}

	fn rank_output<P: KernelDType>(state: &Self::State<P>) -> u32 {
		unsafe { (*state.o).rank() }
	}

	fn shape_input<P: KernelDType>(
		state: &Self::State<P>,
		dim: u32,
	) -> u32 {
		unsafe { (*state.i).shape(dim) }
	}

	fn shape_output<P: KernelDType>(
		state: &Self::State<P>,
		dim: u32,
	) -> u32 {
		unsafe { (*state.o).shape(dim) }
	}

	fn stride_input<P: KernelDType>(
		state: &Self::State<P>,
		dim: u32,
	) -> u32 {
		unsafe { (*state.i).stride(dim) }
	}

	fn stride_output<P: KernelDType>(
		state: &Self::State<P>,
		dim: u32,
	) -> u32 {
		unsafe { (*state.o).stride(dim) }
	}

	fn line_size_input<P: KernelDType>(state: &Self::State<P>) -> comptime_type!(u32) {
		unsafe { (*state.i).line_size() }
	}

	fn line_size_output<P: KernelDType>(state: &Self::State<P>) -> comptime_type!(u32) {
		unsafe { (*state.o).line_size() }
	}
}
