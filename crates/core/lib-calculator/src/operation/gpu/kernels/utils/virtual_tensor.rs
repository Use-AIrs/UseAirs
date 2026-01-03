// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use super::KernelArgs;
use super::*;
use cubecl_core::prelude::*;
use cubecl_core::unexpanded;
use cubecl_std::{
	tensor::r#virtual::{VirtualTensor, VirtualTensorOperations, VirtualTensorOperationsExpand},
	CubeOption, CubeOptionExpand,
};
use std::marker::PhantomData;

pub struct Input;
pub struct Output;
pub struct InArgs;
pub struct OutArgs;

pub struct TensorArg<P: KernelDType, KA: KernelArgs, Tag> {
	_state: *mut KA::State<P>,
	tag: PhantomData<Tag>,
}

pub struct TensorArgExpand<P: KernelDType, KA: KernelArgs, Tag> {
	pub(crate) state: <KA::State<P> as CubeType>::ExpandType,
	tag: PhantomData<Tag>,
}

impl<P: KernelDType, KA: KernelArgs> TensorArg<P, KA, Input> {
	#[allow(dead_code)]
	pub fn new_input(_state: &KA::State<P>) -> Self {
		unexpanded!()
	}

	pub fn __expand_new_input(
		_scope: &mut Scope,
		state: <KA::State<P> as CubeType>::ExpandType,
	) -> TensorArgExpand<P, KA, Input> {
		TensorArgExpand {
			state,
			tag: PhantomData,
		}
	}
}

impl<P: KernelDType, KA: KernelArgs> TensorArg<P, KA, InArgs> {
	#[allow(dead_code)]
	pub fn new_arg(_state: &KA::State<P>) -> Self {
		unexpanded!()
	}

	pub fn __expand_new_arg(
		_scope: &mut Scope,
		state: <KA::State<P> as CubeType>::ExpandType,
	) -> TensorArgExpand<P, KA, InArgs> {
		TensorArgExpand {
			state,
			tag: PhantomData,
		}
	}
}

impl<P: KernelDType, KA: KernelArgs> TensorArg<P, KA, OutArgs> {
	#[allow(dead_code)]
	pub fn new_out_arg(_state: &mut KA::State<P>) -> Self {
		unexpanded!()
	}

	pub fn __expand_new_out_arg(
		_scope: &mut Scope,
		state: <KA::State<P> as CubeType>::ExpandType,
	) -> TensorArgExpand<P, KA, OutArgs> {
		TensorArgExpand {
			state,
			tag: PhantomData,
		}
	}
}

impl<P: KernelDType, KA: KernelArgs> TensorArg<P, KA, Output> {
	#[allow(dead_code)]
	pub fn new_output(_state: &mut KA::State<P>) -> Self {
		unexpanded!()
	}

	pub fn __expand_new_output(
		_scope: &mut Scope,
		state: <KA::State<P> as CubeType>::ExpandType,
	) -> TensorArgExpand<P, KA, Output> {
		TensorArgExpand {
			state,
			tag: PhantomData,
		}
	}
}

impl<P: KernelDType, KA: KernelArgs> VirtualTensorOperations<P::In> for TensorArg<P, KA, Input> {}

impl<P: KernelDType, KA: KernelArgs> VirtualTensorOperationsExpand<P::In>
	for TensorArgExpand<P, KA, Input>
{
	fn __expand_read_method(
		&self,
		scope: &mut Scope,
		index: ExpandElementTyped<u32>,
	) -> ExpandElementTyped<Line<P::In>> {
		KA::__expand_read_input(scope, self.state.clone(), index)
	}

	fn __expand_write_method(
		&self,
		_scope: &mut Scope,
		_index: ExpandElementTyped<u32>,
		_value: ExpandElementTyped<Line<P::In>>,
	) {
		unreachable!("Cannot write to input tensor")
	}

	fn __expand_shape_method(
		&self,
		scope: &mut Scope,
		axis: ExpandElementTyped<u32>,
	) -> ExpandElementTyped<u32> {
		KA::__expand_shape_input(scope, self.state.clone(), axis)
	}

	fn __expand_stride_method(
		&self,
		scope: &mut Scope,
		axis: ExpandElementTyped<u32>,
	) -> ExpandElementTyped<u32> {
		KA::__expand_stride_input(scope, self.state.clone(), axis)
	}

	fn __expand_rank_method(
		&self,
		scope: &mut Scope,
	) -> ExpandElementTyped<u32> {
		KA::__expand_rank_input(scope, self.state.clone())
	}

	fn __expand_len_method(
		&self,
		scope: &mut Scope,
	) -> ExpandElementTyped<u32> {
		KA::__expand_len_input(scope, self.state.clone())
	}

	fn __expand_buffer_len_method(
		&self,
		scope: &mut Scope,
	) -> ExpandElementTyped<u32> {
		KA::__expand_buffer_len_input(scope, self.state.clone())
	}

	fn __expand_read_window_method(
		&self,
		_context: &mut Scope,
		_start: ExpandElementTyped<u32>,
		_end: ExpandElementTyped<u32>,
	) -> SliceExpand<Line<P::In>, ReadOnly> {
		panic!("read_window not supported for KernelArgs-based virtual tensors")
	}

	fn __expand_as_tensor_map_method(
		&self,
		scope: &mut Scope,
	) -> CubeOptionExpand<TensorMap<P::In>> {
		CubeOption::__expand_new_None(scope)
	}
}

impl<P: KernelDType, KA: KernelArgs> Lined for TensorArg<P, KA, Input> {}
impl<P: KernelDType, KA: KernelArgs> LinedExpand for TensorArgExpand<P, KA, Input> {
	fn line_size(&self) -> u32 {
		let mut scope = Scope::root(false);
		KA::__expand_line_size_input(&mut scope, self.state.clone())
	}
}

impl<P: KernelDType, KA: KernelArgs> VirtualTensorOperations<P::Out> for TensorArg<P, KA, Output> {}

impl<P: KernelDType, KA: KernelArgs> VirtualTensorOperationsExpand<P::Out>
	for TensorArgExpand<P, KA, Output>
{
	fn __expand_read_method(
		&self,
		scope: &mut Scope,
		index: ExpandElementTyped<u32>,
	) -> ExpandElementTyped<Line<P::Out>> {
		KA::__expand_read_output(scope, self.state.clone(), index)
	}

	fn __expand_write_method(
		&self,
		scope: &mut Scope,
		index: ExpandElementTyped<u32>,
		value: ExpandElementTyped<Line<P::Out>>,
	) {
		KA::__expand_write_output(scope, self.state.clone(), index, value);
	}

	fn __expand_shape_method(
		&self,
		scope: &mut Scope,
		axis: ExpandElementTyped<u32>,
	) -> ExpandElementTyped<u32> {
		KA::__expand_shape_output(scope, self.state.clone(), axis)
	}

	fn __expand_stride_method(
		&self,
		scope: &mut Scope,
		axis: ExpandElementTyped<u32>,
	) -> ExpandElementTyped<u32> {
		KA::__expand_stride_output(scope, self.state.clone(), axis)
	}

	fn __expand_rank_method(
		&self,
		scope: &mut Scope,
	) -> ExpandElementTyped<u32> {
		KA::__expand_rank_output(scope, self.state.clone())
	}

	fn __expand_len_method(
		&self,
		scope: &mut Scope,
	) -> ExpandElementTyped<u32> {
		KA::__expand_len_output(scope, self.state.clone())
	}

	fn __expand_buffer_len_method(
		&self,
		scope: &mut Scope,
	) -> ExpandElementTyped<u32> {
		KA::__expand_buffer_len_output(scope, self.state.clone())
	}

	fn __expand_read_window_method(
		&self,
		_context: &mut Scope,
		_start: ExpandElementTyped<u32>,
		_end: ExpandElementTyped<u32>,
	) -> SliceExpand<Line<P::Out>, ReadOnly> {
		panic!("read_window not supported for KernelArgs-based virtual tensors")
	}

	fn __expand_as_tensor_map_method(
		&self,
		scope: &mut Scope,
	) -> CubeOptionExpand<TensorMap<P::Out>> {
		CubeOption::__expand_new_None(scope)
	}
}

impl<P: KernelDType, KA: KernelArgs> Lined for TensorArg<P, KA, Output> {}
impl<P: KernelDType, KA: KernelArgs> LinedExpand for TensorArgExpand<P, KA, Output> {
	fn line_size(&self) -> u32 {
		let mut scope = Scope::root(false);
		KA::__expand_line_size_output(&mut scope, self.state.clone())
	}
}

#[derive(CubeType)]
pub struct VirtualIO<In: Numeric, Out: Numeric> {
	pub read: VirtualTensor<In>,
	pub write: VirtualTensor<Out, ReadWrite>,
}

#[cube]
impl<In: Numeric, Out: Numeric> VirtualIO<In, Out> {
	pub fn init_tensors<KA: KernelArgs>(
		input: &KA::Input<In>,
		output: &mut KA::Output<Out>,
	) -> VirtualIO<In, Out> {
		let mut state = KA::init_state::<(In, Out)>(input, output);

		let input_arg = TensorArg::new_input(&state);
		let mut output_arg = TensorArg::new_output(&mut state);

		let input_vt = VirtualTensor::<In>::new::<TensorArg<(In, Out), KA, Input>>(&input_arg);
		let output_vt = VirtualTensor::<Out, ReadWrite>::new::<TensorArg<(In, Out), KA, Output>>(
			&mut output_arg,
		);

		VirtualIO::<In, Out> {
			read: input_vt,
			write: output_vt,
		}
	}
}

mod __tensor_arg {
	use super::*;

	impl<P: KernelDType, KA: KernelArgs, Tag> CubeType for TensorArg<P, KA, Tag> {
		type ExpandType = TensorArgExpand<P, KA, Tag>;
	}

	impl<P: KernelDType, KA: KernelArgs, Tag> IntoMut for TensorArgExpand<P, KA, Tag> {
		fn into_mut(
			self,
			_scope: &mut Scope,
		) -> Self {
			self
		}
	}

	impl<P: KernelDType, KA: KernelArgs, Tag> CubeDebug for TensorArgExpand<P, KA, Tag> {}

	impl<P: KernelDType, KA: KernelArgs, Tag> Clone for TensorArgExpand<P, KA, Tag> {
		fn clone(&self) -> Self {
			Self {
				state: self.state.clone(),
				tag: self.tag,
			}
		}
	}
}
