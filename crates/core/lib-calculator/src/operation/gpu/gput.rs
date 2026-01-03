// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use cubecl_core::prelude::Numeric;
use cubecl_core::server::Handle;
use cubecl_core::CubeElement;
use std::marker::PhantomData;
use std::rc::Rc;

#[derive(Debug, Clone)]
pub struct GpuTensor<N: Numeric + CubeElement> {
	pub handle: Handle,
	pub shape: Rc<[usize]>,
	pub stride: Rc<[usize]>,
	pub id: usize,
	pub byte_size: usize,
	pub element_count: usize,
	pub _phantom: PhantomData<N>,
}

impl<N: Numeric + CubeElement> GpuTensor<N> {
	
	pub fn new(handle: Handle, shape: Vec<usize>, stride: Vec<usize>, id: usize) -> Self {
		Self {
			handle,
			shape: shape.into(),
			stride: stride.into(),
			id,
			_phantom: PhantomData,
		}
	}

	pub fn with_row_major(handle: Handle, shape: Vec<usize>, id: usize) -> Self {
		let stride = compute_row_major_strides(&shape);
		Self::new(handle, shape, stride, id)
	}

	pub fn with_column_major(handle: Handle, shape: Vec<usize>, id: usize) -> Self {
		let stride = compute_column_major_strides(&shape);
		Self::new(handle, shape, stride, id)
	}

	pub fn shape(&self) -> &[usize] {
		&self.shape
	}

	pub fn stride(&self) -> &[usize] {
		&self.stride
	}

	pub fn handle(&self) -> &Handle {
		&self.handle
	}

	pub fn id(&self) -> usize {
		self.id
	}

	pub fn len(&self) -> usize {
		self.shape.iter().product()
	}

	pub fn is_empty(&self) -> bool {
		self.shape.iter().any(|&dim| dim == 0)
	}

	pub fn ndim(&self) -> usize {
		self.shape.len()
	}

	pub fn is_row_major(&self) -> bool {
		let expected_strides = compute_row_major_strides(&self.shape);
		*self.stride == expected_strides[..]
	}

	pub fn is_column_major(&self) -> bool {
		let expected_strides = compute_column_major_strides(&self.shape);
		*self.stride == expected_strides[..]
	}

	pub fn byte_size(&self) -> usize {
		self.len() * std::mem::size_of::<N>()
	}
}

pub fn compute_row_major_strides(shape: &[usize]) -> Vec<usize> {
	if shape.is_empty() {
		return Vec::new();
	}

	let mut strides = vec![1; shape.len()];
	for i in (0..shape.len() - 1).rev() {
		strides[i] = strides[i + 1] * shape[i + 1];
	}
	strides
}

pub fn compute_column_major_strides(shape: &[usize]) -> Vec<usize> {
	if shape.is_empty() {
		return Vec::new();
	}

	let mut strides = vec![1; shape.len()];
	for i in 1..shape.len() {
		strides[i] = strides[i - 1] * shape[i - 1];
	}
	strides
}

