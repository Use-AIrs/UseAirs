// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use bytemuck::cast_slice;
use cubecl_core::prelude::{Numeric, TensorHandleRef};
use cubecl_core::{CubeElement, Runtime};

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct Tensor<N: Numeric + CubeElement> {
	pub data: Vec<N>,
	pub metadata: MetaData,
}

impl<N: Numeric + CubeElement> Tensor<N> {
	pub fn new(
		data: Vec<N>,
		metadata: MetaData,
	) -> Self {
		Self { data, metadata }
	}

	pub fn from_shape(
		data: Vec<N>,
		shape: &[usize],
	) -> Self {
		let stride = Box::leak(MetaData::row_strides(shape));
		let metadata = MetaData::new(stride, shape);
		Self { data, metadata }
	}

	pub fn from_shape_zeros(shape: &[usize]) -> Self
	where
		N: Clone,
	{
		let total_elements = shape.iter().product();
		let data = vec![N::zeroed(); total_elements];
		let stride = Box::leak(MetaData::row_strides(shape));
		let metadata = MetaData::new(stride, shape);
		Self { data, metadata }
	}

	pub fn from_shape_value(
		shape: &[usize],
		value: N,
	) -> Self {
		let total_elements = shape.iter().product();
		let data = vec![value; total_elements];
		let stride = Box::leak(MetaData::row_strides(shape));
		let metadata = MetaData::new(stride, shape);
		Self { data, metadata }
	}

	pub fn data(&self) -> &[u8] {
		cast_slice(&self.data)
	}

	pub fn len(&self) -> usize {
		self.data.len()
	}

	pub fn is_empty(&self) -> bool {
		self.data.is_empty()
	}

	pub fn shape(&self) -> &[usize] {
		&self.metadata.shape
	}

	pub fn stride(&self) -> &[usize] {
		&self.metadata.stride
	}
	pub fn extra_dim_md(&self) -> MetaData {
		let md_in = &self.metadata;

		if md_in.shape.len() == 2 {
			if md_in.shape[0] == 1 {
				let n = md_in.shape[1];
				MetaData {
					stride: vec![n, 1].into_boxed_slice(),
					shape: vec![2, n].into_boxed_slice(),
				}
			} else if md_in.shape[1] == 1 {
				let n = md_in.shape[0];
				MetaData {
					stride: vec![2, 1].into_boxed_slice(),
					shape: vec![n, 2].into_boxed_slice(),
				}
			} else {
				let mut new_shape = md_in.shape.to_vec();
				new_shape.push(2);

				let mut new_strides = md_in.stride.to_vec();
				for stride in &mut new_strides {
					*stride *= 2;
				}
				new_strides.push(1);

				MetaData {
					stride: new_strides.into_boxed_slice(),
					shape: new_shape.into_boxed_slice(),
				}
			}
		} else {
			let mut new_shape = md_in.shape.to_vec();
			new_shape.push(2);

			let mut new_strides = md_in.stride.to_vec();
			for stride in &mut new_strides {
				*stride *= 2;
			}
			new_strides.push(1);

			MetaData {
				stride: new_strides.into_boxed_slice(),
				shape: new_shape.into_boxed_slice(),
			}
		}
	}
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct MetaData {
	pub stride: Box<[usize]>,
	pub shape: Box<[usize]>,
}

impl MetaData {
	pub fn new(
		stride: &[usize],
		shape: &[usize],
	) -> Self {
		Self {
			stride: stride.into(),
			shape: shape.into(),
		}
	}

	pub fn scalar() -> Self {
		Self {
			stride: Box::new([1, 1]),
			shape: Box::new([1, 1]),
		}
	}

	pub fn vector(len: usize) -> Self {
		Self {
			stride: Box::new([len, 1]),
			shape: Box::new([1, len]),
		}
	}

	pub fn matrix(
		rows: usize,
		cols: usize,
	) -> Self {
		Self {
			stride: Box::new([cols, 1]),
			shape: Box::new([rows, cols]),
		}
	}

	pub fn row_strides(shape: &[usize]) -> Box<[usize]> {
		if shape.is_empty() {
			return Box::new([]);
		}

		let mut strides = vec![1; shape.len()];
		for i in (0..shape.len() - 1).rev() {
			strides[i] = strides[i + 1] * shape[i + 1];
		}
		strides.into_boxed_slice()
	}

	pub fn column_strides(shape: &[usize]) -> Box<[usize]> {
		if shape.is_empty() {
			return Box::new([]);
		}

		let mut strides = vec![1; shape.len()];
		for i in 1..shape.len() {
			strides[i] = strides[i - 1] * shape[i - 1];
		}
		strides.into_boxed_slice()
	}

	pub fn total_elements(&self) -> usize {
		self.shape.iter().product()
	}

	pub fn ndim(&self) -> usize {
		self.shape.len()
	}

	pub fn is_vector(&self) -> bool {
		self.shape.len() == 1
	}

	pub fn is_matrix(&self) -> bool {
		self.shape.len() == 2
	}

	pub fn is_scalar(&self) -> bool {
		self.total_elements() == 1
	}

	pub fn is_row_major(&self) -> bool {
		let expected_strides = Self::row_strides(&self.shape);
		*self.stride == *expected_strides
	}

	pub fn is_column_major(&self) -> bool {
		let expected_strides = Self::column_strides(&self.shape);
		*self.stride == *expected_strides
	}

	pub fn bytes_size<N: Numeric>(&self) -> usize {
		self.total_elements() * std::mem::size_of::<N>()
	}

	pub fn reshape(
		&self,
		new_shape: Box<[usize]>,
	) -> Option<Self> {
		let old_elements = self.total_elements();
		let new_elements: usize = new_shape.iter().product();

		if old_elements != new_elements {
			return None;
		}

		let new_stride = Self::row_strides(&new_shape);
		Some(Self {
			stride: new_stride,
			shape: new_shape,
		})
	}
}

#[cfg(test)]
mod test {
	use super::*;

	#[test]
	fn test_metadata_scalar() {
		let md = MetaData::scalar();
		assert!(md.is_scalar());
		assert_eq!(md.total_elements(), 1);
		assert_eq!(md.ndim(), 1);
	}

	#[test]
	fn test_metadata_vector() {
		let md = MetaData::vector(5);
		assert!(md.is_vector());
		assert_eq!(md.total_elements(), 5);
		assert_eq!(md.ndim(), 1);
	}

	#[test]
	fn test_metadata_matrix() {
		let md = MetaData::matrix(3, 4);
		assert!(md.is_matrix());
		assert_eq!(md.total_elements(), 12);
		assert_eq!(md.ndim(), 2);
		assert!(md.is_row_major());
	}

	#[test]
	fn test_row_major_strides() {
		let strides = MetaData::row_strides(&[2, 3, 4]);
		assert_eq!(*strides, [12, 4, 1]);
	}

	#[test]
	fn test_column_major_strides() {
		let strides = MetaData::column_strides(&[2, 3, 4]);
		assert_eq!(*strides, [1, 2, 6]);
	}

	#[test]
	fn test_tensor_creation() {
		let data = vec![1.0, 2.0, 3.0, 4.0];
		let tensor = Tensor::from_shape(data, &[2, 2]);

		assert_eq!(tensor.len(), 4);
		assert_eq!(tensor.shape(), &[2, 2]);
		assert!(tensor.metadata.is_matrix());
	}

	#[test]
	fn test_tensor_zeros() {
		let tensor: Tensor<f32> = Tensor::from_shape_zeros(&[2, 3]);

		assert_eq!(tensor.len(), 6);
		assert!(tensor.data.iter().all(|&x| x == 0.0));
		assert_eq!(tensor.shape(), &[2, 3]);
	}

	#[test]
	fn test_metadata_reshape() {
		let md = MetaData::matrix(2, 6);
		let reshaped = md.reshape(Box::new([3, 4])).unwrap();

		assert_eq!(*reshaped.shape, [3, 4]);
		assert_eq!(reshaped.total_elements(), 12);
		assert!(reshaped.is_row_major());
	}

	#[test]
	fn test_reshape_invalid() {
		let md = MetaData::matrix(2, 3);
		let result = md.reshape(Box::new([2, 2]));
		assert!(result.is_none());
	}
}
