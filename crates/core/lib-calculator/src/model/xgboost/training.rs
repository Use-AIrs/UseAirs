// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use crate::operation::*;
use crate::{MetaData, Tensor};
use cubecl::prelude::Runtime;
use std::marker::PhantomData;

use crate::model::error::Result;
use crate::model::Model;

pub struct XGBoostTraining<R: Runtime> {
	_pd: PhantomData<R>,
}

#[derive(Clone)]
pub struct XGBoostConfig {
	pub max_depth: usize,
	pub min_child_weight: f32,
	pub gamma: f32,  
	pub lambda: f32, 
	pub alpha: f32,  
	pub learning_rate: f32,
	pub subsample: f32,
	pub colsample_bytree: f32,
	pub num_trees: usize,
}

impl Default for XGBoostConfig {
	fn default() -> Self {
		Self {
			max_depth: 6,
			min_child_weight: 1.0,
			gamma: 0.0,
			lambda: 1.0,
			alpha: 0.0,
			learning_rate: 0.3,
			subsample: 1.0,
			colsample_bytree: 1.0,
			num_trees: 100,
		}
	}
}

#[derive(Clone, Debug)]
pub struct TreeNode {
	pub feature_idx: Option<usize>,
	pub split_value: Option<f32>,
	pub weight: f32,
	pub gain: f32,
	pub left_child: Option<usize>,
	pub right_child: Option<usize>,
	pub is_leaf: bool,
}

pub struct XGBoostInput {
	pub features: Tensor<f32>,              
	pub labels: Tensor<f32>,                
	pub sample_weight: Option<Tensor<f32>>, 
}

pub struct XGBoostOutput {
	pub trees: Vec<Vec<TreeNode>>,
	pub feature_importance: Tensor<f32>,
}

#[derive(Clone)]
struct SplitInfo {
	feature_idx: usize,
	split_value: f32,
	gain: f32,
	left_sum_gradient: f32,
	left_sum_hessian: f32,
	right_sum_gradient: f32,
	right_sum_hessian: f32,
}

impl<R: Runtime> Model for XGBoostTraining<R> {
	type Input = XGBoostInput;
	type Output = XGBoostOutput;
	type Ctx = XGBoostConfig;

	fn execute(
		ctx: Self::Ctx,
		input: Self::Input,
	) -> Result<Self::Output> {
		let mut trees = Vec::new();

		let predictions = Tensor::<f32>::from_shape_value(&input.labels.shape(), 0.5);
		let op = Operator::<R, f32>::init()?;

		let features_gpu = op.tensor_copied(&input.features)?;
		let labels_gpu = op.tensor_copied(&input.labels)?;
		let predictions_gpu = op.tensor_copied(&predictions)?;

		let sigmoid_buffer = op.empty_copied(&input.labels.metadata)?;
		
		let one_minus_sigmoid = op.empty_copied(&input.labels.metadata)?;
		
		let gradients_gpu = op.empty_copied(&input.labels.metadata)?;
		let hessians_gpu = op.empty_copied(&input.labels.metadata)?;
		
		let ones = Tensor::<f32>::from_shape_value(&input.labels.shape(), 1.0);
		let ones_gpu = op.tensor_copied(&ones)?;
		
		let buffer_n = op.empty_copied(&input.labels.metadata)?;
		let buffer_2n = op.empty_copied(&input.labels.metadata)?;

		for tree_idx in 0..ctx.num_trees {
			
			op.operation()
				.pipeline_all(
					Sigmoid,
					(),
					predictions_gpu,
					sigmoid_buffer,
				)?
				.pipeline_all(Negate, (), labels_gpu, buffer_n)?
				.pipeline_all(
					Zip,
					(),
					(sigmoid_buffer, buffer_n),
					buffer_2n,
				)?
				.pipeline_all(Sum, 1, buffer_2n, gradients_gpu)?
				
				.pipeline_all(
					OneMinus,
					(),
					sigmoid_buffer,
					one_minus_sigmoid,
				)?
				.pipeline_all(
					Zip,
					(),
					(sigmoid_buffer, one_minus_sigmoid),
					buffer_2n,
				)?
				.pipeline_all(Prod, 1, buffer_2n, hessians_gpu)?
				.execute()?;

			let tree = build_tree_exact_gpu(
				&op,
				features_gpu,
				gradients_gpu,
				hessians_gpu,
				&ctx,
			)?;

			let tree_predictions = op.empty_copied(&MetaData::matrix(n_samples, 1))?;
			op.operation()
				
				.execute()?;

			update_feature_importance(&tree, &mut feature_importance);

			trees.push(tree);

			if tree_idx % 10 == 0 {
				println!(
					"Completed tree {}/{}",
					tree_idx + 1,
					ctx.num_trees
				);
			}
		}

		let feature_importance_tensor = Tensor::new(
			feature_importance,
			MetaData::matrix(1, n_features),
		);

		Ok(XGBoostOutput {
			trees,
			feature_importance: feature_importance_tensor,
		})
	}
}

fn build_tree_exact_gpu<R: Runtime>(
	operator: &Operator<R, f32>,
	features: usize,
	gradients: usize,
	hessians: usize,
	config: &XGBoostConfig,
) -> Result<Vec<TreeNode>> {
	let mut nodes = Vec::new();
	let max_nodes = (1 << (config.max_depth + 1)) - 1;

	nodes.push(TreeNode {
		feature_idx: None,
		split_value: None,
		weight: 0.0,
		gain: 0.0,
		left_child: None,
		right_child: None,
		is_leaf: false,
	});

	let mut node_samples = Tensor::<f32>::from_shape_value(&[max_nodes, n_samples], 0.0);
	
	for i in 0..n_samples {
		node_samples.data[i] = 1.0;
	}
	let node_samples_gpu = operator.tensor_copied(&node_samples)?;

	let sorted_indices = operator.empty_copied(&MetaData::matrix(n_features, n_samples))?;
	
	let cumsum_gradients = operator.empty_copied(&MetaData::matrix(n_features, n_samples))?;
	let cumsum_hessians = operator.empty_copied(&MetaData::matrix(n_features, n_samples))?;
	
	let split_candidates = operator.empty_copied(&MetaData::matrix(n_features, 5))?;

	for depth in 0..config.max_depth {
		let start_idx = (1 << depth) - 1;
		let end_idx = (1 << (depth + 1)) - 1;

		for node_idx in start_idx..end_idx {
			if node_idx >= nodes.len() || nodes[node_idx].is_leaf {
				continue;
			}

			operator
				.operation()
				
				.execute()?;

			operator
				.operation()
				
				.execute()?;

			operator
				.operation()
				
				.execute()?;

			let best_feature_idx = operator.empty_copied(&MetaData::matrix(1, 1))?;
			operator
				.operation()
				.pipeline_all(
					ArgMax,
					0,
					split_candidates,
					best_feature_idx,
				)?
				.execute()?;

			let split_data = operator.get_tensors(split_candidates)?;
			let best_feat = operator.get_tensors(best_feature_idx)?;

			let best_split = SplitInfo {
				feature_idx: best_feat[0].data[0] as usize,
				split_value: 0.0, 
				gain: 0.0,
				left_sum_gradient: 0.0,
				left_sum_hessian: 0.0,
				right_sum_gradient: 0.0,
				right_sum_hessian: 0.0,
			};

			if should_be_leaf(&best_split, config, depth) {
				nodes[node_idx].is_leaf = true;
				continue;
			}

			let left_idx = 2 * node_idx + 1;
			let right_idx = 2 * node_idx + 2;

			nodes[node_idx].feature_idx = Some(best_split.feature_idx);
			nodes[node_idx].split_value = Some(best_split.split_value);
			nodes[node_idx].gain = best_split.gain;
			nodes[node_idx].left_child = Some(left_idx);
			nodes[node_idx].right_child = Some(right_idx);

			while nodes.len() <= right_idx {
				nodes.push(TreeNode {
					feature_idx: None,
					split_value: None,
					weight: 0.0,
					gain: 0.0,
					left_child: None,
					right_child: None,
					is_leaf: false,
				});
			}

			operator
				.operation()
				
				.execute()?;
		}
	}

	compute_leaf_weights_exact_gpu(
		operator,
		&mut nodes,
		gradients,
		hessians,
		node_samples_gpu,
		config,
	)?;

	Ok(nodes)
}

fn compute_leaf_weights_exact_gpu<R: Runtime>(
	operator: &Operator<R, f32>,
	nodes: &mut Vec<TreeNode>,
	gradients: usize,
	hessians: usize,
	node_samples: usize,
	config: &XGBoostConfig,
) -> Result<()> {
	
	let n_nodes = nodes.len();

	let node_grad_sums = operator.empty_copied(&MetaData::matrix(n_nodes, 1))?;
	let node_hess_sums = operator.empty_copied(&MetaData::matrix(n_nodes, 1))?;

	for (node_idx, node) in nodes.iter_mut().enumerate() {
		if !node.is_leaf {
			continue;
		}

		operator
			.operation()
			
			.execute()?;
	}

	let grad_sums = operator.get_tensors(node_grad_sums)?;
	let hess_sums = operator.get_tensors(node_hess_sums)?;

	for (node_idx, node) in nodes.iter_mut().enumerate() {
		if node.is_leaf {
			let sum_g = grad_sums[0].data[node_idx];
			let sum_h = hess_sums[0].data[node_idx];
			node.weight = -sum_g / (sum_h + config.lambda);
		}
	}

	Ok(())
}

fn should_be_leaf(
	split: &SplitInfo,
	config: &XGBoostConfig,
	depth: usize,
) -> bool {
	
	if depth >= config.max_depth {
		return true;
	}

	if split.gain < config.gamma {
		return true;
	}

	if split.left_sum_hessian < config.min_child_weight
		|| split.right_sum_hessian < config.min_child_weight
	{
		return true;
	}

	false
}

fn update_feature_importance(
	tree: &Vec<TreeNode>,
	importance: &mut Vec<f32>,
) {
	for node in tree {
		if let Some(feature_idx) = node.feature_idx {
			importance[feature_idx] += node.gain;
		}
	}
}

impl XGBoostOutput {
	pub fn predict<R: Runtime>(
		&self,
		features: &Tensor<f32>,
		operator: &Operator<R, f32>,
	) -> Result<Tensor<f32>> {
		
		let features_gpu = operator.tensor_copied(features)?;
		let n_samples = features.metadata.shape[0];

		let predictions = Tensor::<f32>::from_shape_value(&[n_samples], 0.0);
		let predictions_gpu = operator.tensor_copied(&predictions)?;

		let tree_preds = operator.empty_copied(&MetaData::matrix(n_samples, 1))?;
		let buffer_2n = operator.empty_copied(&MetaData::matrix(n_samples, 2))?;

		for tree in &self.trees {
			
			operator
				.operation()
				
				.execute()?;
		}

		operator
			.operation()
			
			.execute()?;

		let result = operator.get_tensors(predictions_gpu)?;
		Ok(result[0].clone())
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use cubecl::cuda::CudaRuntime;

	#[test]
	fn test_xgboost_training() -> Result<()> {
		
		let n_samples = 1000;
		let n_features = 10;

		let features_data: Vec<f32> = (0..n_samples * n_features)
			.map(|i| (i as f32) / 100.0)
			.collect();
		let labels_data: Vec<f32> = (0..n_samples)
			.map(|i| if i % 2 == 0 { 0.0 } else { 1.0 })
			.collect();

		let features = Tensor::new(
			features_data,
			MetaData::matrix(n_samples, n_features),
		);
		let labels = Tensor::new(
			labels_data,
			MetaData::matrix(n_samples, 1),
		);

		let config = XGBoostConfig {
			num_trees: 10,
			max_depth: 3,
			..Default::default()
		};

		let input = XGBoostInput {
			features,
			labels,
			sample_weight: None,
		};

		let output = XGBoostTraining::<CudaRuntime>::execute(config, input)?;

		assert_eq!(output.trees.len(), 10);

		Ok(())
	}
}
