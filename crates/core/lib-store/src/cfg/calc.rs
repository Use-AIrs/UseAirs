// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "model_type")]
pub enum Models {
	NeuralNetwork {
		id: usize,
		input_columns: Option<Vec<String>>,
		input_from: Option<String>,
		hyperparams: HPNeuralNetwork,
		mode: Mode,
	},
	GradientBoostedDecisionTree {
		id: usize,
		input_columns: Option<Vec<String>>,
		input_from: Option<String>,
		target_columns: Vec<String>,
		hyperparams: GbdtRules,
		mode: Mode,
	},
	QLearning {
		id: usize,
		input_columns: Option<Vec<String>>,
		input_from: Option<String>,
		hyperparams: HPQLearning,
		mode: Mode,
	},
}

#[derive(Debug, Serialize, Deserialize)]
pub enum Mode {
	Train,
	Generate,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HPNeuralNetwork {
	pub layers: Vec<usize>,
	pub activation: String,
	pub optimizer: Option<String>,
	pub epochs: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HPGradientBoostedDecisionTree {
	pub n_trees: usize,
	pub learning_rate: f64,
	pub max_depth: usize,
	pub sub_sample: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HPQLearning {
	pub environment: String,
	pub learning_rate: f64,
	pub discount: f64,
	pub episodes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GbdtRules {
	pub n_trees: usize,
	pub learning_rate: f64,
	pub max_depth: usize,
}
