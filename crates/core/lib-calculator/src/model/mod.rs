// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

mod error;
pub mod xgboost;

use error::Result;

pub trait Model {
	type Input;
	type Output;
	type Ctx;

	fn execute(
		ctx: Self::Ctx,
		model_input: Self::Input,
	) -> Result<Self::Output>;
}
