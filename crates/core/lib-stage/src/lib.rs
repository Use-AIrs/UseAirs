// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

#[allow(unused_imports, dead_code, unused_variables)]
use crate::data::{RawTable, Table};
use crate::error::{Result, StageError};

use lib_store::{cfg::DataSection, get_active_config};

pub mod data;

pub mod error;
pub mod output_guard;

pub fn stager() -> Result<()> {
	let cfg = get_active_config()?;
	let instruction = cfg.data;
	let data = get_data(&instruction)?;
	let result = stage(data, &instruction)?;
	println!("{:?}", result);
	Ok(())
}

fn get_data(data_source: &DataSection) -> Result<RawTable<String>> {
	let config = data_source;
	match data_source.source.source_type.as_str() {
		"csv" => RawTable::<String>::from_csv(config),
		_ => Err(StageError::DataTypeNotSupported),
	}
}
pub fn stage(
	mut table: RawTable<String>,
	instructions: &DataSection,
) -> Result<Table> {
	let transformations = match &instructions.transformer {
		Some(transformations) => transformations,
		None => return Err(StageError::NoTransformationsDefined),
	};

	for transformation in transformations {
		match transformation.operation.as_str() {
			"categories" => {
				let params = transformation
					.params
					.clone()
					.ok_or(StageError::NoTransformationParams)?;

				let columns = params
					.columns
					.ok_or(StageError::TransformationParamsWrong)?;

				println!(
					"Applying 'categories' to columns: {:?}",
					columns
				);
				table = table.categories(columns)?;
			},

			"date_conv" => {
				todo!()
			},

			other => {
				println!("Unknown operation: {}", other);
			},
		}
	}

	table.convert_to_f32_table()
}
