// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use crate::data::RawTable;
use crate::error::{Result, StageError};
use csv::{ReaderBuilder, StringRecord};
use lib_store::cfg::DataSection;
use ndarray::{stack, Array1, Array2, Axis};
use rayon::prelude::*;
use std::collections::HashMap;

impl<T> RawTable<T> {
	pub fn from_csv(data: &DataSection) -> Result<RawTable<String>> {
		let path = data.source.path.clone().unwrap();
		let scheme = data.scheme.clone().unwrap();
		let header = scheme.columns;

		Ok(build(path, header)?)
	}
}

fn string_records(path: String) -> Result<Vec<StringRecord>> {
	let mut rdr = ReaderBuilder::new().from_path(&path).unwrap();

	let mut records = Vec::new();
	for result in rdr.records() {
		let record = result.unwrap();
		records.push(record);
	}
	Ok(records)
}

pub fn build(
	path: String,
	input: Option<Vec<String>>,
) -> Result<RawTable<String>> {
	let records = string_records(path)?;
	println!("{:?}", input);
	let header = input.map(|header_vec| {
		header_vec
			.into_iter()
			.enumerate()
			.map(|(index, value)| (value, index))
			.collect::<HashMap<String, usize>>()
	});
	println!("{:?}", header);

	let processed: Vec<Array1<String>> = records
		.par_iter()
		.map(|record| {
			let parsed: Vec<String> = record.iter().map(|field| field.to_string()).collect();
			Ok(Array1::from(parsed))
		})
		.collect::<Result<Vec<Array1<String>>>>()?;

	let combined: Array2<String> = stack(
		Axis(0),
		&processed.iter().map(|arr| arr.view()).collect::<Vec<_>>(),
	)
	.map_err(|e| StageError::ShapingError(e))?;

	Ok(RawTable {
		header,
		data: combined,
		map: None,
	})
}
