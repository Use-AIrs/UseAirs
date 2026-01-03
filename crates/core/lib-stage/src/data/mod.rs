// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

#[allow(unused_imports, dead_code, unused_variables)]
pub mod for_raw_table;

use crate::error::{Result, StageError};

use ndarray::Array2;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::collections::HashMap;

#[derive(Debug)]
pub struct Table {
	pub header: Option<HashMap<String, usize>>,
	pub data: Result<Array2<f32>>,
	pub map: Option<HashMap<usize, HashMap<String, usize>>>,
}

#[derive(Debug, Clone)]
pub struct RawTable<T> {
	pub header: Option<HashMap<String, usize>>,
	pub data: Array2<T>,
	pub map: Option<HashMap<usize, HashMap<String, usize>>>,
}

impl RawTable<String> {
	pub fn convert_to_f32_table(self) -> Result<Table> {
		let array_f32 = self
			.data
			.map(|string_val| string_val.parse::<f32>().unwrap_or(f32::NAN));

		let data = Ok(array_f32);

		Ok(Table {
			header: self.header,
			data,
			map: self.map,
		})
	}
	pub fn categories(
		mut self,
		columns: Vec<String>,
	) -> Result<Self> {
		let header = match &self.header {
			Some(h) => h,
			None => Err(StageError::NoInputColumns)?,
		};

		let mut global_map = self.map.take().unwrap_or_else(HashMap::new);

		let column_maps: Vec<(usize, HashMap<String, usize>)> = columns
			.par_iter()
			.filter_map(|col_name| {
				let col_index = header.get(col_name)?;

				let mut local_map = HashMap::new();
				let mut next_id: usize = 0;

				for row_index in 0..self.data.nrows() {
					let entry = &self.data[[row_index, *col_index]];
					if !local_map.contains_key(entry) {
						local_map.insert(entry.clone(), next_id);
						next_id += 1;
					}
				}
				Some((*col_index, local_map))
			})
			.collect();
		for (col_index, col_map) in &column_maps {
			global_map.insert(*col_index, col_map.clone());
		}
		self.map = Some(global_map);

		for (col_index, col_map) in &column_maps {
			for row_index in 0..self.data.nrows() {
				let key = &self.data[[row_index, *col_index]];
				let mapped_id = match col_map.get(key) {
					Some(id) => id.to_string(),
					None => "0".to_string(),
				};
				self.data[[row_index, *col_index]] = mapped_id;
			}
		}

		Ok(self)
	}
}
