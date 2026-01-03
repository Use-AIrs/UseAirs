// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use crate::error::Result;
use inquire::Select;
use lib_stage::stager;

pub fn start_menu() -> Result<()> {
	loop {
		let st_menu = vec!["Init Transformation", "Init Training", "Test Model", "Back"];

		let selection = Select::new("Executions:", st_menu)
			.with_help_message("Here you can load, list, import and create configurations.")
			.prompt()?;

		match selection {
			"Init Transformation" => {
				init_transform()?;
			},
			"Init Training" => {
				println!("Not implemented yet");
			},
			"Test Model" => {
				println!("Not implemented yet");
			},
			"Back" => {
				return Ok(());
			},
			_ => (),
		}
	}
}

fn init_transform() -> Result<()> {
	Ok(stager()?)
}
