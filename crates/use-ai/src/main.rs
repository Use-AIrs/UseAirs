// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

mod config_menu;
mod error;
mod start_menu;

use crate::config_menu::config_menu;
use crate::error::Result;
use crate::start_menu::start_menu;
use inquire::Select;

fn main() -> Result<()> {
	loop {
		let main_menu = vec!["Start", "Config", "Exit"];

		let selection = Select::new(
			"Welcome to Use-AI.rs! Please select:",
			main_menu,
		)
		.with_help_message("Use the arrow keys to navigate and press Enter to select.")
		.prompt()?;

		match selection {
			"Start" => start_menu()?,
			"Config" => config_menu()?,
			"Exit" => {
				println!("Have a nice day!");
				break;
			},
			_ => (),
		}
	}

	Ok(())
}
