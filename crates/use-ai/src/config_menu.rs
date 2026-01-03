// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use crate::error::Result;
use inquire::{Select, Text};
use lib_store::{activate_config, config_from_file, list_configs};

pub fn config_menu() -> Result<()> {
	loop {
		let con_menu = vec![
			"Load Config",
			"List Configs",
			"Import",
			"New Config",
			"Back",
		];

		let selection = Select::new("Configuration:", con_menu)
			.with_help_message("Here you can load, list, import and create configurations.")
			.prompt();

		match selection {
			Ok("Load Config") => {
				if let Err(e) = load_config() {
					eprintln!("Error loading configuration: {}", e);
				}
			},
			Ok("List Configs") => {
				if let Err(e) = list_config() {
					eprintln!("Error listing configurations: {}", e);
				}
			},
			Ok("Import") => {
				if let Err(e) = import_config() {
					eprintln!("Error importing configuration: {}", e);
				}
			},
			Ok("New Config") => new_config(),
			Ok("Back") => return Ok(()),
			Err(e) => {
				eprintln!("Error in menu selection: {}", e);
			},
			_ => (),
		}
	}
}

pub fn load_config() -> Result<()> {
	let list = list_configs()?;
	let selection = Select::new("Please select:", list).prompt()?;
	let conf = activate_config(selection)?;
	println!("{:?}", conf);
	Ok(())
}

pub fn list_config() -> Result<()> {
	let list = list_configs()?;
	println!();
	println!("{:?}", list);
	println!();
	Ok(())
}

pub fn import_config() -> Result<()> {
	let path = Text::new("Please enter path to Config:").prompt()?;
	let name = Text::new("Please enter name of Config:").prompt()?;
	Ok(config_from_file(path, name)?)
}

pub fn new_config() {
	println!();
	println!("Not implemented yet");
	println!();
}
