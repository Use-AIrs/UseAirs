// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

pub mod cfg;
pub mod error;

use mongodb::bson::{doc, Document};
use mongodb::{
	bson,
	sync::{Client, Collection},
};

use crate::cfg::Config;
use crate::error::{Result, StoreError};

pub fn config_from_file(
	path: String,
	name: String,
) -> Result<()> {
	let client = connect()?;
	let collection = client.database("use-ai").collection("configs");
	let config = Config::get_config(path)?;
	let mut doc = bson::to_document(&config).unwrap();
	doc.insert("name", name);
	doc.insert("active", false);

	let result = collection.insert_one(doc).run()?;
	println!(
		"Inserted a document with _id: {}",
		result.inserted_id
	);
	Ok(())
}

pub fn list_configs() -> Result<Vec<String>> {
	let client = connect()?;
	let collection: Collection<Document> = client.database("use-ai").collection("configs");

	let cursor = collection.find(Default::default()).run()?;

	let list = cursor
		.into_iter()
		.filter_map(|doc_result| doc_result.ok())
		.map(|doc| doc.get_str("name").unwrap().to_string())
		.collect::<Vec<String>>();

	Ok(list)
}

pub fn activate_config(name: String) -> Result<()> {
	let client = connect()?;

	let collection: Collection<Document> = client.database("use-ai").collection("configs");

	collection
		.update_many(
			doc! {},
			doc! { "$set": { "active": false } },
		)
		.run()?;

	collection
		.update_one(
			doc! { "name": name },
			doc! { "$set": { "active": true } },
		)
		.run()?;

	Ok(())
}

pub fn get_active_config() -> Result<Config> {
	let client = connect()?;
	let collection: Collection<Config> = client.database("use-ai").collection("configs");
	let cursor = collection.find_one(doc! { "active": true }).run()?;
	match cursor {
		Some(config) => Ok(config),
		None => Err(StoreError::NoConfigActive),
	}
}

pub fn cfg_runtime() -> Result<String> {
	let client = connect()?;
	let collection: Collection<Document> = client.database("use-ai").collection("runtimes");
	let doc = collection.find_one(doc! { "active": true }).run()?;
	match doc {
		Some(document) => match document.get_str("runtime") {
			Ok(runtime) => Ok(runtime.to_string()),
			Err(_) => Err(StoreError::NoConfigActive),
		},
		None => Err(StoreError::NoConfigActive),
	}
}

fn connect() -> Result<Client> {
	let uri = "mongodb://127.0.0.1:27017/";
	let client = Client::with_uri_str(uri)?;
	Ok(client)
}
