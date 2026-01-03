// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use cubecl_cuda::CudaRuntime;
use lib_calculator::*;

fn main() {
	let manager = ManagingThread::<CudaRuntime, f32>::init().unwrap();

	let metadata0 = MetaData::new(
		vec![8, 1].as_slice(),
		vec![1, 8].as_slice(),
	);
	let metadata1 = MetaData::new(
		vec![1, 1].as_slice(),
		vec![1, 8].as_slice(),
	);

	let md1 = MetaData::new(
		vec![8, 1].as_slice(),
		vec![2, 8].as_slice(),
	);
	let t2 = vec![
		Tensor::<f32>::new(
			vec![4.0, 5.0, 1.0, 7.0, 10.0, 5.0, 3.0, 8.0],
			metadata0.clone(),
		),
		Tensor::<f32>::new(
			vec![234.0, 100.0, 123.0, 123.0, 220.0, 1234.0, 122.0, 111.0],
			metadata0.clone(),
		),
	];

	let t0 = Tensor::<f32>::new(
		vec![234.0, 100.0, 123.0, 123.0, 220.0, 1234.0, 122.0, 111.0],
		metadata0.clone(),
	);

	let int0 = manager.tensor_send_broadcast(t0).unwrap();
	let int00 = manager.tensor_empty_broadcast(&metadata0).unwrap();
	let int1 = manager.tensor_empty_broadcast(&metadata1).unwrap();
	let int2 = manager.tensor_send_distributed(t2.clone()).unwrap();
	let int3 = manager.tensor_send_distributed(t2).unwrap();
	let int4 = manager.tensor_empty_broadcast(&md1).unwrap();

	manager.gpu_sync_all().unwrap();
	manager
		.exec_kernel_broadcast(&int0, &int00, Sort, 1)
		.unwrap();

	let resultss = manager.tensors_get_all(&int00).unwrap();

	for result in resultss {
		println!("################################");
		println!("################################");
		println!("Data: {:?}", result.data);
		println!("################################");
		println!("################################");
	}

	manager
		.exec_kernel_broadcast(&int4, &int1, Prod, 0)
		.unwrap();
	manager.gpu_sync_all().unwrap();

	let results = manager.tensors_get_all(&int1).unwrap();

	for result in results {
		println!("--------------------------------");
		println!("################################");
		println!("Data: {:?}", result.data);
		println!("################################");
		println!("--------------------------------");
	}

	manager.interval_remove(int1).unwrap();
	manager.interval_remove(int2).unwrap();
	manager.interval_remove(int3).unwrap();

	manager.shutdown().unwrap();
}
