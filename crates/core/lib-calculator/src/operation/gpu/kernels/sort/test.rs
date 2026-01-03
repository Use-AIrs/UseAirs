// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use cubecl::prelude::*;
use cubecl_common::rand::{get_seeded_rng, Rng};

use super::cfg::ComptimeCfg;
use super::entry::{bitonic_argsort_kernel, bitonic_sort_kernel};

fn get_seeds() -> [u32; 4] {
	let mut rng = get_seeded_rng();
	[rng.random(), rng.random(), rng.random(), rng.random()]
}

#[cube]
fn taus_step(z: u32, s1: u32, s2: u32, s3: u32, m: u32) -> u32 {
	let b = (z << s1) ^ z;
	let b = b >> s2;
	((z & m) << s3) ^ b
}

#[cube]
fn taus_step_0(z: u32) -> u32 {
	taus_step(z, 13u32, 19u32, 12u32, 4294967294u32)
}

#[cube]
fn taus_step_1(z: u32) -> u32 {
	taus_step(z, 2u32, 25u32, 4u32, 4294967288u32)
}

#[cube]
fn taus_step_2(z: u32) -> u32 {
	taus_step(z, 3u32, 11u32, 17u32, 4294967280u32)
}

#[cube]
fn lcg_step(z: u32) -> u32 {
	z * 1664525u32 + 1013904223u32
}

#[cube]
fn to_unit_interval(int_random: u32) -> f32 {
	let shifted = int_random >> 8;
	f32::cast_from(shifted) / 16777216.0
}

#[cube]
fn next_random(s0: &mut u32, s1: &mut u32, s2: &mut u32, s3: &mut u32) -> f32 {
	*s0 = taus_step_0(*s0);
	*s1 = taus_step_1(*s1);
	*s2 = taus_step_2(*s2);
	*s3 = lcg_step(*s3);
	to_unit_interval(*s0 ^ *s1 ^ *s2 ^ *s3)
}

#[cube(launch_unchecked)]
fn fill_random_kernel(
	output: &mut Tensor<Line<f32>>,
	seed_0: u32,
	seed_1: u32,
	seed_2: u32,
	seed_3: u32,
	scale: f32,
) {
	let idx = ABSOLUTE_POS;
	let thread_seed = 1000000007u32 * idx;

	let mut s0 = thread_seed + seed_0;
	let mut s1 = thread_seed + seed_1;
	let mut s2 = thread_seed + seed_2;
	let mut s3 = thread_seed + seed_3;

	let v0 = next_random(&mut s0, &mut s1, &mut s2, &mut s3) * scale;
	let v1 = next_random(&mut s0, &mut s1, &mut s2, &mut s3) * scale;
	let v2 = next_random(&mut s0, &mut s1, &mut s2, &mut s3) * scale;
	let v3 = next_random(&mut s0, &mut s1, &mut s2, &mut s3) * scale;

	let mut line = Line::<f32>::empty(4u32);
	line[0u32] = v0;
	line[1u32] = v1;
	line[2u32] = v2;
	line[3u32] = v3;

	output[idx] = line;
}

fn format_size(bytes: usize) -> String {
	if bytes >= 1024 * 1024 {
		format!("{:.2} MiB", bytes as f64 / (1024.0 * 1024.0))
	} else if bytes >= 1024 {
		format!("{:.2} KiB", bytes as f64 / 1024.0)
	} else {
		format!("{} B", bytes)
	}
}

fn verify_sorted(data: &[f32]) -> (bool, Option<usize>) {
	for (i, window) in data.windows(2).enumerate() {
		if window[0] > window[1] {
			return (false, Some(i));
		}
	}
	(true, None)
}

pub fn test_sort_2048<R: Runtime>(client: &ComputeClient<R::Server>) {
	let num_elements = 2048usize;
	let line_size = 4u8;
	let num_lines = num_elements / line_size as usize;
	let size_bytes = num_elements * std::mem::size_of::<f32>();

	println!("\n=== Sort Test: 2048 elements ({}) ===", format_size(size_bytes));

	let input_data: Vec<f32> = (0..num_elements).map(|i| (num_elements - i) as f32).collect();
	let input_handle = client.create(bytemuck::cast_slice(&input_data));
	let output_handle = client.empty(size_bytes);

	let props = client.properties();
	let warp_size = props.hardware.plane_size_max;
	let cfg = ComptimeCfg {
		line_size: line_size as u32,
		row_size: num_lines as u32,
		warp_size,
		shmem_len: (num_lines as u32).next_power_of_two(),
		cycles_per_axis: 1,
	};

	unsafe {
		bitonic_sort_kernel::launch_unchecked::<f32, R>(
			client,
			CubeCount::Static(1, 1, 1),
			CubeDim::new(512, 1, 1),
			TensorArg::from_raw_parts::<f32>(&input_handle, &[num_lines], &[1], line_size),
			TensorArg::from_raw_parts::<f32>(&output_handle, &[num_lines], &[1], line_size),
			cfg,
			1u32,
		);
	}

	let output_bytes = client.read_one(output_handle.clone());
	let output_f32: Vec<f32> = bytemuck::cast_slice(&output_bytes).to_vec();
	let (is_sorted, violation) = verify_sorted(&output_f32);

	println!("Output[0..8]: {:?}", &output_f32[..8]);
	println!("Sorted: {}", if is_sorted { "✅ PASS" } else { "❌ FAIL" });

	if let Some(pos) = violation {
		eprintln!("First violation at {}: {} > {}", pos, output_f32[pos], output_f32[pos + 1]);
	}
	assert!(is_sorted, "Output should be sorted!");
}

pub fn test_argsort_2048<R: Runtime>(client: &ComputeClient<R::Server>) {
	let num_elements = 2048usize;
	let line_size = 4u8;
	let num_lines = num_elements / line_size as usize;
	let size_bytes = num_elements * std::mem::size_of::<f32>();

	println!("\n=== ArgSort Test: 2048 elements ({}) ===", format_size(size_bytes));

	let input_data: Vec<f32> = (0..num_elements).map(|i| (num_elements - i) as f32).collect();
	let input_handle = client.create(bytemuck::cast_slice(&input_data));
	let output_values = client.empty(size_bytes);
	let output_indices = client.empty(num_elements * std::mem::size_of::<u32>());

	let props = client.properties();
	let warp_size = props.hardware.plane_size_max;
	let cfg = ComptimeCfg {
		line_size: line_size as u32,
		row_size: num_lines as u32,
		warp_size,
		shmem_len: (num_lines as u32).next_power_of_two(),
		cycles_per_axis: 1,
	};

	unsafe {
		bitonic_argsort_kernel::launch_unchecked::<f32, R>(
			client,
			CubeCount::Static(1, 1, 1),
			CubeDim::new(512, 1, 1),
			TensorArg::from_raw_parts::<f32>(&input_handle, &[num_lines], &[1], line_size),
			TensorArg::from_raw_parts::<f32>(&output_values, &[num_lines], &[1], line_size),
			TensorArg::from_raw_parts::<u32>(&output_indices, &[num_lines], &[1], line_size),
			cfg,
			1u32,
		);
	}

	let val_bytes = client.read_one(output_values.clone());
	let val_f32: Vec<f32> = bytemuck::cast_slice(&val_bytes).to_vec();
	let idx_bytes = client.read_one(output_indices.clone());
	let idx_u32: Vec<u32> = bytemuck::cast_slice(&idx_bytes).to_vec();

	let (is_sorted, _) = verify_sorted(&val_f32);
	let indices_valid = idx_u32.iter().enumerate().all(|(pos, &idx)| {
		let idx = idx as usize;
		if idx >= input_data.len() { return false; }
		(input_data[idx] - val_f32[pos]).abs() < 0.001
	});

	println!("Values[0..8]: {:?}", &val_f32[..8]);
	println!("Indices[0..8]: {:?}", &idx_u32[..8]);
	println!("Values sorted: {}", if is_sorted { "✅" } else { "❌" });
	println!("Indices valid: {}", if indices_valid { "✅ PASS" } else { "❌ FAIL" });

	assert!(is_sorted, "Values should be sorted!");
	assert!(indices_valid, "Indices should be valid!");
}

pub fn test_sort_pow2<R: Runtime>(client: &ComputeClient<R::Server>, num_elements: usize, use_random: bool) {
	assert!(num_elements <= 4096, "Bitonic sort only supports <= 4k elements (single-pass)");

	let line_size = 4u8;
	let num_lines = num_elements / line_size as usize;
	let size_bytes = num_elements * std::mem::size_of::<f32>();

	println!("\n=== Sort Test: {} elements ({}) ===", num_elements, format_size(size_bytes));

	let input_handle = if use_random {
		let handle = client.empty(size_bytes);
		let seeds = get_seeds();
		unsafe {
			fill_random_kernel::launch_unchecked::<R>(
				client,
				CubeCount::Static(num_lines as u32, 1, 1),
				CubeDim::new(1, 1, 1),
				TensorArg::from_raw_parts::<f32>(&handle, &[num_lines], &[1], line_size),
				ScalarArg::new(seeds[0]),
				ScalarArg::new(seeds[1]),
				ScalarArg::new(seeds[2]),
				ScalarArg::new(seeds[3]),
				ScalarArg::new(10000.0f32),
			);
		}
		handle
	} else {
		let input_data: Vec<f32> = (0..num_elements).map(|i| (num_elements - i) as f32).collect();
		client.create(bytemuck::cast_slice(&input_data))
	};

	let output_handle = client.empty(size_bytes);
	let props = client.properties();
	let warp_size = props.hardware.plane_size_max;
	let shmem_len = std::cmp::min((num_lines as u32).next_power_of_two(), 1024);
	let cfg = ComptimeCfg {
		line_size: line_size as u32,
		row_size: num_lines as u32,
		warp_size,
		shmem_len,
		cycles_per_axis: 1,
	};

	let threads = shmem_len;
	let start = std::time::Instant::now();

	unsafe {
		bitonic_sort_kernel::launch_unchecked::<f32, R>(
			client,
			CubeCount::Static(1, 1, 1),
			CubeDim::new(threads, 1, 1),
			TensorArg::from_raw_parts::<f32>(&input_handle, &[num_lines as i32 as usize], &[1i32 as usize], line_size),
			TensorArg::from_raw_parts::<f32>(&output_handle, &[num_lines as i32 as usize], &[1i32 as usize], line_size),
			cfg,
			1u32,
		);
	}

	client.sync();
	let elapsed = start.elapsed();
	let throughput_mb = size_bytes as f64 / elapsed.as_secs_f64() / (1024.0 * 1024.0);

	println!("Time: {:?} ({:.1} MiB/s)", elapsed, throughput_mb);

	let output_bytes = client.read_one(output_handle.clone());
	let output_f32: Vec<f32> = bytemuck::cast_slice(&output_bytes).to_vec();
	let (is_sorted, violation) = verify_sorted(&output_f32);

	let show = std::cmp::min(16, num_elements);
	println!("Output[0..{}]: {:?}", show, &output_f32[..show]);
	println!("Sorted: {}", if is_sorted { "✅ PASS" } else { "❌ FAIL" });

	if let Some(pos) = violation {
		eprintln!("First violation at {}: {} > {}", pos, output_f32[pos], output_f32[pos + 1]);
	}
	assert!(is_sorted, "Output should be sorted!");
}

pub fn test_sort_4k<R: Runtime>(client: &ComputeClient<R::Server>) {
	test_sort_pow2::<R>(client, 4096, false); 
}

pub fn run_all_tests<R: Runtime>(client: &ComputeClient<R::Server>) {
	println!("\n========================================");
	println!("  Bitonic Sort Tests (128 - 4K)");
	println!("========================================");

	test_sort_2048::<R>(client);
	test_argsort_2048::<R>(client);
	test_sort_4k::<R>(client);

	println!("\n========================================");
	println!("  All bitonic sort tests PASSED!");
	println!("========================================\n");
}

use super::radix::{f2u, rx1_hist, rx2_prefix, rx3_scatter, rx3a_scatter, u2f, RxCfg};

pub fn test_radix_sort<R: Runtime>(client: &ComputeClient<R::Server>, n: usize) {
	println!("\n=== Radix Sort Test: {} elements ===", n);

	let cfg = RxCfg::new(n as u32);
	let mut rng = get_seeded_rng();
	let in_f32: Vec<f32> = (0..n).map(|_| rng.random::<f32>() * 10000.0 - 5000.0).collect();
	let in_u32: Vec<u32> = in_f32.iter().map(|&v| f2u(v)).collect();

	let mut buf_a = client.create(bytemuck::cast_slice(&in_u32));
	let mut buf_b = client.empty(n * 4);
	let hist_h = client.empty(cfg.hist_sz() * 4);
	let offs_h = client.empty(cfg.hist_sz() * 4);

	for pass in 0..8u32 {
		let (inp, out) = if pass % 2 == 0 { (&buf_a, &mut buf_b) } else { (&buf_b, &mut buf_a) };

		unsafe {
			rx1_hist::launch_unchecked::<R>(
				client,
				CubeCount::Static(cfg.nb, 1, 1),
				CubeDim::new(cfg.bs, 1, 1),
				TensorArg::from_raw_parts::<u32>(inp, &[n], &[1], 1),
				TensorArg::from_raw_parts::<u32>(&hist_h, &[cfg.hist_sz()], &[1], 1),
				ScalarArg::new(pass),
				ScalarArg::new(n as u32),
				cfg.bs,
			);
			rx2_prefix::launch_unchecked::<R>(
				client,
				CubeCount::Static(1, 1, 1),
				CubeDim::new(16, 1, 1),
				TensorArg::from_raw_parts::<u32>(&hist_h, &[cfg.hist_sz()], &[1], 1),
				TensorArg::from_raw_parts::<u32>(&offs_h, &[cfg.hist_sz()], &[1], 1),
				ScalarArg::new(cfg.nb),
			);
			rx3_scatter::launch_unchecked::<R>(
				client,
				CubeCount::Static(cfg.nb, 1, 1),
				CubeDim::new(cfg.bs, 1, 1),
				TensorArg::from_raw_parts::<u32>(inp, &[n], &[1], 1),
				TensorArg::from_raw_parts::<u32>(out, &[n], &[1], 1),
				TensorArg::from_raw_parts::<u32>(&offs_h, &[cfg.hist_sz()], &[1], 1),
				ScalarArg::new(pass),
				ScalarArg::new(n as u32),
				cfg.bs,
			);
		}
	}

	client.sync();
	let res_bytes = client.read_one(buf_a.clone());
	let res_u32: Vec<u32> = bytemuck::cast_slice(&res_bytes).to_vec();
	let res_f32: Vec<f32> = res_u32.iter().map(|&v| u2f(v)).collect();

	let (ok, viol) = verify_sorted(&res_f32);
	println!("Output[0..8]: {:?}", &res_f32[..std::cmp::min(8, n)]);
	println!("Sorted: {}", if ok { "✅ PASS" } else { "❌ FAIL" });

	if let Some(i) = viol {
		eprintln!("Violation at {}: {} > {}", i, res_f32[i], res_f32[i + 1]);
	}
	assert!(ok, "Radix sort failed!");
}

#[cube(launch_unchecked)]
fn init_indices(out: &mut Tensor<u32>, n: u32) {
	let gid = ABSOLUTE_POS;
	if gid < n {
		out[gid] = gid;
	}
}

pub fn test_radix_argsort<R: Runtime>(client: &ComputeClient<R::Server>, n: usize) {
	println!("\n=== Radix ArgSort Test: {} elements ===", n);

	let cfg = RxCfg::new(n as u32);
	let mut rng = get_seeded_rng();
	let in_f32: Vec<f32> = (0..n).map(|_| rng.random::<f32>() * 10000.0 - 5000.0).collect();
	let in_u32: Vec<u32> = in_f32.iter().map(|&v| f2u(v)).collect();

	let mut buf_v_a = client.create(bytemuck::cast_slice(&in_u32));
	let mut buf_v_b = client.empty(n * 4);
	let mut buf_i_a = client.empty(n * 4);
	let mut buf_i_b = client.empty(n * 4);
	let hist_h = client.empty(cfg.hist_sz() * 4);
	let offs_h = client.empty(cfg.hist_sz() * 4);

	unsafe {
		init_indices::launch_unchecked::<R>(
			client,
			CubeCount::Static(cfg.nb, 1, 1),
			CubeDim::new(cfg.bs, 1, 1),
			TensorArg::from_raw_parts::<u32>(&buf_i_a, &[n], &[1], 1),
			ScalarArg::new(n as u32),
		);
	}

	for pass in 0..8u32 {
		let (inp_v, out_v, inp_i, out_i) = if pass % 2 == 0 {
			(&buf_v_a, &mut buf_v_b, &buf_i_a, &mut buf_i_b)
		} else {
			(&buf_v_b, &mut buf_v_a, &buf_i_b, &mut buf_i_a)
		};

		unsafe {
			rx1_hist::launch_unchecked::<R>(
				client,
				CubeCount::Static(cfg.nb, 1, 1),
				CubeDim::new(cfg.bs, 1, 1),
				TensorArg::from_raw_parts::<u32>(inp_v, &[n], &[1], 1),
				TensorArg::from_raw_parts::<u32>(&hist_h, &[cfg.hist_sz()], &[1], 1),
				ScalarArg::new(pass),
				ScalarArg::new(n as u32),
				cfg.bs,
			);
			rx2_prefix::launch_unchecked::<R>(
				client,
				CubeCount::Static(1, 1, 1),
				CubeDim::new(16, 1, 1),
				TensorArg::from_raw_parts::<u32>(&hist_h, &[cfg.hist_sz()], &[1], 1),
				TensorArg::from_raw_parts::<u32>(&offs_h, &[cfg.hist_sz()], &[1], 1),
				ScalarArg::new(cfg.nb),
			);
			rx3a_scatter::launch_unchecked::<R>(
				client,
				CubeCount::Static(cfg.nb, 1, 1),
				CubeDim::new(cfg.bs, 1, 1),
				TensorArg::from_raw_parts::<u32>(inp_v, &[n], &[1], 1),
				TensorArg::from_raw_parts::<u32>(inp_i, &[n], &[1], 1),
				TensorArg::from_raw_parts::<u32>(out_v, &[n], &[1], 1),
				TensorArg::from_raw_parts::<u32>(out_i, &[n], &[1], 1),
				TensorArg::from_raw_parts::<u32>(&offs_h, &[cfg.hist_sz()], &[1], 1),
				ScalarArg::new(pass),
				ScalarArg::new(n as u32),
				cfg.bs,
			);
		}
	}

	client.sync();
	let res_v_bytes = client.read_one(buf_v_a.clone());
	let res_v_u32: Vec<u32> = bytemuck::cast_slice(&res_v_bytes).to_vec();
	let res_v_f32: Vec<f32> = res_v_u32.iter().map(|&v| u2f(v)).collect();

	let res_i_bytes = client.read_one(buf_i_a.clone());
	let res_i_u32: Vec<u32> = bytemuck::cast_slice(&res_i_bytes).to_vec();

	let (ok, viol) = verify_sorted(&res_v_f32);
	let indices_valid = res_i_u32.iter().enumerate().all(|(pos, &idx)| {
		let idx = idx as usize;
		if idx >= in_f32.len() { return false; }
		(in_f32[idx] - res_v_f32[pos]).abs() < 0.001
	});

	println!("Values[0..8]: {:?}", &res_v_f32[..std::cmp::min(8, n)]);
	println!("Indices[0..8]: {:?}", &res_i_u32[..std::cmp::min(8, n)]);
	println!("Values sorted: {}", if ok { "✅" } else { "❌" });
	println!("Indices valid: {}", if indices_valid { "✅ PASS" } else { "❌ FAIL" });

	if let Some(i) = viol {
		eprintln!("Violation at {}: {} > {}", i, res_v_f32[i], res_v_f32[i + 1]);
	}
	assert!(ok, "Radix argsort values failed!");
	assert!(indices_valid, "Radix argsort indices failed!");
}

#[cfg(test)]
mod test_cuda {
	use super::*;
	use cubecl::Runtime;
	use cubecl_cuda::{CudaDevice, CudaRuntime};

	fn get_client(device_index: usize) -> ComputeClient<<CudaRuntime as Runtime>::Server> {
		CudaRuntime::client(&CudaDevice::new(device_index as u16, None, device_index))
	}

	#[test]
	fn cuda_sort_2048() {
		test_sort_2048::<CudaRuntime>(&get_client(0));
	}

	#[test]
	fn cuda_argsort_2048() {
		test_argsort_2048::<CudaRuntime>(&get_client(0));
	}

	#[test]
	fn cuda_sort_4k() {
		test_sort_4k::<CudaRuntime>(&get_client(0));
	}

	#[test]
	fn cuda_all_bitonic() {
		run_all_tests::<CudaRuntime>(&get_client(0));
	}

	#[test]
	fn cuda_radix_sort_8k() {
		test_radix_sort::<CudaRuntime>(&get_client(0), 8192);
	}

	#[test]
	fn cuda_radix_sort_64k() {
		test_radix_sort::<CudaRuntime>(&get_client(0), 65536);
	}

	#[test]
	fn cuda_radix_sort_1m() {
		test_radix_sort::<CudaRuntime>(&get_client(0), 1048576);
	}

	#[test]
	fn cuda_radix_argsort_8k() {
		test_radix_argsort::<CudaRuntime>(&get_client(0), 8192);
	}

	#[test]
	fn cuda_radix_argsort_64k() {
		test_radix_argsort::<CudaRuntime>(&get_client(0), 65536);
	}

	#[test]
	fn cuda_radix_argsort_1m() {
		test_radix_argsort::<CudaRuntime>(&get_client(0), 1048576);
	}
}
