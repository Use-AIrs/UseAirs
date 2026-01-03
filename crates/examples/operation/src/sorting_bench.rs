// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

//! GPU Sorting Benchmark - Performance Analysis
//!
//! Measures Radix Sort performance with detailed statistics:
//! - Latency (min, max, median, P95, P99)
//! - Throughput (GB/s)
//! - Stability metrics (CV, jitter)
//!
//! **Run with**: cargo run --release --bin sorting_bench [SIZE] [ITERATIONS]
//!   SIZE: 64k, 256k, 1m, 4m, 16m, or exact number (default: all sizes)
//!   ITERATIONS: number of benchmark iterations (default: 100)

use cubecl::prelude::*;
use cubecl_common::rand::{get_seeded_rng, Rng};
use cubecl_cuda::{CudaDevice, CudaRuntime};
use std::env;
use std::io::{self, Write};
use std::time::Instant;

use lib_calculator::{f2u, rx1_hist, rx2_prefix, rx3_scatter, RxCfg};

const KB: usize = 1024;
const MB: usize = 1024 * 1024;
const GB: usize = 1024 * 1024 * 1024;

#[derive(Debug, Clone)]
struct BenchmarkResult {
	name: String,
	size_elements: usize,
	size_bytes: usize,
	iterations: usize,
	times_ms: Vec<f64>,
	min_ms: f64,
	max_ms: f64,
	median_ms: f64,
	avg_ms: f64,
	std_dev_ms: f64,
	p95_ms: f64,
	p99_ms: f64,
	cv_percent: f64,
	jitter_percent: f64,
	avg_throughput_gbs: f64,
	peak_throughput_gbs: f64,
}

impl BenchmarkResult {
	fn from_times(name: String, size_elements: usize, times: Vec<f64>) -> Self {
		let size_bytes = size_elements * 4;
		let mut sorted = times.clone();
		sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

		let n = sorted.len();
		let min_ms = sorted[0];
		let max_ms = sorted[n - 1];
		let avg_ms: f64 = sorted.iter().sum::<f64>() / n as f64;
		let median_ms = if n % 2 == 0 {
			(sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
		} else {
			sorted[n / 2]
		};

		let variance: f64 = sorted.iter().map(|&t| (t - avg_ms).powi(2)).sum::<f64>() / n as f64;
		let std_dev_ms = variance.sqrt();

		let p95_ms = sorted[((n as f64 * 0.95) as usize).min(n - 1)];
		let p99_ms = sorted[((n as f64 * 0.99) as usize).min(n - 1)];

		let cv_percent = (std_dev_ms / avg_ms) * 100.0;
		let jitter_percent = ((max_ms - min_ms) / avg_ms) * 100.0;

		let gb = size_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
		let avg_throughput_gbs = gb / (avg_ms / 1000.0);
		let peak_throughput_gbs = gb / (min_ms / 1000.0);

		BenchmarkResult {
			name,
			size_elements,
			size_bytes,
			iterations: n,
			times_ms: times,
			min_ms,
			max_ms,
			median_ms,
			avg_ms,
			std_dev_ms,
			p95_ms,
			p99_ms,
			cv_percent,
			jitter_percent,
			avg_throughput_gbs,
			peak_throughput_gbs,
		}
	}
}

fn parse_size(s: &str) -> Option<usize> {
	let s = s.to_lowercase();
	if s.ends_with("gb") {
		s[..s.len() - 2].parse::<usize>().ok().map(|n| n * GB / 4)
	} else if s.ends_with("mb") {
		s[..s.len() - 2].parse::<usize>().ok().map(|n| n * MB / 4)
	} else if s.ends_with("kb") {
		s[..s.len() - 2].parse::<usize>().ok().map(|n| n * KB / 4)
	} else if s.ends_with('g') {
		s[..s.len() - 1].parse::<usize>().ok().map(|n| n * GB / 4)
	} else if s.ends_with('m') {
		s[..s.len() - 1].parse::<usize>().ok().map(|n| n * MB / 4)
	} else if s.ends_with('k') {
		s[..s.len() - 1].parse::<usize>().ok().map(|n| n * KB / 4)
	} else {
		s.parse::<usize>().ok()
	}
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let args: Vec<String> = env::args().collect();

	// Parse optional size argument (in bytes: KB, MB, GB or raw element count)
	let size: usize = if args.len() > 1 {
		parse_size(&args[1]).unwrap_or_else(|| {
			eprintln!("Invalid size: {}. Use 64kb, 256mb, 1gb, or element count", args[1]);
			std::process::exit(1);
		})
	} else {
		1 * MB // default 1M elements = 4MB
	};

	// Parse optional iterations argument
	let benchmark_iterations: usize = if args.len() > 2 {
		args[2].parse().unwrap_or_else(|_| {
			eprintln!("Invalid iterations: {}. Use a positive number", args[2]);
			std::process::exit(1);
		})
	} else {
		100
	};

	let warmup_iterations = benchmark_iterations.min(20);

	print_banner();

	let client = CudaRuntime::client(&CudaDevice::new(0, None, 0));
	let props = client.properties();
	println!("GPU: CUDA Device 0 (warp_size={})\n", props.hardware.plane_size_max);

	println!("Configuration:");
	println!("  Size:       {} elements ({})", size, format_size(size * 4));
	println!("  Warmup:     {} iterations", warmup_iterations);
	println!("  Benchmark:  {} iterations", benchmark_iterations);
	println!();

	// Run benchmark
	let result = benchmark_radix_sort::<CudaRuntime>(
		&client,
		&format_size(size * 4),
		size,
		warmup_iterations,
		benchmark_iterations,
	)?;
	print_result(&result);

	println!("\nBenchmark complete!\n");
	Ok(())
}

fn print_banner() {
	println!("\n========================================");
	println!("  GPU RADIX SORT BENCHMARK");
	println!("  Performance Analysis");
	println!("========================================\n");
}

fn format_size(bytes: usize) -> String {
	if bytes >= 1024 * 1024 * 1024 {
		format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
	} else if bytes >= MB {
		format!("{:.2} MB", bytes as f64 / MB as f64)
	} else if bytes >= KB {
		format!("{:.2} KB", bytes as f64 / KB as f64)
	} else {
		format!("{} B", bytes)
	}
}

fn benchmark_radix_sort<R: Runtime>(
	client: &ComputeClient<R::Server>,
	name: &str,
	n: usize,
	warmup_iterations: usize,
	benchmark_iterations: usize,
) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
	let size_bytes = n * 4;

	println!("----------------------------------------");
	println!("Radix Sort: {} elements ({})", n, format_size(size_bytes));
	println!("----------------------------------------");

	let cfg = RxCfg::new(n as u32);

	// Generate random input data
	let mut rng = get_seeded_rng();
	let in_f32: Vec<f32> = (0..n).map(|_| rng.random::<f32>() * 10000.0 - 5000.0).collect();
	let in_u32: Vec<u32> = in_f32.iter().map(|&v| f2u(v)).collect();

	// Warmup
	print!("Warming up ({} iterations)...", warmup_iterations);
	io::stdout().flush().ok();

	for i in 0..warmup_iterations {
		run_radix_sort_pass::<R>(client, &in_u32, &cfg, n);
		if i % 5 == 4 {
			print!(".");
			io::stdout().flush().ok();
		}
	}
	println!(" done");

	// Benchmark
	print!("Benchmarking ({} iterations)...", benchmark_iterations);
	io::stdout().flush().ok();

	let mut times = Vec::with_capacity(benchmark_iterations);
	for i in 0..benchmark_iterations {
		let elapsed = run_radix_sort_pass::<R>(client, &in_u32, &cfg, n);
		times.push(elapsed);
		if i % 20 == 19 {
			print!(".");
			io::stdout().flush().ok();
		}
	}
	println!(" done");

	Ok(BenchmarkResult::from_times(
		format!("Radix Sort {}", name),
		n,
		times,
	))
}

fn run_radix_sort_pass<R: Runtime>(
	client: &ComputeClient<R::Server>,
	in_u32: &[u32],
	cfg: &RxCfg,
	n: usize,
) -> f64 {
	let mut buf_a = client.create(bytemuck::cast_slice(in_u32));
	let mut buf_b = client.empty(n * 4);
	let hist_h = client.empty(cfg.hist_sz() * 4);
	let offs_h = client.empty(cfg.hist_sz() * 4);

	let start = Instant::now();

	for pass in 0..8u32 {
		let (inp, out) = if pass % 2 == 0 {
			(&buf_a, &mut buf_b)
		} else {
			(&buf_b, &mut buf_a)
		};

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
	start.elapsed().as_secs_f64() * 1000.0
}

fn print_result(result: &BenchmarkResult) {
	let kernels_per_sort = 24; // 8 passes × 3 kernels
	let avg_kernel_us = (result.avg_ms * 1000.0) / kernels_per_sort as f64;

	println!();
	println!("  Timing Statistics:");
	println!("    Min (best):    {:>8.3} ms", result.min_ms);
	println!("    Median (P50):  {:>8.3} ms", result.median_ms);
	println!("    Average:       {:>8.3} ms", result.avg_ms);
	println!("    Avg/Kernel:    {:>8.2} µs  (24 kernels)", avg_kernel_us);
	println!("    P95:           {:>8.3} ms", result.p95_ms);
	println!("    P99:           {:>8.3} ms", result.p99_ms);
	println!("    Max (worst):   {:>8.3} ms", result.max_ms);
	println!("    Std Dev:       {:>8.3} ms", result.std_dev_ms);

	println!();
	println!("  Stability:");
	println!("    CV:            {:>8.2}%", result.cv_percent);
	println!("    Jitter:        {:>8.2}%", result.jitter_percent);

	println!();
	println!("  Throughput:");
	println!("    Peak:          {:>8.2} GB/s", result.peak_throughput_gbs);
	println!("    Average:       {:>8.2} GB/s", result.avg_throughput_gbs);

	let stability = if result.cv_percent < 3.0 {
		"Excellent"
	} else if result.cv_percent < 5.0 {
		"Good"
	} else if result.cv_percent < 10.0 {
		"Fair"
	} else {
		"Variable"
	};
	println!();
	println!("  Assessment: {}", stability);
	println!();
}
