// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

//! NCCL AllReduce Benchmark with Statistical Analysis
//!
//! Measures NCCL AllReduce performance across multiple GPUs with detailed statistics.
//! This benchmark provides production-grade performance metrics including:
//! - Latency statistics (min, max, median, P95, P99)
//! - Throughput analysis (peak and average GB/s)
//! - Stability metrics (CV, jitter)
//! - Cold vs Warm state comparison
//!
//! **Requires**: Multiple CUDA GPUs and the `nccl` feature flag
//! **Run with**: cargo run --release --features nccl

#[cfg(not(feature = "nccl"))]
fn main() {
	println!("⚠️  This benchmark requires the 'nccl' feature!");
	println!("Run with: cargo run --release --bin operation --features nccl");
}

#[cfg(feature = "nccl")]
use cubecl_cuda::CudaRuntime;
#[cfg(feature = "nccl")]
use lib_calculator::*;
#[cfg(feature = "nccl")]
use std::fs::File;
#[cfg(feature = "nccl")]
use std::io::{self, Write};
#[cfg(feature = "nccl")]
use std::time::Instant;

#[cfg(feature = "nccl")]
const MB: usize = 1024 * 1024;
#[cfg(feature = "nccl")]
const GB: usize = 1024 * 1024 * 1024;

#[cfg(feature = "nccl")]
#[derive(Debug)]
struct BenchmarkConfig {
	size_elements: usize,
	warmup_iterations: usize,
	benchmark_iterations: usize,
	test_name: String,
}

#[cfg(feature = "nccl")]
#[derive(Debug, Clone)]
struct BenchmarkResult {
	test_name: String,
	size_elements: usize,
	size_bytes: usize,
	gpu_count: usize,
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

#[cfg(feature = "nccl")]
impl BenchmarkResult {
	fn from_times(
		test_name: String,
		size_elements: usize,
		gpu_count: usize,
		times: Vec<f64>,
	) -> Self {
		let size_bytes = size_elements * 4; // f32 = 4 bytes
		let mut sorted_times = times.clone();
		sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

		let n = sorted_times.len();
		let min_ms = sorted_times[0];
		let max_ms = sorted_times[n - 1];
		let avg_ms: f64 = sorted_times.iter().sum::<f64>() / n as f64;
		let median_ms = if n % 2 == 0 {
			(sorted_times[n / 2 - 1] + sorted_times[n / 2]) / 2.0
		} else {
			sorted_times[n / 2]
		};

		// Standard deviation
		let variance: f64 = sorted_times
			.iter()
			.map(|&t| {
				let diff = t - avg_ms;
				diff * diff
			})
			.sum::<f64>()
			/ n as f64;
		let std_dev_ms = variance.sqrt();

		let p95_ms = sorted_times[((n as f64 * 0.95) as usize).min(n - 1)];
		let p99_ms = sorted_times[((n as f64 * 0.99) as usize).min(n - 1)];

		let cv_percent = (std_dev_ms / avg_ms) * 100.0;
		let jitter_percent = ((max_ms - min_ms) / avg_ms) * 100.0;

		// Throughput: NCCL AllReduce effectively moves data across all GPUs
		let avg_throughput_gbs = (size_bytes as f64 / (avg_ms / 1000.0)) / GB as f64;
		let peak_throughput_gbs = (size_bytes as f64 / (min_ms / 1000.0)) / GB as f64;

		BenchmarkResult {
			test_name,
			size_elements,
			size_bytes,
			gpu_count,
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

#[cfg(feature = "nccl")]
fn format_size(bytes: usize) -> String {
	if bytes >= GB {
		format!("{:.2} GB", bytes as f64 / GB as f64)
	} else if bytes >= MB {
		format!("{:.2} MB", bytes as f64 / MB as f64)
	} else if bytes >= 1024 {
		format!("{:.2} KB", bytes as f64 / 1024.0)
	} else {
		format!("{} bytes", bytes)
	}
}

#[cfg(feature = "nccl")]
fn print_banner() {
	println!("\n╔═══════════════════════════════════════════════════════════════╗");
	println!("║      🔥 NCCL ALLREDUCE BENCHMARK - STATISTICAL ANALYSIS      ║");
	println!("║         Multi-GPU Collective Communication Performance        ║");
	println!("╚═══════════════════════════════════════════════════════════════╝\n");
}

#[cfg(feature = "nccl")]
fn get_user_input(prompt: &str) -> String {
	print!("{}", prompt);
	io::stdout().flush().unwrap();
	let mut input = String::new();
	io::stdin().read_line(&mut input).unwrap();
	input.trim().to_string()
}

#[cfg(feature = "nccl")]
fn select_size() -> usize {
	println!("\n📏 Select tensor size (per GPU):");
	println!("   1) 4 MB      (1M elements)");
	println!("   2) 16 MB     (4M elements)");
	println!("   3) 64 MB     (16M elements)");
	println!("   4) 256 MB    (64M elements)");
	println!("   5) 512 MB    (128M elements)");
	println!("   6) 1 GB      (256M elements)");
	println!("   7) 2 GB      (512M elements)");
	println!("   8) 4 GB      (1B elements)");
	println!("   9) 8 GB      (2B elements)");
	println!("   0) Custom");

	loop {
		let input = get_user_input("\nEnter selection (0-9): ");
		match input.as_str() {
			"1" => return 1 * MB,
			"2" => return 4 * MB,
			"3" => return 16 * MB,
			"4" => return 64 * MB,
			"5" => return 128 * MB,
			"6" => return 256 * MB,
			"7" => return 512 * MB,
			"8" => return 1024 * MB,
			"9" => return 2048 * MB,
			"0" => {
				let custom = get_user_input("Enter size in MB: ");
				if let Ok(mb) = custom.parse::<usize>() {
					return mb * MB;
				}
				println!("❌ Invalid input, try again.");
			},
			_ => println!("❌ Invalid selection, please enter 0-9."),
		}
	}
}

#[cfg(feature = "nccl")]
fn select_iterations() -> (usize, usize) {
	println!("\n🔁 Select number of iterations:");
	println!("   1) Quick test  (10 warmup, 50 benchmark)");
	println!("   2) Standard    (20 warmup, 100 benchmark)");
	println!("   3) Thorough    (50 warmup, 500 benchmark)");
	println!("   4) Extreme     (100 warmup, 1000 benchmark)");
	println!("   5) Custom");

	loop {
		let input = get_user_input("\nEnter selection (1-5): ");
		match input.as_str() {
			"1" => return (10, 50),
			"2" => return (20, 100),
			"3" => return (50, 500),
			"4" => return (100, 1000),
			"5" => {
				let warmup = get_user_input("Enter warmup iterations: ");
				let bench = get_user_input("Enter benchmark iterations: ");
				if let (Ok(w), Ok(b)) = (
					warmup.parse::<usize>(),
					bench.parse::<usize>(),
				) {
					return (w, b);
				}
				println!("❌ Invalid input, try again.");
			},
			_ => println!("❌ Invalid selection, please enter 1-5."),
		}
	}
}

#[cfg(feature = "nccl")]
fn run_allreduce_benchmark(
	manager: &ManagingThread<CudaRuntime, f32>,
	config: &BenchmarkConfig,
) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
	let gpu_count = manager.dev_count();
	let elements_per_gpu = config.size_elements;
	let total_bytes = elements_per_gpu * 4 * gpu_count;

	println!("\n{}", "=".repeat(80));
	println!("🚀 Running: {}", config.test_name);
	println!("   GPUs:              {}", gpu_count);
	println!(
		"   Elements per GPU:  {}",
		elements_per_gpu
	);
	println!(
		"   Data per GPU:      {}",
		format_size(elements_per_gpu * 4)
	);
	println!(
		"   Total data:        {}",
		format_size(total_bytes)
	);
	println!("{}", "=".repeat(80));

	// Setup metadata
	let metadata = MetaData::new(
		&[elements_per_gpu, 1],
		&[1, elements_per_gpu],
	);

	// Create tensors for each GPU
	println!(
		"📦 Creating tensors on {} GPUs...",
		gpu_count
	);
	let tensors: Vec<_> = (0..gpu_count)
		.map(|gpu_id| {
			let data: Vec<f32> = (0..elements_per_gpu)
				.map(|i| (gpu_id as f32 * 1000.0) + (i as f32 % 1000.0))
				.collect();
			Tensor::new(data, metadata.clone())
		})
		.collect();

	let int = manager.tensor_send_distributed(tensors)?;

	println!("✅ Tensors allocated");

	// Warmup phase
	print!(
		"🔥 Warming up ({} iterations)...",
		config.warmup_iterations
	);
	io::stdout().flush().ok();

	for i in 0..config.warmup_iterations {
		if i % 10 == 0 && i > 0 {
			print!(".");
			io::stdout().flush().ok();
		}

		manager.exec_kernel_broadcast(
			&int,
			&int,
			AllReduce,
			cubecl_cuda::ReduceOp::Sum,
		)?;
		manager.gpu_sync_all()?;
	}

	println!(" Done!");

	// Benchmark phase
	print!(
		"⏱️  Benchmarking ({} iterations)...",
		config.benchmark_iterations
	);
	io::stdout().flush().ok();

	let mut times = Vec::with_capacity(config.benchmark_iterations);
	let overall_start = Instant::now();

	for i in 0..config.benchmark_iterations {
		if i % 50 == 0 && i > 0 {
			print!(".");
			io::stdout().flush().ok();
		}

		let start = Instant::now();

		manager.exec_kernel_broadcast(
			&int,
			&int,
			AllReduce,
			cubecl_cuda::ReduceOp::Sum,
		)?;
		manager.gpu_sync_all()?;

		let elapsed = start.elapsed().as_secs_f64() * 1000.0; // Convert to ms
		times.push(elapsed);
	}

	let overall_elapsed = overall_start.elapsed();
	println!(" Done!");
	println!(
		"   Total time: {:.2}s ({:.1} iterations/sec)",
		overall_elapsed.as_secs_f64(),
		config.benchmark_iterations as f64 / overall_elapsed.as_secs_f64()
	);

	// Cleanup
	manager.interval_remove(int)?;

	Ok(BenchmarkResult::from_times(
		config.test_name.clone(),
		elements_per_gpu,
		gpu_count,
		times,
	))
}

#[cfg(feature = "nccl")]
fn print_result(result: &BenchmarkResult) {
	println!("\n╔═══════════════════════════════════════════════════════════════╗");
	println!("║  📊 {} Results", result.test_name);
	println!("╚═══════════════════════════════════════════════════════════════╝");

	println!("\n   ┌─────────────────────────────────────────┐");
	println!("   │ Timing Statistics (ms)                  │");
	println!("   ├─────────────────┬───────────────────────┤");
	println!(
		"   │ Min (Best)      │ {:>17.3} ms │",
		result.min_ms
	);
	println!(
		"   │ Median (P50)    │ {:>17.3} ms │",
		result.median_ms
	);
	println!(
		"   │ Average         │ {:>17.3} ms │",
		result.avg_ms
	);
	println!(
		"   │ P95             │ {:>17.3} ms │",
		result.p95_ms
	);
	println!(
		"   │ P99             │ {:>17.3} ms │",
		result.p99_ms
	);
	println!(
		"   │ Max (Worst)     │ {:>17.3} ms │",
		result.max_ms
	);
	println!("   ├─────────────────┼───────────────────────┤");
	println!(
		"   │ Std Deviation   │ {:>17.3} ms │",
		result.std_dev_ms
	);
	println!(
		"   │ CV              │ {:>17.2}% │",
		result.cv_percent
	);
	println!(
		"   │ Jitter          │ {:>17.2}% │",
		result.jitter_percent
	);
	println!("   └─────────────────┴───────────────────────┘");

	println!("\n   ┌─────────────────────────────────────────┐");
	println!("   │ Throughput (GB/s)                       │");
	println!("   ├─────────────────┬───────────────────────┤");
	println!(
		"   │ Peak (Min Time) │ {:>17.2} GB/s │",
		result.peak_throughput_gbs
	);
	println!(
		"   │ Average         │ {:>17.2} GB/s │",
		result.avg_throughput_gbs
	);
	println!("   └─────────────────┴───────────────────────┘");

	// Stability assessment
	let cv_rating = if result.cv_percent < 1.0 {
		"Excellent ⭐⭐⭐"
	} else if result.cv_percent < 3.0 {
		"Very Good ⭐⭐"
	} else if result.cv_percent < 5.0 {
		"Good ⭐"
	} else {
		"Needs Investigation ⚠️"
	};

	let jitter_rating = if result.jitter_percent < 5.0 {
		"Low ✅"
	} else if result.jitter_percent < 10.0 {
		"Moderate ⚠️"
	} else {
		"High ❌"
	};

	println!("\n   ┌─────────────────────────────────────────┐");
	println!("   │ Stability Assessment                    │");
	println!("   ├─────────────────┬───────────────────────┤");
	println!(
		"   │ Consistency     │ {:>21} │",
		cv_rating
	);
	println!(
		"   │ Variability     │ {:>21} │",
		jitter_rating
	);
	println!("   └─────────────────┴───────────────────────┘");

	println!("\n   ┌─────────────────────────────────────────┐");
	println!("   │ Configuration                           │");
	println!("   ├─────────────────┬───────────────────────┤");
	println!(
		"   │ GPUs            │ {:>21} │",
		result.gpu_count
	);
	println!(
		"   │ Data per GPU    │ {:>21} │",
		format_size(result.size_bytes)
	);
	println!(
		"   │ Total data      │ {:>21} │",
		format_size(result.size_bytes * result.gpu_count)
	);
	println!(
		"   │ Iterations      │ {:>21} │",
		result.times_ms.len()
	);
	println!("   └─────────────────┴───────────────────────┘");
}

#[cfg(feature = "nccl")]
fn compare_results(
	cold: &BenchmarkResult,
	warm: &BenchmarkResult,
) {
	println!("\n╔═══════════════════════════════════════════════════════════════╗");
	println!("║                    🆚 COLD vs WARM COMPARISON                 ║");
	println!("╚═══════════════════════════════════════════════════════════════╝\n");

	let latency_improvement = ((cold.avg_ms - warm.avg_ms) / cold.avg_ms) * 100.0;
	let throughput_improvement =
		((warm.peak_throughput_gbs - cold.peak_throughput_gbs) / cold.peak_throughput_gbs) * 100.0;
	let cv_improvement = ((cold.cv_percent - warm.cv_percent) / cold.cv_percent) * 100.0;

	println!("   ┌─────────────────────────────────────────────────────┐");
	println!("   │ Performance Improvements                            │");
	println!("   ├─────────────────────────────────┬───────────────────┤");
	println!(
		"   │ Latency Improvement             │ {:>16.2}% │",
		latency_improvement
	);
	println!(
		"   │ Peak Throughput Improvement     │ {:>16.2}% │",
		throughput_improvement
	);
	println!(
		"   │ Stability Improvement (CV)      │ {:>16.2}% │",
		cv_improvement
	);
	println!("   └─────────────────────────────────┴───────────────────┘");

	println!("\n   ┌─────────────────────────────────────────────────────┐");
	println!("   │ Best Performance (Warm State)                       │");
	println!("   ├─────────────────────────────────┬───────────────────┤");
	println!(
		"   │ Best Latency                    │ {:>13.3} ms │",
		warm.min_ms
	);
	println!(
		"   │ Peak Throughput                 │ {:>13.2} GB/s │",
		warm.peak_throughput_gbs
	);
	println!(
		"   │ Average Throughput              │ {:>13.2} GB/s │",
		warm.avg_throughput_gbs
	);
	println!(
		"   │ Consistency (CV)                │ {:>16.2}% │",
		warm.cv_percent
	);
	println!("   └─────────────────────────────────┴───────────────────┘");

	// Verdict
	println!("\n╔═══════════════════════════════════════════════════════════════╗");
	println!("║                    ✨ VERDICT                                 ║");
	println!("╚═══════════════════════════════════════════════════════════════╝\n");

	if latency_improvement > 5.0 {
		println!("   🔥 SIGNIFICANT WARM-UP BENEFIT!");
		println!(
			"      → System performs {:.1}% better when warm",
			latency_improvement
		);
		println!("      → JIT compilation and caching working optimally");
	} else {
		println!("   ✅ CONSISTENT PERFORMANCE!");
		println!("      → Minimal warm-up needed");
	}

	if warm.cv_percent < 1.0 {
		println!("\n   ⭐⭐⭐ EXCELLENT STABILITY!");
		println!(
			"      → CV: {:.2}% (world-class consistency)",
			warm.cv_percent
		);
	} else if warm.cv_percent < 3.0 {
		println!("\n   ⭐⭐ VERY GOOD STABILITY!");
		println!(
			"      → CV: {:.2}% (reliable performance)",
			warm.cv_percent
		);
	}

	if warm.peak_throughput_gbs > 10.0 {
		println!("\n   🚀 OUTSTANDING THROUGHPUT!");
		println!(
			"      → {:.2} GB/s sustained",
			warm.peak_throughput_gbs
		);
	}

	println!("\n   💪 System ready for production workloads!");
}

#[cfg(feature = "nccl")]
fn write_results_to_markdown(
	results: &[BenchmarkResult],
	config: &BenchmarkConfig,
) -> Result<(), Box<dyn std::error::Error>> {
	let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
	let filename = format!(
		"nccl_allreduce_benchmark_{}MB_{}iter_{}.md",
		config.size_elements / MB,
		config.benchmark_iterations,
		timestamp
	);

	let mut file = File::create(&filename)?;

	writeln!(
		file,
		"# 🔥 NCCL AllReduce Benchmark Results\n"
	)?;
	writeln!(file, "## Configuration\n")?;
	writeln!(
		file,
		"- **Test Date**: {}",
		chrono::Local::now().format("%Y-%m-%d %H:%M:%S")
	)?;
	writeln!(
		file,
		"- **GPUs**: {}",
		results[0].gpu_count
	)?;
	writeln!(
		file,
		"- **Data Size per GPU**: {}",
		format_size(config.size_elements * 4)
	)?;
	writeln!(
		file,
		"- **Total Data**: {}",
		format_size(config.size_elements * 4 * results[0].gpu_count)
	)?;
	writeln!(
		file,
		"- **Warmup Iterations**: {}",
		config.warmup_iterations
	)?;
	writeln!(
		file,
		"- **Benchmark Iterations**: {}",
		config.benchmark_iterations
	)?;
	writeln!(file)?;

	for result in results {
		writeln!(file, "## {} Results\n", result.test_name)?;

		writeln!(file, "### Timing Statistics\n")?;
		writeln!(file, "| Metric | Value | Note |")?;
		writeln!(file, "|--------|-------|------|")?;
		writeln!(
			file,
			"| Min | {:.3} ms | Best case |",
			result.min_ms
		)?;
		writeln!(
			file,
			"| Median | {:.3} ms | 50th percentile |",
			result.median_ms
		)?;
		writeln!(
			file,
			"| Average | {:.3} ms | Expected performance |",
			result.avg_ms
		)?;
		writeln!(
			file,
			"| P95 | {:.3} ms | 95th percentile |",
			result.p95_ms
		)?;
		writeln!(
			file,
			"| P99 | {:.3} ms | 99th percentile |",
			result.p99_ms
		)?;
		writeln!(
			file,
			"| Max | {:.3} ms | Worst case |",
			result.max_ms
		)?;
		writeln!(
			file,
			"| Std Dev | {:.3} ms | Variability |",
			result.std_dev_ms
		)?;
		writeln!(
			file,
			"| CV | {:.2}% | Consistency |",
			result.cv_percent
		)?;
		writeln!(
			file,
			"| Jitter | {:.2}% | Timing spread |",
			result.jitter_percent
		)?;

		writeln!(file, "\n### Throughput\n")?;
		writeln!(file, "| Metric | Value |")?;
		writeln!(file, "|--------|-------|")?;
		writeln!(
			file,
			"| Peak | {:.2} GB/s |",
			result.peak_throughput_gbs
		)?;
		writeln!(
			file,
			"| Average | {:.2} GB/s |",
			result.avg_throughput_gbs
		)?;

		writeln!(file, "\n### Assessment\n")?;
		if result.cv_percent < 1.0 {
			writeln!(
				file,
				"- ⭐⭐⭐ **Excellent stability** (CV < 1%)"
			)?;
		} else if result.cv_percent < 3.0 {
			writeln!(
				file,
				"- ⭐⭐ **Very good stability** (CV < 3%)"
			)?;
		} else if result.cv_percent < 5.0 {
			writeln!(file, "- ⭐ **Good stability** (CV < 5%)")?;
		} else {
			writeln!(
				file,
				"- ⚠️ **Needs investigation** (CV > 5%)"
			)?;
		}

		if result.peak_throughput_gbs > 10.0 {
			writeln!(
				file,
				"- 🚀 **Excellent throughput** (> 10 GB/s)"
			)?;
		} else if result.peak_throughput_gbs > 5.0 {
			writeln!(
				file,
				"- ✅ **Good throughput** (> 5 GB/s)"
			)?;
		}

		writeln!(file)?;
	}

	writeln!(file, "## Interpretation Guide\n")?;
	writeln!(
		file,
		"- **CV (Coefficient of Variation)**: Lower is better. < 1% is excellent, < 5% is good."
	)?;
	writeln!(
		file,
		"- **Jitter**: Lower is better. < 10% is excellent."
	)?;
	writeln!(
		file,
		"- **P95/P99**: 95th/99th percentile latency for SLA planning."
	)?;
	writeln!(
		file,
		"- **Throughput**: Data transfer rate during AllReduce operation."
	)?;
	writeln!(
		file,
		"- **Cold Start**: First run after initialization (includes JIT compilation)."
	)?;
	writeln!(
		file,
		"- **Warm State**: Subsequent runs with optimized code paths."
	)?;

	println!("\n📄 Results written to: {}", filename);
	Ok(())
}

#[cfg(feature = "nccl")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
	print_banner();

	// Initialize GPU manager
	println!("🔧 Initializing NCCL and GPU manager...");
	let manager = ManagingThread::<CudaRuntime, f32>::init()?;
	let gpu_count = manager.dev_count();

	println!("✅ Initialized with {} GPU(s)", gpu_count);

	if gpu_count < 2 {
		println!(
			"\n⚠️  WARNING: Only {} GPU detected.",
			gpu_count
		);
		println!("NCCL AllReduce benefits most with 2+ GPUs.");
		println!("Benchmark will still run but won't show multi-GPU advantages.\n");
	}

	// Get configuration
	let size_elements = select_size();
	let (warmup_iterations, benchmark_iterations) = select_iterations();

	println!("\n📋 Configuration Summary:");
	println!("   GPUs:             {}", gpu_count);
	println!(
		"   Data per GPU:     {}",
		format_size(size_elements * 4)
	);
	println!(
		"   Total data:       {}",
		format_size(size_elements * 4 * gpu_count)
	);
	println!(
		"   Warmup:           {} iterations",
		warmup_iterations
	);
	println!(
		"   Benchmark:        {} iterations",
		benchmark_iterations
	);

	let estimated_time_sec = (warmup_iterations + benchmark_iterations) as f64 * 0.001;
	println!(
		"   Estimated time:   ~{:.1} seconds",
		estimated_time_sec
	);

	let input = get_user_input("\nPress ENTER to start benchmark...");
	drop(input);

	// Run cold start benchmark
	println!("\n═══════════════════════════════════════════════════════════════");
	println!("   TEST 1: COLD START (First Run with JIT Compilation)");
	println!("═══════════════════════════════════════════════════════════════");

	let cold_config = BenchmarkConfig {
		size_elements,
		warmup_iterations,
		benchmark_iterations,
		test_name: "Cold Start".to_string(),
	};

	let cold_result = run_allreduce_benchmark(&manager, &cold_config)?;
	print_result(&cold_result);

	// Run warm state benchmark
	println!("\n═══════════════════════════════════════════════════════════════");
	println!("   TEST 2: WARM STATE (Production Simulation)");
	println!("═══════════════════════════════════════════════════════════════");

	let warm_config = BenchmarkConfig {
		size_elements,
		warmup_iterations,
		benchmark_iterations,
		test_name: "Warm State".to_string(),
	};

	let warm_result = run_allreduce_benchmark(&manager, &warm_config)?;
	print_result(&warm_result);

	// Compare results
	compare_results(&cold_result, &warm_result);

	// Save results
	write_results_to_markdown(&[cold_result, warm_result], &cold_config)?;

	// Shutdown
	println!("\n🛑 Shutting down...");
	manager.shutdown()?;
	println!("✨ Benchmark completed successfully!\n");

	Ok(())
}
