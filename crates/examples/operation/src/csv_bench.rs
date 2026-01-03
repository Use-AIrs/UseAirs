// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

//! CSV-based GPU Sort Benchmark
//!
//! Benchmark sorting on CSV data or generated datasets up to 4GB.
//!
//! Usage:
//!   cargo run --release --bin csv_bench -- --size 1GB
//!   cargo run --release --bin csv_bench -- --size 512MB --verify
//!   cargo run --release --bin csv_bench -- --size 256MB --iterations 50 -o results.csv

use cubecl_cuda::CudaRuntime;
use cubecl_common::device::DeviceId;
use lib_calculator::*;
use std::io::{BufRead, BufReader, Write};
use std::fs::File;
use std::time::Instant;

const KB: usize = 1024;
const MB: usize = 1024 * KB;
const GB: usize = 1024 * MB;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let config = parse_args(&args)?;

    println!("\n========================================");
    println!("  GPU Sort Benchmark (CSV/Size Mode)");
    println!("========================================\n");

    // Load or generate data
    let data = match &config.mode {
        DataMode::Csv { path, column } => load_csv(path, column.as_deref())?,
        DataMode::Size(bytes) => generate_data(*bytes)?,
    };

    let element_count = data.len();
    let bytes = element_count * 4;
    println!("Data: {} elements ({:.2} GB)", element_count, bytes as f64 / GB as f64);
    println!("Iterations: {}", config.iterations);
    if config.verify { println!("Verify: enabled"); }
    println!();

    // Init GPU
    let manager = ManagingThread::<CudaRuntime, f32>::init()?;
    let dev_id = DeviceId::new(0, 0);
    println!("GPU initialized: {} device(s)\n", manager.dev_count());

    // Run verification if requested
    if config.verify {
        run_verification(&manager, dev_id, &data)?;
    }

    // Run benchmark
    let results = run_benchmark(&manager, dev_id, &data, config.iterations)?;

    // Print results
    print_results(&results, element_count);

    // Export if requested
    if let Some(output) = &config.output {
        export_results(output, &results, element_count)?;
        println!("\nResults saved to: {}", output);
    }

    manager.shutdown()?;
    Ok(())
}

#[derive(Debug)]
struct Config {
    mode: DataMode,
    iterations: usize,
    output: Option<String>,
    verify: bool,
}

#[derive(Debug)]
enum DataMode {
    Csv { path: String, column: Option<String> },
    Size(usize),
}

fn parse_args(args: &[String]) -> Result<Config, Box<dyn std::error::Error>> {
    let mut mode: Option<DataMode> = None;
    let mut iterations = 20;
    let mut output: Option<String> = None;
    let mut verify = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--size" | "-s" => {
                i += 1;
                let size = parse_size(&args.get(i).ok_or("missing size value")?)?;
                mode = Some(DataMode::Size(size));
            }
            "--csv" | "-c" => {
                i += 1;
                let path = args.get(i).ok_or("missing csv path")?.clone();
                mode = Some(DataMode::Csv { path, column: None });
            }
            "--column" => {
                i += 1;
                if let Some(DataMode::Csv { column, .. }) = &mut mode {
                    *column = Some(args.get(i).ok_or("missing column name")?.clone());
                }
            }
            "--iterations" | "-i" => {
                i += 1;
                iterations = args.get(i).ok_or("missing iterations")?.parse()?;
            }
            "--output" | "-o" => {
                i += 1;
                output = Some(args.get(i).ok_or("missing output path")?.clone());
            }
            "--verify" | "-v" => {
                verify = true;
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }

    let mode = mode.unwrap_or_else(|| {
        println!("No --size or --csv specified. Using default 256MB.\n");
        DataMode::Size(256 * MB)
    });

    Ok(Config { mode, iterations, output, verify })
}

fn parse_size(s: &str) -> Result<usize, Box<dyn std::error::Error>> {
    let s = s.to_uppercase();
    let (num, mult) = if s.ends_with("GB") {
        (s.trim_end_matches("GB"), GB)
    } else if s.ends_with("MB") {
        (s.trim_end_matches("MB"), MB)
    } else if s.ends_with("KB") {
        (s.trim_end_matches("KB"), KB)
    } else {
        (s.as_str(), 1)
    };
    let num: usize = num.trim().parse()?;
    let total = num * mult;
    if total > 4 * GB {
        return Err("Max size is 4GB".into());
    }
    Ok(total)
}

fn print_help() {
    println!("GPU Sort Benchmark - CSV/Size Mode");
    println!();
    println!("USAGE:");
    println!("  csv_bench [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("  --size, -s <SIZE>      Generate data of SIZE (e.g., 1GB, 512MB, 256KB)");
    println!("  --csv, -c <PATH>       Load data from CSV file");
    println!("  --column <NAME>        Column name to sort (default: first numeric column)");
    println!("  --iterations, -i <N>   Number of iterations (default: 20)");
    println!("  --output, -o <PATH>    Export results to CSV file");
    println!("  --verify, -v           Show first/last 64 values for verification");
    println!("  --help, -h             Show this help");
    println!();
    println!("EXAMPLES:");
    println!("  csv_bench --size 1GB");
    println!("  csv_bench --size 512MB --verify");
    println!("  csv_bench --size 256MB -i 50 -o results.csv");
}

fn load_csv(path: &str, column: Option<&str>) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    println!("Loading CSV: {}", path);
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let header = lines.next().ok_or("empty CSV")??;
    let columns: Vec<&str> = header.split(',').map(|s| s.trim()).collect();

    let col_idx = if let Some(name) = column {
        columns.iter().position(|&c| c == name)
            .ok_or_else(|| format!("column '{}' not found", name))?
    } else {
        0
    };

    println!("Using column: {} (index {})", columns[col_idx], col_idx);

    let mut data = Vec::new();
    for line in lines {
        let line = line?;
        let values: Vec<&str> = line.split(',').collect();
        if let Some(val) = values.get(col_idx) {
            if let Ok(num) = val.trim().parse::<f32>() {
                data.push(num);
            }
        }
    }

    println!("Loaded {} values", data.len());
    Ok(data)
}

fn generate_data(bytes: usize) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let count = bytes / 4;
    println!("Generating {} elements ({:.2} GB)...", count, bytes as f64 / GB as f64);

    let start = Instant::now();
    let data: Vec<f32> = (0..count)
        .map(|i| {
            let x = ((i as u64).wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)) as u32;
            (x as f32) / (u32::MAX as f32)
        })
        .collect();

    println!("Generated in {:.2}s", start.elapsed().as_secs_f64());
    Ok(data)
}

// ============================================================================
// VERIFICATION - Show first/last 64 values
// ============================================================================

fn run_verification(
    manager: &ManagingThread<CudaRuntime, f32>,
    dev_id: DeviceId,
    data: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let size = data.len();

    // Pad to power of 2
    let padded_size = ((size + 3) / 4 * 4).next_power_of_two();
    let mut padded_data = data.to_vec();
    padded_data.resize(padded_size, f32::MAX);

    // MetaData::new(stride, shape) - stride first!
    // Shape: [1, padded_size] - axis 1 has padded_size elements (sort axis)
    // Strides: [padded_size, 1] - axis 1 is contiguous
    let metadata = MetaData::new(&[padded_size, 1], &[1, padded_size]);
    let input_tensor = Tensor::new(padded_data.clone(), metadata.clone());
    let input_int = manager.tensor_send(dev_id, &input_tensor)?;
    let output_int = manager.tensor_empty(dev_id, &metadata)?;

    // Run Sort
    println!("========================================");
    println!("         VERIFICATION (Sort)");
    println!("========================================\n");

    manager.exec_kernel_on_gpu(dev_id, input_int, output_int, Sort, 1_usize)?;
    manager.gpu_sync(dev_id)?;

    let output_tensor = manager.tensor_get(&output_int, dev_id)?;
    let result = &output_tensor.data;

    // Show first 64
    println!("--- FIRST 64 VALUES (GPU sorted) ---");
    print_values(&result[..64.min(size)]);

    println!("\n--- LAST 64 VALUES (GPU sorted) ---");
    let start = if size > 64 { size - 64 } else { 0 };
    print_values(&result[start..size]);

    // Verify sorting
    let mut sorted = true;
    let mut first_error = None;
    for i in 1..size {
        if result[i] < result[i-1] {
            if first_error.is_none() {
                first_error = Some((i-1, result[i-1], result[i]));
            }
            sorted = false;
        }
    }
    if let Some((idx, a, b)) = first_error {
        println!("\n[ERROR] First unsorted at index {}: {} > {}", idx, a, b);
    }
    println!("\n--- SORT CHECK: {} ---", if sorted { "PASSED" } else { "FAILED" });

    // CPU reference (first/last 64)
    let mut cpu_sorted = data.to_vec();
    cpu_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    println!("\n\n--- CPU REFERENCE (first 64) ---");
    print_values(&cpu_sorted[..64.min(size)]);

    println!("\n--- CPU REFERENCE (last 64) ---");
    print_values(&cpu_sorted[start..size]);

    // Compare
    let mut match_count = 0;
    for i in 0..size {
        if (result[i] - cpu_sorted[i]).abs() < 1e-6 {
            match_count += 1;
        }
    }
    println!("\n--- MATCH: {}/{} ({:.2}%) ---", match_count, size, 100.0 * match_count as f64 / size as f64);

    // Export to CSV for analysis
    let csv_path = "sort_verify.csv";
    println!("\nExporting to {}...", csv_path);
    let mut file = File::create(csv_path)?;
    writeln!(file, "index,input,gpu_sorted,cpu_sorted,match")?;
    for i in 0..size {
        let m = if (result[i] - cpu_sorted[i]).abs() < 1e-6 { 1 } else { 0 };
        writeln!(file, "{},{:.8},{:.8},{:.8},{}", i, data[i], result[i], cpu_sorted[i], m)?;
    }
    println!("Exported {} rows to {}", size, csv_path);

    manager.interval_remove(input_int)?;
    manager.interval_remove(output_int)?;

    println!("\n");
    Ok(())
}

fn print_values(values: &[f32]) {
    for (i, chunk) in values.chunks(8).enumerate() {
        print!("{:4}: ", i * 8);
        for v in chunk {
            print!("{:10.6} ", v);
        }
        println!();
    }
}

// ============================================================================
// BENCHMARK
// ============================================================================

#[derive(Debug, Clone)]
struct BenchResult {
    times_ms: Vec<f64>,
    avg_ms: f64,
    min_ms: f64,
    max_ms: f64,
    std_ms: f64,
    throughput_gbs: f64,
}

fn run_benchmark(
    manager: &ManagingThread<CudaRuntime, f32>,
    dev_id: DeviceId,
    data: &[f32],
    iterations: usize,
) -> Result<BenchResult, Box<dyn std::error::Error>> {
    let size = data.len();
    let bytes = size * 4;

    let padded_size = ((size + 3) / 4 * 4).next_power_of_two();
    let mut padded_data = data.to_vec();
    padded_data.resize(padded_size, f32::MAX);

    // MetaData::new(stride, shape) - stride first!
    // Shape: [1, padded_size] - axis 1 has padded_size elements (sort axis)
    // Strides: [padded_size, 1] - axis 1 is contiguous
    let metadata = MetaData::new(&[padded_size, 1], &[1, padded_size]);
    let input_tensor = Tensor::new(padded_data, metadata.clone());
    let input_int = manager.tensor_send(dev_id, &input_tensor)?;

    // Warmup
    print!("Warmup...");
    std::io::stdout().flush()?;
    for _ in 0..5 {
        let output_int = manager.tensor_empty(dev_id, &metadata)?;
        manager.exec_kernel_on_gpu(dev_id, input_int, output_int, Sort, 1_usize)?;
        manager.gpu_sync(dev_id)?;
        manager.interval_remove(output_int)?;
    }
    println!(" done");

    // Benchmark
    print!("Running {} iterations: ", iterations);
    std::io::stdout().flush()?;

    let mut times = Vec::with_capacity(iterations);
    for i in 0..iterations {
        let output_int = manager.tensor_empty(dev_id, &metadata)?;

        let start = Instant::now();
        manager.exec_kernel_on_gpu(dev_id, input_int, output_int, Sort, 1_usize)?;
        manager.gpu_sync(dev_id)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        times.push(elapsed);
        manager.interval_remove(output_int)?;

        if (i + 1) % 5 == 0 {
            print!(".");
            std::io::stdout().flush()?;
        }
    }
    println!(" done\n");

    manager.interval_remove(input_int)?;

    // Stats
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min_ms = times[0];
    let max_ms = times[times.len() - 1];
    let avg_ms: f64 = times.iter().sum::<f64>() / times.len() as f64;
    let variance: f64 = times.iter().map(|t| (t - avg_ms).powi(2)).sum::<f64>() / times.len() as f64;
    let std_ms = variance.sqrt();
    let throughput_gbs = (bytes as f64 / GB as f64) / (avg_ms / 1000.0);

    Ok(BenchResult { times_ms: times, avg_ms, min_ms, max_ms, std_ms, throughput_gbs })
}

fn print_results(results: &BenchResult, element_count: usize) {
    let bytes = element_count * 4;
    println!("========================================");
    println!("               RESULTS");
    println!("========================================");
    println!();
    println!("Data Size:      {} elements ({:.3} GB)", element_count, bytes as f64 / GB as f64);
    println!();
    println!("Timing:");
    println!("  Average:      {:.3} ms", results.avg_ms);
    println!("  Min:          {:.3} ms", results.min_ms);
    println!("  Max:          {:.3} ms", results.max_ms);
    println!("  Std Dev:      {:.3} ms", results.std_ms);
    println!();
    println!("Throughput:     {:.2} GB/s", results.throughput_gbs);
    println!("                {:.1} M elements/s", (element_count as f64 / results.avg_ms) / 1000.0);
}

fn export_results(path: &str, results: &BenchResult, element_count: usize) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(path)?;
    writeln!(file, "iteration,time_ms")?;
    for (i, t) in results.times_ms.iter().enumerate() {
        writeln!(file, "{},{:.6}", i + 1, t)?;
    }
    writeln!(file)?;
    writeln!(file, "# Summary")?;
    writeln!(file, "# elements,{}", element_count)?;
    writeln!(file, "# avg_ms,{:.6}", results.avg_ms)?;
    writeln!(file, "# min_ms,{:.6}", results.min_ms)?;
    writeln!(file, "# max_ms,{:.6}", results.max_ms)?;
    writeln!(file, "# std_ms,{:.6}", results.std_ms)?;
    writeln!(file, "# throughput_gbs,{:.6}", results.throughput_gbs)?;
    Ok(())
}
