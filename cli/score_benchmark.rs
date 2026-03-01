//! # Log Dot Product Benchmark CLI
//!
//! Minimal CLI tool for benchmarking multiple implementations of axis_log_dot_product.
//! Compares performance and verifies numerical accuracy between implementations.
//!
//! ## Usage Examples
//!
//! ```bash
//! # Run benchmark with predefined test cases
//! cargo run --bin score-benchmark
//! ```

use numpy::ndarray::{Array1, Array2};
use rand::prelude::*;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

// Import from the library
#[cfg(target_arch = "aarch64")]
use alphadia_search_rs::score::neon::{axis_log_dot_product_neon, axis_log_dot_product_neon_v2};
use alphadia_search_rs::score::scalar::axis_log_dot_product_scalar;

// Constants
const ACCURACY_TOLERANCE: f32 = 0.20;
const DEFAULT_ITERATIONS: usize = 200;
const RANDOM_ROUNDS: usize = 10;

#[derive(Debug, Clone)]
struct TestCase {
    rows: usize,
    cols: usize,
    name: String,
    iterations: usize,
}

#[derive(Debug)]
struct BenchmarkResult {
    implementation: String,
    test_case: String,
    time_seconds: f64,
    speedup: f64,
    accuracy_verified: bool,
    avg_rel_error: f32,
    max_rel_error: f32,
}

#[derive(Debug)]
struct BenchmarkConfig {
    test_cases: Vec<TestCase>,
}

impl BenchmarkConfig {
    fn default() -> Self {
        Self {
            test_cases: vec![
                TestCase {
                    rows: 12,
                    cols: 100,
                    name: "12x100".to_string(),
                    iterations: DEFAULT_ITERATIONS,
                },
                TestCase {
                    rows: 12,
                    cols: 1000,
                    name: "12x1000".to_string(),
                    iterations: DEFAULT_ITERATIONS,
                },
                TestCase {
                    rows: 48,
                    cols: 1000,
                    name: "48x1000".to_string(),
                    iterations: DEFAULT_ITERATIONS,
                },
                TestCase {
                    rows: 12,
                    cols: 10000,
                    name: "12x10000".to_string(),
                    iterations: DEFAULT_ITERATIONS,
                },
                TestCase {
                    rows: 48,
                    cols: 10000,
                    name: "48x10000".to_string(),
                    iterations: DEFAULT_ITERATIONS,
                },
                TestCase {
                    rows: 12,
                    cols: 100000,
                    name: "12x100000".to_string(),
                    iterations: DEFAULT_ITERATIONS,
                },
                TestCase {
                    rows: 48,
                    cols: 100000,
                    name: "48x100000".to_string(),
                    iterations: DEFAULT_ITERATIONS,
                },
            ],
        }
    }
}

// Define a type for log dot product functions
type LogDotProductFunction = fn(&Array2<f32>, &[f32]) -> Array1<f32>;

fn generate_test_data(rows: usize, cols: usize) -> (Array2<f32>, Vec<f32>) {
    let mut rng = rand::rng();

    // Generate random data array with positive values
    let mut data = Array2::<f32>::zeros((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            data[[i, j]] = rng.random_range(0.1..10.0);
        }
    }

    // Generate random weights
    let weights: Vec<f32> = (0..rows).map(|_| rng.random_range(0.1..2.0)).collect();

    (data, weights)
}

fn verify_accuracy(
    scalar_result: &Array1<f32>,
    simd_result: &Array1<f32>,
    tolerance: f32,
) -> (bool, f32, f32) {
    let mut max_rel_diff: f32 = 0.0;
    let mut sum_rel_diff: f32 = 0.0;
    let mut all_within_tolerance = true;

    for i in 0..scalar_result.len() {
        let diff = (scalar_result[i] - simd_result[i]).abs();
        let rel_diff = diff / scalar_result[i].abs();
        max_rel_diff = max_rel_diff.max(rel_diff);
        sum_rel_diff += rel_diff;

        if rel_diff > tolerance {
            all_within_tolerance = false;
        }
    }

    let avg_rel_diff = sum_rel_diff / scalar_result.len() as f32;
    (all_within_tolerance, avg_rel_diff, max_rel_diff)
}

fn get_available_implementations() -> Vec<(String, LogDotProductFunction)> {
    #[allow(unused_mut)]
    let mut implementations: Vec<(String, LogDotProductFunction)> =
        vec![("Scalar".to_string(), axis_log_dot_product_scalar)];

    // Add platform-specific implementations
    #[cfg(target_arch = "aarch64")]
    implementations.push(("NEON".to_string(), axis_log_dot_product_neon));

    #[cfg(target_arch = "aarch64")]
    implementations.push(("NEON-v2".to_string(), axis_log_dot_product_neon_v2));

    implementations
}

fn warmup_implementations(
    implementations: &[(String, LogDotProductFunction)],
    test_data: &Array2<f32>,
    test_weights: &[f32],
) {
    for (_, implementation) in implementations {
        let _ = implementation(test_data, test_weights);
    }
}

fn benchmark_single_case(test_case: &TestCase) -> Vec<BenchmarkResult> {
    let implementations = get_available_implementations();

    // Generate all random datasets upfront
    let datasets: Vec<(Array2<f32>, Vec<f32>)> = (0..RANDOM_ROUNDS)
        .map(|_| generate_test_data(test_case.rows, test_case.cols))
        .collect();

    // Warmup on the first dataset
    warmup_implementations(&implementations, &datasets[0].0, &datasets[0].1);

    // Accumulate timing and accuracy per implementation
    let mut total_times = vec![0.0_f64; implementations.len()];
    let mut total_avg_err = vec![0.0_f32; implementations.len()];
    let mut worst_max_err = vec![0.0_f32; implementations.len()];
    let mut all_passed = vec![true; implementations.len()];

    for (data, weights) in &datasets {
        // Run scalar first to get the reference result
        let start = Instant::now();
        let mut scalar_result = Array1::zeros(data.ncols());
        for _ in 0..test_case.iterations {
            scalar_result = (implementations[0].1)(data, weights);
        }
        total_times[0] += start.elapsed().as_secs_f64();

        // Benchmark remaining implementations
        for (idx, (_, func)) in implementations.iter().enumerate().skip(1) {
            let start = Instant::now();
            let mut result = Array1::zeros(data.ncols());
            for _ in 0..test_case.iterations {
                result = func(data, weights);
            }
            total_times[idx] += start.elapsed().as_secs_f64();

            let (passed, avg_err, max_err) =
                verify_accuracy(&scalar_result, &result, ACCURACY_TOLERANCE);
            total_avg_err[idx] += avg_err;
            worst_max_err[idx] = worst_max_err[idx].max(max_err);
            if !passed {
                all_passed[idx] = false;
            }
        }
    }

    // Build results averaged over rounds
    let n = RANDOM_ROUNDS as f64;
    let baseline_time = total_times[0];

    implementations
        .iter()
        .enumerate()
        .map(|(idx, (name, _))| BenchmarkResult {
            implementation: name.clone(),
            test_case: test_case.name.clone(),
            time_seconds: total_times[idx] / n,
            speedup: baseline_time / total_times[idx],
            accuracy_verified: all_passed[idx],
            avg_rel_error: total_avg_err[idx] / n as f32,
            max_rel_error: worst_max_err[idx],
        })
        .collect()
}

fn print_results_table(results: &[BenchmarkResult]) {
    println!();
    println!(
        "========================================================================================"
    );
    println!(
        "{:<14} {:<12} {:>10} {:>10} {:>12} {:>12} {:>8}",
        "Impl", "Test Case", "Time (s)", "Speedup", "Avg Err (%)", "Max Err (%)", "Status"
    );
    println!(
        "========================================================================================"
    );

    for result in results {
        let status = if result.accuracy_verified {
            "PASS"
        } else {
            "FAIL"
        };

        println!(
            "{:<14} {:<12} {:>10.4} {:>9.2}x {:>12.2} {:>12.2} {:>8}",
            result.implementation,
            result.test_case,
            result.time_seconds,
            result.speedup,
            result.avg_rel_error * 100.0,
            result.max_rel_error * 100.0,
            status
        );
    }
    println!(
        "========================================================================================"
    );
}

fn run_benchmark_suite(config: &BenchmarkConfig) -> Vec<BenchmarkResult> {
    let mut all_results = Vec::new();

    for test_case in &config.test_cases {
        println!(
            "Running benchmark for {} ({} rounds x {} iterations)...",
            test_case.name, RANDOM_ROUNDS, test_case.iterations
        );
        let case_results = benchmark_single_case(test_case);
        all_results.extend(case_results);
    }

    all_results
}

fn save_results_to_tsv(results: &[BenchmarkResult], filename: &str) -> Result<(), std::io::Error> {
    let mut file = File::create(filename)?;

    writeln!(
        file,
        "Implementation\tTest Case\tTime (s)\tSpeedup\tAvg Err (%)\tMax Err (%)\tStatus"
    )?;

    for result in results {
        let status = if result.accuracy_verified {
            "PASS"
        } else {
            "FAIL"
        };

        writeln!(
            file,
            "{}\t{}\t{:.4}\t{:.2}\t{:.2}\t{:.2}\t{}",
            result.implementation,
            result.test_case,
            result.time_seconds,
            result.speedup,
            result.avg_rel_error * 100.0,
            result.max_rel_error * 100.0,
            status
        )?;
    }

    Ok(())
}

fn main() {
    println!("Log Dot Product Benchmark Tool");
    println!("Architecture: {}", std::env::consts::ARCH);
    println!("Random rounds per test case: {RANDOM_ROUNDS}");
    println!();

    let config = BenchmarkConfig::default();
    let results = run_benchmark_suite(&config);
    print_results_table(&results);

    // Save results to TSV file
    const TSV_FILENAME: &str = "score_benchmark.tsv";
    match save_results_to_tsv(&results, TSV_FILENAME) {
        Ok(()) => println!("\n✓ Results saved to {TSV_FILENAME}"),
        Err(e) => eprintln!("\n✗ Failed to save results to {TSV_FILENAME}: {e}"),
    }
}
