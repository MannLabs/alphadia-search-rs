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
use alphadia_search_rs::score::neon::axis_log_dot_product_neon;
use alphadia_search_rs::score::scalar::axis_log_dot_product_scalar;

// Constants
const ACCURACY_TOLERANCE: f32 = 0.20;
const DEFAULT_ITERATIONS: usize = 200;

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

    // Future implementations can be added here:
    // #[cfg(target_arch = "x86_64")]
    // {
    //     if is_x86_feature_detected!("avx2") {
    //         implementations.push(("AVX2".to_string(), axis_log_dot_product_avx2));
    //     }
    //     if is_x86_feature_detected!("avx512f") {
    //         implementations.push(("AVX512".to_string(), axis_log_dot_product_avx512));
    //     }
    // }

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

fn benchmark_implementation(
    implementation: LogDotProductFunction,
    test_data: &Array2<f32>,
    test_weights: &[f32],
    iterations: usize,
) -> (Array1<f32>, f64) {
    let start = Instant::now();
    let mut result = Array1::zeros(test_data.ncols());
    for _ in 0..iterations {
        result = implementation(test_data, test_weights);
    }
    let time_seconds = start.elapsed().as_secs_f64();
    (result, time_seconds)
}

fn create_benchmark_result(
    implementation_name: String,
    test_case_name: String,
    time_seconds: f64,
    speedup: f64,
    result: &Array1<f32>,
    scalar_result: Option<&Array1<f32>>,
) -> BenchmarkResult {
    let (accuracy_verified, avg_rel_error, max_rel_error) = match scalar_result {
        Some(scalar_ref) => verify_accuracy(scalar_ref, result, ACCURACY_TOLERANCE),
        None => (true, 0.0, 0.0), // First implementation (scalar) is always considered accurate
    };

    BenchmarkResult {
        implementation: implementation_name,
        test_case: test_case_name,
        time_seconds,
        speedup,
        accuracy_verified,
        avg_rel_error,
        max_rel_error,
    }
}

fn benchmark_single_case(test_case: &TestCase) -> Vec<BenchmarkResult> {
    let mut results = Vec::new();
    let (test_data, test_weights) = generate_test_data(test_case.rows, test_case.cols);

    // Get available implementations
    let implementations = get_available_implementations();

    // Warm up all implementations
    warmup_implementations(&implementations, &test_data, &test_weights);

    let mut baseline_time = 0.0;
    let mut scalar_result: Option<Array1<f32>> = None;

    // Benchmark all implementations
    for (i, (name, implementation)) in implementations.iter().enumerate() {
        let (result, time_seconds) = benchmark_implementation(
            *implementation,
            &test_data,
            &test_weights,
            test_case.iterations,
        );

        // Save baseline time and result from first implementation (scalar)
        if i == 0 {
            baseline_time = time_seconds;
            scalar_result = Some(result.clone());
        }

        let speedup = baseline_time / time_seconds;

        let benchmark_result = create_benchmark_result(
            name.clone(),
            test_case.name.clone(),
            time_seconds,
            speedup,
            &result,
            scalar_result.as_ref(),
        );

        results.push(benchmark_result);
    }

    results
}

fn format_error_percentage(error: f32) -> String {
    if error == 0.0 {
        "-".to_string()
    } else {
        format!("{:.2}", error * 100.0)
    }
}

fn print_implementation_table(implementation_name: &str, results: &[BenchmarkResult]) {
    println!("\n{implementation_name} Implementation Results:");
    println!("==========================================================================");
    println!(
        "{:<12} {:>10} {:>10} {:>12} {:>12} {:>8}",
        "Test Case", "Time (s)", "Speedup", "Avg Err (%)", "Max Err (%)", "Status"
    );
    println!("==========================================================================");

    for result in results
        .iter()
        .filter(|r| r.implementation == implementation_name)
    {
        let status = if result.accuracy_verified {
            "✓ PASS"
        } else {
            "✗ FAIL"
        };
        let avg_err_pct = format_error_percentage(result.avg_rel_error);
        let max_err_pct = format_error_percentage(result.max_rel_error);

        println!(
            "{:<12} {:>10.4} {:>9.2}x {:>12} {:>12} {:>8}",
            result.test_case, result.time_seconds, result.speedup, avg_err_pct, max_err_pct, status
        );
    }
    println!("==========================================================================");
}

fn run_benchmark_suite(config: &BenchmarkConfig) -> Vec<BenchmarkResult> {
    let mut all_results = Vec::new();

    // Run benchmarks for each test case
    for test_case in &config.test_cases {
        println!(
            "Running benchmark for {} with {} iterations...",
            test_case.name, test_case.iterations
        );
        let case_results = benchmark_single_case(test_case);
        all_results.extend(case_results);
    }

    all_results
}

fn save_results_to_tsv(results: &[BenchmarkResult], filename: &str) -> Result<(), std::io::Error> {
    let mut file = File::create(filename)?;

    // Write TSV header
    writeln!(
        file,
        "Implementation\tTest Case\tTime (s)\tSpeedup\tAvg Err (%)\tMax Err (%)\tStatus"
    )?;

    // Write data rows
    for result in results {
        let status = if result.accuracy_verified {
            "PASS"
        } else {
            "FAIL"
        };
        let avg_err_pct = if result.avg_rel_error == 0.0 {
            "-".to_string()
        } else {
            format!("{:.2}", result.avg_rel_error * 100.0)
        };
        let max_err_pct = if result.max_rel_error == 0.0 {
            "-".to_string()
        } else {
            format!("{:.2}", result.max_rel_error * 100.0)
        };

        writeln!(
            file,
            "{}\t{}\t{:.4}\t{:.2}\t{}\t{}\t{}",
            result.implementation,
            result.test_case,
            result.time_seconds,
            result.speedup,
            avg_err_pct,
            max_err_pct,
            status
        )?;
    }

    Ok(())
}

fn print_results(results: &[BenchmarkResult]) {
    // Print results table for each available implementation
    let available_implementations = get_available_implementations();
    for (implementation_name, _) in available_implementations {
        print_implementation_table(&implementation_name, results);
    }
}

fn main() {
    println!("Log Dot Product Benchmark Tool");
    println!("Architecture: {}", std::env::consts::ARCH);
    println!();

    let config = BenchmarkConfig::default();
    let results = run_benchmark_suite(&config);
    print_results(&results);

    // Save results to TSV file
    const TSV_FILENAME: &str = "score_benchmark.tsv";
    match save_results_to_tsv(&results, TSV_FILENAME) {
        Ok(()) => println!("\n✓ Results saved to {TSV_FILENAME}"),
        Err(e) => eprintln!("\n✗ Failed to save results to {TSV_FILENAME}: {e}"),
    }
}
