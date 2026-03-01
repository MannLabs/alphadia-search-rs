//! Unified benchmark for score and convolution operations.
//!
//! ```bash
//! cargo run --release --bin benchmark
//! cargo run --release --bin benchmark -- -n 5
//! cargo run --release --bin benchmark -- --tag conv
//! cargo run --release --bin benchmark -- -n 5 --tag log_dot
//! ```

mod convolution;
mod hyperscore;
mod runner;
mod score;

use runner::BenchmarkResult;
use std::fs::File;
use std::io::Write;

fn fmt_err(val: f32) -> String {
    if val == 0.0 {
        "0.00".to_string()
    } else {
        format!("{:.4}", val * 100.0)
    }
}

fn print_results(results: &[BenchmarkResult]) {
    println!(
        "{:<25} {:<22} {:<22} {:>10} {:>8} {:>12} {:>12} {:>8}",
        "Tag", "Test Case", "Impl", "Time (s)", "Speedup x", "Avg Err (%)", "Max Err (%)", "Status"
    );
    println!("{}", "-".repeat(119));

    for r in results {
        println!(
            "{:<25} {:<22} {:<22} {:>10.6} {:>8.2} {:>12} {:>12} {:>8}",
            r.tag,
            r.test_case,
            r.implementation,
            r.time_seconds,
            r.speedup,
            fmt_err(r.avg_rel_error),
            fmt_err(r.max_rel_error),
            r.status,
        );
    }
}

fn save_tsv(results: &[BenchmarkResult], filename: &str) -> std::io::Result<()> {
    let mut file = File::create(filename)?;
    writeln!(
        file,
        "Test Case\tImpl\tTime (s)\tSpeedup\tAvg Err (%)\tMax Err (%)\tStatus"
    )?;
    for r in results {
        writeln!(
            file,
            "{}\t{}\t{:.6}\t{:.2}\t{}\t{}\t{}",
            r.test_case,
            r.implementation,
            r.time_seconds,
            r.speedup,
            fmt_err(r.avg_rel_error),
            fmt_err(r.max_rel_error),
            r.status,
        )?;
    }
    Ok(())
}

fn parse_args() -> (usize, Option<String>) {
    let args: Vec<String> = std::env::args().collect();
    let mut n_rounds = 10;
    let mut tag_filter: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-n" => {
                i += 1;
                n_rounds = args[i].parse().expect("invalid value for -n");
            }
            "--tag" => {
                i += 1;
                tag_filter = Some(args[i].clone());
            }
            other => {
                eprintln!("Unknown argument: {other}");
                eprintln!("Usage: benchmark [-n ROUNDS] [--tag TAG]");
                std::process::exit(1);
            }
        }
        i += 1;
    }
    (n_rounds, tag_filter)
}

fn main() {
    let (n_rounds, tag_filter) = parse_args();

    println!(
        "Benchmark — arch: {}, rounds: {n_rounds}",
        std::env::consts::ARCH
    );
    if let Some(ref tag) = tag_filter {
        println!("Filtering by tag: {tag}");
    }
    println!();

    let mut all_cases = score::test_cases();
    all_cases.extend(convolution::test_cases());
    all_cases.extend(hyperscore::test_cases());

    let cases: Vec<_> = match &tag_filter {
        Some(tag) => all_cases
            .into_iter()
            .filter(|tc| tc.tag() == tag.as_str())
            .collect(),
        None => all_cases,
    };

    if cases.is_empty() {
        eprintln!(
            "No test cases matched tag filter '{}'",
            tag_filter.unwrap_or_default()
        );
        eprintln!("Available tags: log_dot, sqrt_dot, conv, hyperscore");
        std::process::exit(1);
    }

    let mut all_results = Vec::new();
    for tc in &cases {
        println!("Running {:<30}", tc.name());
        let results = tc.run(n_rounds);
        all_results.extend(results);
    }

    println!();
    print_results(&all_results);

    let tsv_file = "benchmark.tsv";
    match save_tsv(&all_results, tsv_file) {
        Ok(()) => println!("\nResults saved to {tsv_file}"),
        Err(e) => eprintln!("\nFailed to save {tsv_file}: {e}"),
    }
}
