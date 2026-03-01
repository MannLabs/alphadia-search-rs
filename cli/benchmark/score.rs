use numpy::ndarray::{Array1, Array2};
use rand::prelude::*;

use alphadia_search_rs::score::scalar::{
    axis_log_dot_product_scalar, axis_sqrt_dot_product_scalar,
};

#[cfg(target_arch = "aarch64")]
use alphadia_search_rs::score::neon::{
    axis_log_dot_product_neon, axis_log_dot_product_neon_v2, axis_sqrt_dot_product_neon,
};

use super::runner::{BenchmarkCase, ImplList, TypedBenchmarkCase};

const TAG_LOG_DOT: &str = "log_dot";
const TAG_SQRT_DOT: &str = "sqrt_dot";

struct ScoreData {
    array: Array2<f32>,
    weights: Vec<f32>,
}

fn gen_score_data(rows: usize, cols: usize) -> ScoreData {
    let mut rng = rand::rng();
    let mut array = Array2::<f32>::zeros((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            array[[i, j]] = rng.random_range(0.1..10.0);
        }
    }
    let weights: Vec<f32> = (0..rows).map(|_| rng.random_range(0.1..2.0)).collect();
    ScoreData { array, weights }
}

fn score_error(reference: &Array1<f32>, candidate: &Array1<f32>) -> (f32, f32) {
    let mut sum_rel = 0.0f32;
    let mut max_rel = 0.0f32;
    for i in 0..reference.len() {
        let rel = (reference[i] - candidate[i]).abs() / reference[i].abs().max(1e-12);
        sum_rel += rel;
        max_rel = max_rel.max(rel);
    }
    let avg_rel = sum_rel / reference.len() as f32;
    (avg_rel, max_rel)
}

fn log_dot_scalar(data: &ScoreData) -> Array1<f32> {
    axis_log_dot_product_scalar(&data.array, &data.weights)
}

#[cfg(target_arch = "aarch64")]
fn log_dot_neon(data: &ScoreData) -> Array1<f32> {
    axis_log_dot_product_neon(&data.array, &data.weights)
}

#[cfg(target_arch = "aarch64")]
fn log_dot_neon_v2(data: &ScoreData) -> Array1<f32> {
    axis_log_dot_product_neon_v2(&data.array, &data.weights)
}

fn sqrt_dot_scalar(data: &ScoreData) -> Array1<f32> {
    axis_sqrt_dot_product_scalar(&data.array, &data.weights)
}

#[cfg(target_arch = "aarch64")]
fn sqrt_dot_neon(data: &ScoreData) -> Array1<f32> {
    axis_sqrt_dot_product_neon(&data.array, &data.weights)
}

type ScoreImpls = ImplList<ScoreData, Array1<f32>>;

fn log_dot_impls() -> ScoreImpls {
    let mut v: ScoreImpls = vec![("Scalar", log_dot_scalar)];
    #[cfg(target_arch = "aarch64")]
    v.push(("NEON", log_dot_neon));
    #[cfg(target_arch = "aarch64")]
    v.push(("NEON-v2", log_dot_neon_v2));
    v
}

fn sqrt_dot_impls() -> ScoreImpls {
    let mut v: ScoreImpls = vec![("Scalar", sqrt_dot_scalar)];
    #[cfg(target_arch = "aarch64")]
    v.push(("NEON", sqrt_dot_neon));
    v
}

fn make_score(
    name: &'static str,
    tag: &'static str,
    rows: usize,
    cols: usize,
    implementations: ScoreImpls,
) -> TypedBenchmarkCase<ScoreData, Array1<f32>> {
    TypedBenchmarkCase {
        name,
        tag,
        generate: Box::new(move || gen_score_data(rows, cols)),
        implementations,
        compute_error: score_error,
        max_error_tolerance: 0.20,
    }
}

pub fn test_cases() -> Vec<Box<dyn BenchmarkCase>> {
    vec![
        Box::new(make_score(
            "log_dot_12x100",
            TAG_LOG_DOT,
            12,
            100,
            log_dot_impls(),
        )),
        Box::new(make_score(
            "log_dot_12x1000",
            TAG_LOG_DOT,
            12,
            1000,
            log_dot_impls(),
        )),
        Box::new(make_score(
            "log_dot_48x1000",
            TAG_LOG_DOT,
            48,
            1000,
            log_dot_impls(),
        )),
        Box::new(make_score(
            "log_dot_12x10000",
            TAG_LOG_DOT,
            12,
            10000,
            log_dot_impls(),
        )),
        Box::new(make_score(
            "log_dot_48x10000",
            TAG_LOG_DOT,
            48,
            10000,
            log_dot_impls(),
        )),
        Box::new(make_score(
            "log_dot_12x100000",
            TAG_LOG_DOT,
            12,
            100000,
            log_dot_impls(),
        )),
        Box::new(make_score(
            "log_dot_48x100000",
            TAG_LOG_DOT,
            48,
            100000,
            log_dot_impls(),
        )),
        Box::new(make_score(
            "sqrt_dot_12x1000",
            TAG_SQRT_DOT,
            12,
            1000,
            sqrt_dot_impls(),
        )),
        Box::new(make_score(
            "sqrt_dot_48x1000",
            TAG_SQRT_DOT,
            48,
            1000,
            sqrt_dot_impls(),
        )),
        Box::new(make_score(
            "sqrt_dot_12x10000",
            TAG_SQRT_DOT,
            12,
            10000,
            sqrt_dot_impls(),
        )),
    ]
}
