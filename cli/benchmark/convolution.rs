use numpy::ndarray::Array2;
use rand::prelude::*;

use alphadia_search_rs::benchmark::{
    benchmark_nonpadded_convolution_simd, benchmark_padded_convolution,
    benchmark_padded_convolution_branching, benchmark_padded_convolution_branching_simd,
    benchmark_symmetric_kernel_simd,
};
use alphadia_search_rs::convolution::convolution;

#[cfg(target_arch = "aarch64")]
use alphadia_search_rs::convolution::neon::convolution_neon_v2;
use alphadia_search_rs::GaussianKernel;

use super::runner::{BenchmarkCase, ImplList, TypedBenchmarkCase};

const TAG: &str = "conv";

struct ConvData {
    kernel: GaussianKernel,
    arrays: Vec<Array2<f32>>,
}

fn gen_conv_data(rows: usize, cols: usize, kernel_width: usize) -> ConvData {
    let mut rng = rand::rng();
    let kernel = GaussianKernel::new(2.0, 1.0, kernel_width, 1.0);
    let mut arrays = Vec::with_capacity(100);
    for _ in 0..100 {
        let mut arr = Array2::<f32>::zeros((rows, cols));
        for i in 0..rows {
            for j in 0..cols {
                arr[[i, j]] = rng.random_range(0.0..1.0);
            }
        }
        arrays.push(arr);
    }
    ConvData { kernel, arrays }
}

fn conv_error(reference: &Array2<f32>, candidate: &Array2<f32>) -> (f32, f32) {
    let mut sum_abs = 0.0f32;
    let mut max_abs = 0.0f32;
    let (rows, cols) = reference.dim();
    let n = (rows * cols) as f32;
    for i in 0..rows {
        for j in 0..cols {
            let diff = (reference[[i, j]] - candidate[[i, j]]).abs();
            sum_abs += diff;
            max_abs = max_abs.max(diff);
        }
    }
    (sum_abs / n, max_abs)
}

fn run_padded(data: &ConvData) -> Array2<f32> {
    let mut out = Array2::zeros((0, 0));
    for arr in &data.arrays {
        out = benchmark_padded_convolution(&data.kernel, arr);
    }
    out
}

fn run_branching(data: &ConvData) -> Array2<f32> {
    let mut out = Array2::zeros((0, 0));
    for arr in &data.arrays {
        out = benchmark_padded_convolution_branching(&data.kernel, arr);
    }
    out
}

fn run_branching_simd(data: &ConvData) -> Array2<f32> {
    let mut out = Array2::zeros((0, 0));
    for arr in &data.arrays {
        out = benchmark_padded_convolution_branching_simd(&data.kernel, arr);
    }
    out
}

fn run_nonpadded_simd(data: &ConvData) -> Array2<f32> {
    let mut out = Array2::zeros((0, 0));
    for arr in &data.arrays {
        out = benchmark_nonpadded_convolution_simd(&data.kernel, arr);
    }
    out
}

fn run_symmetric_simd(data: &ConvData) -> Array2<f32> {
    let mut out = Array2::zeros((0, 0));
    for arr in &data.arrays {
        out = benchmark_symmetric_kernel_simd(&data.kernel, arr);
    }
    out
}

fn run_production(data: &ConvData) -> Array2<f32> {
    let mut out = Array2::zeros((0, 0));
    for arr in &data.arrays {
        out = convolution(&data.kernel, arr);
    }
    out
}

#[cfg(target_arch = "aarch64")]
fn run_neon_v2(data: &ConvData) -> Array2<f32> {
    let mut out = Array2::zeros((0, 0));
    for arr in &data.arrays {
        out = convolution_neon_v2(&data.kernel, arr);
    }
    out
}

type ConvImpls = ImplList<ConvData, Array2<f32>>;

fn conv_impls() -> ConvImpls {
    let mut v: ConvImpls = vec![
        ("Scalar", run_padded as fn(&ConvData) -> Array2<f32>),
        ("Branching", run_branching),
        ("Branching+SIMD", run_branching_simd),
        ("Nonpadded+SIMD", run_nonpadded_simd),
        ("Symmetric+SIMD", run_symmetric_simd),
        ("Nonpadded+Symmetric", run_production),
    ];
    #[cfg(target_arch = "aarch64")]
    v.push(("NEON-v2", run_neon_v2));
    v
}

fn make_conv(
    name: &'static str,
    rows: usize,
    cols: usize,
    kernel_width: usize,
) -> TypedBenchmarkCase<ConvData, Array2<f32>> {
    TypedBenchmarkCase {
        name,
        tag: TAG,
        generate: Box::new(move || gen_conv_data(rows, cols, kernel_width)),
        implementations: conv_impls(),
        compute_error: conv_error,
        max_error_tolerance: 1.0,
    }
}

pub fn test_cases() -> Vec<Box<dyn BenchmarkCase>> {
    vec![
        Box::new(make_conv("conv_12x1000", 12, 1000, 20)),
        Box::new(make_conv("conv_48x1000", 48, 1000, 20)),
        Box::new(make_conv("conv_12x10000", 12, 10000, 20)),
    ]
}
