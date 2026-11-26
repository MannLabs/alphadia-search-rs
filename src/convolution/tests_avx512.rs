#![cfg(target_arch = "x86_64")]

use super::avx512::convolution_avx512;
use super::scalar::convolution_scalar;
use crate::kernel::GaussianKernel;
use approx::assert_relative_eq;
use numpy::ndarray::Array2;

fn have_avx512() -> bool {
    if !std::is_x86_feature_detected!("avx512f") {
        println!("AVX-512F not detected on this host; skipping");
        return false;
    }
    true
}

#[test]
fn test_against_scalar_reference() {
    if !have_avx512() {
        return;
    }
    let kernel = GaussianKernel::new(0.5, 1.0, 5, 1.0);
    let xic_data: Vec<f32> = (0..64).map(|i| ((i * 7) % 11) as f32).collect();
    let xic = Array2::from_shape_vec((1, 64), xic_data).unwrap();

    let result = unsafe { convolution_avx512(&kernel, &xic) };
    let reference = convolution_scalar(&kernel, &xic);

    let half_kernel = kernel.kernel_array.len() / 2;
    for i in half_kernel..(64 - half_kernel) {
        assert_relative_eq!(result[[0, i]], reference[[0, i]], epsilon = 1e-5);
    }
}

#[test]
fn test_multiple_fragments() {
    if !have_avx512() {
        return;
    }
    let kernel = GaussianKernel::new(1.0, 1.0, 7, 1.0);
    let mut xic = Array2::<f32>::zeros((4, 80));
    for f in 0..4 {
        for j in 0..80 {
            xic[[f, j]] = ((f * 13 + j * 5) % 17) as f32;
        }
    }
    let result = unsafe { convolution_avx512(&kernel, &xic) };
    let reference = convolution_scalar(&kernel, &xic);
    let hk = kernel.kernel_array.len() / 2;
    for f in 0..4 {
        for i in hk..(80 - hk) {
            assert_relative_eq!(result[[f, i]], reference[[f, i]], epsilon = 1e-5);
        }
    }
}

#[test]
fn test_edge_cases() {
    if !have_avx512() {
        return;
    }
    let kernel = GaussianKernel::new(1.0, 1.0, 11, 1.0);
    let xic_small = Array2::<f32>::ones((1, 5));
    let result = unsafe { convolution_avx512(&kernel, &xic_small) };
    assert_eq!(result.dim(), xic_small.dim());

    let xic_empty = Array2::<f32>::zeros((0, 10));
    let result = unsafe { convolution_avx512(&kernel, &xic_empty) };
    assert_eq!(result.dim(), xic_empty.dim());
}

#[test]
fn test_tail_handling() {
    if !have_avx512() {
        return;
    }
    // Interior width not a multiple of 16 forces the scalar tail.
    let kernel = GaussianKernel::new(1.0, 1.0, 5, 1.0);
    let xic_data: Vec<f32> = (0..47).map(|i| (i as f32) * 0.3).collect();
    let xic = Array2::from_shape_vec((1, 47), xic_data).unwrap();
    let result = unsafe { convolution_avx512(&kernel, &xic) };
    let reference = convolution_scalar(&kernel, &xic);
    let hk = kernel.kernel_array.len() / 2;
    for i in hk..(47 - hk) {
        assert_relative_eq!(result[[0, i]], reference[[0, i]], epsilon = 1e-5);
    }
}
