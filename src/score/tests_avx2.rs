#![cfg(target_arch = "x86_64")]

use super::avx2::{axis_log_dot_product_avx2, axis_sqrt_dot_product_avx2};
use super::scalar::{axis_log_dot_product_scalar, axis_sqrt_dot_product_scalar};
use numpy::ndarray::Array2;

fn have_avx2() -> bool {
    if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
        println!("AVX2/FMA not detected on this host; skipping");
        return false;
    }
    true
}

#[test]
fn test_log_simd_vs_scalar() {
    if !have_avx2() {
        return;
    }
    let n_rows = 5;
    let n_cols = 20;
    let mut data = Array2::zeros((n_rows, n_cols));
    let mut weights = Vec::with_capacity(n_rows);
    for i in 0..n_rows {
        weights.push((i as f32) * 0.5 + 0.5);
        for j in 0..n_cols {
            data[[i, j]] = (i as f32) * 0.5 + (j as f32) * 0.25;
        }
    }
    let scalar_result = axis_log_dot_product_scalar(&data, &weights);
    let simd_result = unsafe { axis_log_dot_product_avx2(&data, &weights) };
    for (s, v) in scalar_result.iter().zip(simd_result.iter()) {
        assert!((s - v).abs() / s.abs() < 0.15);
    }
}

#[test]
fn test_log_unaligned() {
    if !have_avx2() {
        return;
    }
    use numpy::ndarray::arr2;
    let array = arr2(&[
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.6, 0.7, 0.8, 0.9, 1.0],
        [1.1, 1.2, 1.3, 1.4, 1.5],
    ]);
    let weights = vec![0.5, 1.0, 1.5];
    let scalar_result = axis_log_dot_product_scalar(&array, &weights);
    let simd_result = unsafe { axis_log_dot_product_avx2(&array, &weights) };
    for (s, v) in scalar_result.iter().zip(simd_result.iter()) {
        assert!((s - v).abs() / s.abs() < 0.15);
    }
}

#[test]
fn test_sqrt_simd_vs_scalar() {
    if !have_avx2() {
        return;
    }
    let n_rows = 5;
    let n_cols = 20;
    let mut data = Array2::zeros((n_rows, n_cols));
    let mut weights = Vec::with_capacity(n_rows);
    for i in 0..n_rows {
        weights.push((i as f32) * 0.5 + 0.5);
        for j in 0..n_cols {
            data[[i, j]] = ((i as f32) * 0.5 + (j as f32) * 0.25).powi(2);
        }
    }
    let scalar_result = axis_sqrt_dot_product_scalar(&data, &weights);
    let simd_result = unsafe { axis_sqrt_dot_product_avx2(&data, &weights) };
    for (s, v) in scalar_result.iter().zip(simd_result.iter()) {
        assert!((s - v).abs() / s.abs() < 0.01);
    }
}

#[test]
fn test_sqrt_unaligned() {
    if !have_avx2() {
        return;
    }
    use numpy::ndarray::arr2;
    let array = arr2(&[
        [0.0, 1.0, 4.0, 9.0, 16.0],
        [25.0, 36.0, 49.0, 64.0, 81.0],
        [100.0, 121.0, 144.0, 169.0, 196.0],
    ]);
    let weights = vec![0.5, 1.0, 1.5];
    let scalar_result = axis_sqrt_dot_product_scalar(&array, &weights);
    let simd_result = unsafe { axis_sqrt_dot_product_avx2(&array, &weights) };
    for (s, v) in scalar_result.iter().zip(simd_result.iter()) {
        assert!((s - v).abs() / s.abs() < 0.01);
    }
}
