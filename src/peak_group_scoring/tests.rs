#[allow(unused_imports)]
use super::utils::{correlation, correlation_axis_0, median_axis_0, normalize_profiles};
#[allow(unused_imports)]
use numpy::ndarray::arr2;

#[test]
fn test_median_axis_0_basic() {
    let array = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);

    let result = median_axis_0(&array);
    assert_eq!(result, vec![4.0, 5.0, 6.0]);
}

#[test]
fn test_median_axis_0_even_rows() {
    let array = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]);

    let result = median_axis_0(&array);
    assert_eq!(result, vec![4.0, 5.0]); // (3+5)/2, (4+6)/2
}

#[test]
fn test_median_axis_0_odd_rows() {
    let array = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);

    let result = median_axis_0(&array);
    assert_eq!(result, vec![3.0, 4.0]); // middle value
}

#[test]
fn test_normalize_profiles_basic() {
    let array = arr2(&[
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [2.0, 4.0, 6.0, 8.0, 10.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]);

    let result = normalize_profiles(&array, 1);

    // First row: center is 3.0, window [2,3,4] = 3.0, so normalized by 3.0
    assert_eq!(result[[0, 0]], 1.0 / 3.0);
    assert_eq!(result[[0, 1]], 2.0 / 3.0);
    assert_eq!(result[[0, 2]], 3.0 / 3.0);
    assert_eq!(result[[0, 3]], 4.0 / 3.0);
    assert_eq!(result[[0, 4]], 5.0 / 3.0);

    // Second row: center is 6.0, window [4,6,8] = 6.0, so normalized by 6.0
    assert_eq!(result[[1, 0]], 2.0 / 6.0);
    assert_eq!(result[[1, 1]], 4.0 / 6.0);
    assert_eq!(result[[1, 2]], 6.0 / 6.0);
    assert_eq!(result[[1, 3]], 8.0 / 6.0);
    assert_eq!(result[[1, 4]], 10.0 / 6.0);

    // Third row: center intensity is 0, so should remain zeros
    for j in 0..5 {
        assert_eq!(result[[2, j]], 0.0);
    }
}

#[test]
fn test_normalize_profiles_edge_cases() {
    let array = arr2(&[
        [1.0, 2.0], // Only 2 columns, center is 1
        [3.0, 4.0],
    ]);

    let result = normalize_profiles(&array, 1);

    // With center_dilations=1, window should be [1,2] for first row
    // Mean is 1.5, so normalized by 1.5
    assert_eq!(result[[0, 0]], 1.0 / 1.5);
    assert_eq!(result[[0, 1]], 2.0 / 1.5);

    // Second row: window [3,4], mean is 3.5
    assert_eq!(result[[1, 0]], 3.0 / 3.5);
    assert_eq!(result[[1, 1]], 4.0 / 3.5);
}

#[test]
fn test_normalize_profiles_zero_center() {
    let array = arr2(&[
        [1.0, 0.0, 3.0], // Center is 0
        [1.0, 2.0, 3.0], // Center is 2
    ]);

    let result = normalize_profiles(&array, 0); // center_dilations=0, only center point

    // First row: center is 0, so should remain unchanged (all zeros)
    for j in 0..3 {
        assert_eq!(result[[0, j]], 0.0);
    }

    // Second row: center is 2, so normalized by 2
    assert_eq!(result[[1, 0]], 1.0 / 2.0);
    assert_eq!(result[[1, 1]], 2.0 / 2.0);
    assert_eq!(result[[1, 2]], 3.0 / 2.0);
}

#[test]
fn test_correlation_axis_0_basic() {
    let median_profile = vec![1.0, 2.0, 3.0];
    let dense_xic = arr2(&[
        [1.0, 2.0, 3.0], // Perfect correlation
        [2.0, 4.0, 6.0], // Perfect correlation (scaled)
        [3.0, 2.0, 1.0], // Perfect negative correlation
        [0.0, 0.0, 0.0], // All zeros
    ]);

    let result = correlation_axis_0(&median_profile, &dense_xic);

    // First row: perfect positive correlation
    assert!((result[0] - 1.0).abs() < 1e-6);

    // Second row: perfect positive correlation (scaled)
    assert!((result[1] - 1.0).abs() < 1e-6);

    // Third row: perfect negative correlation
    assert!((result[2] - (-1.0)).abs() < 1e-6);

    // Fourth row: all zeros, should return 0
    assert_eq!(result[3], 0.0);
}

#[test]
fn test_correlation_axis_0_edge_cases() {
    let median_profile = vec![1.0, 2.0];
    let dense_xic = arr2(&[
        [1.0, 2.0], // Perfect correlation
        [1.0, 1.0], // Constant values
        [0.0, 0.0], // All zeros
    ]);

    let result = correlation_axis_0(&median_profile, &dense_xic);

    // First row: perfect correlation
    assert!((result[0] - 1.0).abs() < 1e-6);

    // Second row: constant values, should return 0
    assert_eq!(result[1], 0.0);

    // Third row: all zeros, should return 0
    assert_eq!(result[2], 0.0);
}

#[test]
fn test_correlation_axis_0_mismatched_lengths() {
    let median_profile = vec![1.0, 2.0, 3.0];
    let dense_xic = arr2(&[
        [1.0, 2.0], // Different length
    ]);

    let result = correlation_axis_0(&median_profile, &dense_xic);

    // Should return 0 for mismatched lengths
    assert_eq!(result[0], 0.0);
}

#[test]
fn test_correlation_standalone() {
    // Test perfect positive correlation
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![2.0, 4.0, 6.0];
    assert!((correlation(&x, &y) - 1.0).abs() < 1e-6);

    // Test perfect negative correlation
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![3.0, 2.0, 1.0];
    assert!((correlation(&x, &y) - (-1.0)).abs() < 1e-6);

    // Test zero correlation
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![1.0, 1.0, 1.0];
    assert_eq!(correlation(&x, &y), 0.0);

    // Test all zeros
    let x = vec![0.0, 0.0, 0.0];
    let y = vec![1.0, 2.0, 3.0];
    assert_eq!(correlation(&x, &y), 0.0);

    // Test mismatched lengths
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![1.0, 2.0];
    assert_eq!(correlation(&x, &y), 0.0);
}
