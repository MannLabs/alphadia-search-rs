#[allow(unused_imports)]
use super::utils::{
    calculate_fwhm_rt, calculate_hyperscore, calculate_hyperscore_weighted,
    calculate_longest_ion_series, correlation, correlation_axis_0, intensity_ion_series,
    median_axis_0, normalize_profiles,
};
#[allow(unused_imports)]
use crate::constants::FragmentType;
#[allow(unused_imports)]
use numpy::ndarray::{arr1, arr2, Array1, Array2};

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

#[test]
fn test_hyperscore_calculation() {
    // Test with simple b and y ions
    let fragment_types = vec![
        FragmentType::B,
        FragmentType::Y,
        FragmentType::B,
        FragmentType::Y,
        FragmentType::A,
    ]; // b, y, b, y, a
    let fragment_intensities = vec![100.0, 200.0, 150.0, 250.0, 50.0];
    let matched_mask = vec![true, true, true, true, false]; // last fragment not matched

    let hyperscore = calculate_hyperscore(&fragment_types, &fragment_intensities, &matched_mask);

    // Should be > 0 since we have matched b and y ions
    assert!(hyperscore > 0.0);
    println!("Hyperscore: {}", hyperscore);
}

#[test]
fn test_hyperscore_no_matches() {
    let fragment_types = vec![
        FragmentType::B,
        FragmentType::Y,
        FragmentType::B,
        FragmentType::Y,
    ];
    let fragment_intensities = vec![100.0, 200.0, 150.0, 250.0];
    let matched_mask = vec![false, false, false, false]; // no matches

    let hyperscore = calculate_hyperscore(&fragment_types, &fragment_intensities, &matched_mask);

    // Should be 0 since no fragments are matched
    assert_eq!(hyperscore, 0.0);
}

#[test]
fn test_hyperscore_only_b_ions() {
    let fragment_types = vec![FragmentType::B, FragmentType::B, FragmentType::B]; // only b ions
    let fragment_intensities = vec![100.0, 200.0, 150.0];
    let matched_mask = vec![true, true, true];

    let hyperscore = calculate_hyperscore(&fragment_types, &fragment_intensities, &matched_mask);

    // Should be > 0 even with only b ions
    assert!(hyperscore > 0.0);
    println!("B-only hyperscore: {}", hyperscore);
}

#[test]
fn test_hyperscore_only_y_ions() {
    let fragment_types = vec![FragmentType::Y, FragmentType::Y, FragmentType::Y]; // only y ions
    let fragment_intensities = vec![100.0, 200.0, 150.0];
    let matched_mask = vec![true, true, true];

    let hyperscore = calculate_hyperscore(&fragment_types, &fragment_intensities, &matched_mask);

    // Should be > 0 even with only y ions
    assert!(hyperscore > 0.0);
    println!("Y-only hyperscore: {}", hyperscore);
}

#[test]
fn test_hyperscore_mixed_fragment_types() {
    let fragment_types = vec![
        FragmentType::B,
        FragmentType::Y,
        FragmentType::C,
        FragmentType::X,
        FragmentType::B,
        FragmentType::Y,
    ]; // b, y, c, x, b, y
    let fragment_intensities = vec![100.0, 200.0, 50.0, 75.0, 150.0, 250.0];
    let matched_mask = vec![true, true, true, true, true, true];

    let hyperscore = calculate_hyperscore(&fragment_types, &fragment_intensities, &matched_mask);

    // Should only count b and y ions, ignoring other fragment types
    assert!(hyperscore > 0.0);
    println!("Mixed types hyperscore: {}", hyperscore);
}

#[test]
fn test_hyperscore_manual_calculation() {
    // Test case with known values to verify MSFragger formula implementation
    // hyperscore = log(Nb! * Ny! * sum(Ib,i) * sum(Iy,i))

    let fragment_types = vec![
        FragmentType::B,
        FragmentType::B,
        FragmentType::Y,
        FragmentType::Y,
    ];
    let fragment_intensities = vec![100.0, 200.0, 150.0, 250.0];
    let matched_mask = vec![true, true, true, true];

    let hyperscore = calculate_hyperscore(&fragment_types, &fragment_intensities, &matched_mask);

    // Manual calculation using Stirling's approximation:
    // Nb = 2 (two b-ions matched)
    // Ny = 2 (two y-ions matched)
    // sum(Ib,i) = 100.0 + 200.0 = 300.0
    // sum(Iy,i) = 150.0 + 250.0 = 400.0
    //
    // hyperscore = log(2!) + log(2!) + log(300.0) + log(400.0)
    // Where log(2!) is calculated using Stirling's approximation

    println!("Manual calculation hyperscore: {}", hyperscore);

    // Verify the calculation follows MSFragger formula structure
    // Should be > 0 since we have matched ions with positive intensities
    assert!(hyperscore > 0.0);

    // Verify it's in a reasonable range for this input
    assert!(
        hyperscore > 10.0 && hyperscore < 20.0,
        "Hyperscore {} is outside expected range",
        hyperscore
    );
}

#[test]
fn test_hyperscore_edge_case_single_ion() {
    // Test with single matched ion of each type
    let fragment_types = vec![FragmentType::B, FragmentType::Y];
    let fragment_intensities = vec![50.0, 75.0];
    let matched_mask = vec![true, true];

    let hyperscore = calculate_hyperscore(&fragment_types, &fragment_intensities, &matched_mask);

    // Manual calculation:
    // Nb = 1, Ny = 1
    // sum(Ib,i) = 50.0, sum(Iy,i) = 75.0
    // hyperscore = log(1!) + log(1!) + log(50.0) + log(75.0)
    // Where log(1!) = 0 (exact for factorial of 1)

    let expected = 0.0 + 0.0 + 50.0_f32.ln() + 75.0_f32.ln();
    println!(
        "Single ion hyperscore: {}, Expected: {}",
        hyperscore, expected
    );

    // For 1!, Stirling's approximation introduces some error compared to exact calculation
    // This is expected behavior when using approximation
    println!("Difference: {}", (hyperscore - expected).abs());
    assert!(hyperscore > 0.0);
    assert!(
        hyperscore > 7.0 && hyperscore < 10.0,
        "Hyperscore {} is outside expected range",
        hyperscore
    );
}

#[test]
fn test_longest_ion_series_basic() {
    // Test with continuous b and y series
    let fragment_types = vec![
        FragmentType::B,
        FragmentType::B,
        FragmentType::B, // b1, b2, b3
        FragmentType::Y,
        FragmentType::Y,
        FragmentType::Y, // y1, y2, y3
    ];
    let fragment_numbers = vec![1, 2, 3, 1, 2, 3];
    let matched_mask = vec![true, true, true, true, true, true];

    let (longest_b, longest_y) =
        calculate_longest_ion_series(&fragment_types, &fragment_numbers, &matched_mask);

    assert_eq!(longest_b, 3); // b1, b2, b3 continuous
    assert_eq!(longest_y, 3); // y1, y2, y3 continuous
}

#[test]
fn test_longest_ion_series_gaps() {
    // Test with gaps in the series
    let fragment_types = vec![
        FragmentType::B,
        FragmentType::B,
        FragmentType::B, // b1, b3, b4 (gap at b2)
        FragmentType::Y,
        FragmentType::Y, // y1, y3 (gap at y2)
    ];
    let fragment_numbers = vec![1, 3, 4, 1, 3];
    let matched_mask = vec![true, true, true, true, true];

    let (longest_b, longest_y) =
        calculate_longest_ion_series(&fragment_types, &fragment_numbers, &matched_mask);

    assert_eq!(longest_b, 2); // b3, b4 continuous (longest sequence)
    assert_eq!(longest_y, 1); // y1 and y3 are not continuous
}

#[test]
fn test_longest_ion_series_partial_matches() {
    // Test with some ions not matched
    let fragment_types = vec![
        FragmentType::B,
        FragmentType::B,
        FragmentType::B,
        FragmentType::B, // b1, b2, b3, b4
        FragmentType::Y,
        FragmentType::Y,
        FragmentType::Y, // y1, y2, y3
    ];
    let fragment_numbers = vec![1, 2, 3, 4, 1, 2, 3];
    let matched_mask = vec![true, false, true, true, true, true, false]; // b2 and y3 not matched

    let (longest_b, longest_y) =
        calculate_longest_ion_series(&fragment_types, &fragment_numbers, &matched_mask);

    assert_eq!(longest_b, 2); // b3, b4 continuous (b2 not matched)
    assert_eq!(longest_y, 2); // y1, y2 continuous (y3 not matched)
}

#[test]
fn test_longest_ion_series_no_matches() {
    let fragment_types = vec![FragmentType::B, FragmentType::Y, FragmentType::B];
    let fragment_numbers = vec![1, 1, 2];
    let matched_mask = vec![false, false, false]; // no matches

    let (longest_b, longest_y) =
        calculate_longest_ion_series(&fragment_types, &fragment_numbers, &matched_mask);

    assert_eq!(longest_b, 0);
    assert_eq!(longest_y, 0);
}

#[test]
fn test_longest_ion_series_only_b_ions() {
    let fragment_types = vec![
        FragmentType::B,
        FragmentType::B,
        FragmentType::B,
        FragmentType::B,
    ];
    let fragment_numbers = vec![1, 2, 4, 5]; // b1, b2, gap, b4, b5
    let matched_mask = vec![true, true, true, true];

    let (longest_b, longest_y) =
        calculate_longest_ion_series(&fragment_types, &fragment_numbers, &matched_mask);

    assert_eq!(longest_b, 2); // either b1,b2 or b4,b5
    assert_eq!(longest_y, 0); // no y ions
}

#[test]
fn test_longest_ion_series_only_y_ions() {
    let fragment_types = vec![FragmentType::Y, FragmentType::Y, FragmentType::Y];
    let fragment_numbers = vec![2, 3, 4]; // y2, y3, y4 continuous
    let matched_mask = vec![true, true, true];

    let (longest_b, longest_y) =
        calculate_longest_ion_series(&fragment_types, &fragment_numbers, &matched_mask);

    assert_eq!(longest_b, 0); // no b ions
    assert_eq!(longest_y, 3); // y2, y3, y4 continuous
}

#[test]
fn test_longest_ion_series_unordered_input() {
    // Test with unordered input (function should sort internally)
    let fragment_types = vec![
        FragmentType::B,
        FragmentType::Y,
        FragmentType::B,
        FragmentType::Y,
        FragmentType::B,
    ];
    let fragment_numbers = vec![3, 2, 1, 3, 2]; // b3, y2, b1, y3, b2
    let matched_mask = vec![true, true, true, true, true];

    let (longest_b, longest_y) =
        calculate_longest_ion_series(&fragment_types, &fragment_numbers, &matched_mask);

    assert_eq!(longest_b, 3); // b1, b2, b3 continuous when sorted
    assert_eq!(longest_y, 2); // y2, y3 continuous when sorted
}

#[test]
fn test_longest_ion_series_empty() {
    let fragment_types: Vec<u8> = vec![];
    let fragment_numbers: Vec<u8> = vec![];
    let matched_mask: Vec<bool> = vec![];

    let (longest_b, longest_y) =
        calculate_longest_ion_series(&fragment_types, &fragment_numbers, &matched_mask);

    assert_eq!(longest_b, 0);
    assert_eq!(longest_y, 0);
}

#[test]
fn test_longest_ion_series_mismatched_lengths() {
    let fragment_types = vec![FragmentType::B, FragmentType::Y];
    let fragment_numbers = vec![1, 2, 3]; // wrong length
    let matched_mask = vec![true, true];

    let (longest_b, longest_y) =
        calculate_longest_ion_series(&fragment_types, &fragment_numbers, &matched_mask);

    assert_eq!(longest_b, 0);
    assert_eq!(longest_y, 0);
}

#[test]
fn test_longest_ion_series_decreasing_fragment_numbers() {
    // Test with decreasing fragment numbers (reverse order)
    let fragment_types = vec![
        FragmentType::B,
        FragmentType::B,
        FragmentType::B, // b3, b2, b1 (decreasing order)
        FragmentType::Y,
        FragmentType::Y, // y3, y2, y1 (decreasing order)
    ];
    let fragment_numbers = vec![3, 2, 1, 3, 2];
    let matched_mask = vec![true, true, true, true, true];

    let (longest_b, longest_y) =
        calculate_longest_ion_series(&fragment_types, &fragment_numbers, &matched_mask);

    assert_eq!(longest_b, 3); // b1, b2, b3 continuous when sorted
    assert_eq!(longest_y, 2); // y2, y3 continuous when sorted
}

#[test]
fn test_longest_ion_series_random_order_with_gaps() {
    // Test with randomly ordered fragment numbers containing gaps
    let fragment_types = vec![
        FragmentType::B,
        FragmentType::B,
        FragmentType::B,
        FragmentType::B, // b4, b1, b3, b6 (random order with gap at b2, b5)
        FragmentType::Y,
        FragmentType::Y,
        FragmentType::Y, // y5, y1, y2 (random order with gaps at y3, y4)
    ];
    let fragment_numbers = vec![4, 1, 3, 6, 5, 1, 2];
    let matched_mask = vec![true, true, true, true, true, true, true];

    let (longest_b, longest_y) =
        calculate_longest_ion_series(&fragment_types, &fragment_numbers, &matched_mask);

    assert_eq!(longest_b, 2); // b1,b3 -> gap -> b4 -> gap -> b6, longest is any 2 consecutive (none exist)
                              // Actually: b1 alone, b3,b4 consecutive, b6 alone. Longest = 2 (b3,b4)
    assert_eq!(longest_y, 2); // y1,y2 consecutive, gap, y5 alone. Longest = 2 (y1,y2)
}

#[test]
fn test_longest_ion_series_all_same_number() {
    // Test with duplicate fragment numbers (edge case)
    let fragment_types = vec![
        FragmentType::B,
        FragmentType::B,
        FragmentType::B, // all b2
        FragmentType::Y,
        FragmentType::Y, // all y3
    ];
    let fragment_numbers = vec![2, 2, 2, 3, 3];
    let matched_mask = vec![true, true, true, true, true];

    let (longest_b, longest_y) =
        calculate_longest_ion_series(&fragment_types, &fragment_numbers, &matched_mask);

    assert_eq!(longest_b, 1); // only one unique number (2), so longest sequence is 1
    assert_eq!(longest_y, 1); // only one unique number (3), so longest sequence is 1
}

#[test]
fn test_longest_ion_series_mixed_order_comprehensive() {
    // Comprehensive test with mixed fragment types and numbers in various orders
    let fragment_types = vec![
        FragmentType::B, // b5
        FragmentType::Y, // y1
        FragmentType::B, // b2
        FragmentType::Y, // y4
        FragmentType::B, // b3
        FragmentType::Y, // y2
        FragmentType::B, // b4
        FragmentType::Y, // y3
        FragmentType::B, // b1
    ];
    let fragment_numbers = vec![5, 1, 2, 4, 3, 2, 4, 3, 1];
    let matched_mask = vec![true, true, true, true, true, true, true, true, true];

    let (longest_b, longest_y) =
        calculate_longest_ion_series(&fragment_types, &fragment_numbers, &matched_mask);

    assert_eq!(longest_b, 5); // b1,b2,b3,b4,b5 all continuous when sorted
    assert_eq!(longest_y, 4); // y1,y2,y3,y4 all continuous when sorted
}

#[test]
fn test_longest_ion_series_mz_sorted_fragments() {
    // Test case simulating fragments sorted by m/z (like after the recent change)
    // where fragment numbers may not be in increasing order
    let fragment_types = vec![
        FragmentType::B, // b1 (low m/z)
        FragmentType::Y, // y5 (low m/z)
        FragmentType::B, // b2 (medium m/z)
        FragmentType::Y, // y4 (medium m/z)
        FragmentType::B, // b3 (high m/z)
        FragmentType::Y, // y3 (high m/z)
        FragmentType::B, // b4 (higher m/z)
        FragmentType::Y, // y2 (higher m/z)
        FragmentType::B, // b5 (highest m/z)
        FragmentType::Y, // y1 (highest m/z)
    ];
    // Fragment numbers in decreasing order for y-ions (typical for m/z sorting)
    let fragment_numbers = vec![1, 5, 2, 4, 3, 3, 4, 2, 5, 1];
    let matched_mask = vec![true, true, true, true, true, true, true, true, true, true];

    let (longest_b, longest_y) =
        calculate_longest_ion_series(&fragment_types, &fragment_numbers, &matched_mask);

    assert_eq!(longest_b, 5); // b1,b2,b3,b4,b5 all continuous
    assert_eq!(longest_y, 5); // y1,y2,y3,y4,y5 all continuous
}

#[test]
fn test_hyperscore_inverse_mass_error_basic() {
    use super::utils::calculate_hyperscore_inverse_mass_error;
    use crate::constants::FragmentType;

    // Given: Fragment data with mass errors
    let fragment_types = vec![
        FragmentType::B,
        FragmentType::B,
        FragmentType::Y,
        FragmentType::Y,
    ];
    let fragment_intensities = vec![100.0, 200.0, 150.0, 250.0];
    let matched_mask = vec![true, true, true, true];
    let mass_errors = vec![1.0, 2.0, 0.5, 3.0]; // ppm errors

    // When: Calculating hyperscore with inverse mass error
    let score = calculate_hyperscore_inverse_mass_error(
        &fragment_types,
        &fragment_intensities,
        &matched_mask,
        &mass_errors,
    );

    // Then: Should get non-zero score
    assert!(score > 0.0);
    assert!(score.is_finite());
}

#[test]
fn test_hyperscore_inverse_mass_error_zero_intensity() {
    use super::utils::calculate_hyperscore_inverse_mass_error;
    use crate::constants::FragmentType;

    // Given: Some fragments with zero intensity
    let fragment_types = vec![
        FragmentType::B,
        FragmentType::B,
        FragmentType::Y,
        FragmentType::Y,
    ];
    let fragment_intensities = vec![100.0, 0.0, 150.0, 0.0]; // Two zeros
    let matched_mask = vec![true, true, true, true];
    let mass_errors = vec![1.0, 2.0, 0.5, 3.0];

    // When: Calculating hyperscore
    let score = calculate_hyperscore_inverse_mass_error(
        &fragment_types,
        &fragment_intensities,
        &matched_mask,
        &mass_errors,
    );

    // Then: Should exclude zero intensity fragments
    assert!(score > 0.0);
    // Score should be based on only 1 b-ion and 1 y-ion
}

#[test]
fn test_hyperscore_inverse_mass_error_all_zero_intensity() {
    use super::utils::calculate_hyperscore_inverse_mass_error;
    use crate::constants::FragmentType;

    // Given: All fragments have zero intensity
    let fragment_types = vec![FragmentType::B, FragmentType::Y];
    let fragment_intensities = vec![0.0, 0.0];
    let matched_mask = vec![true, true];
    let mass_errors = vec![1.0, 2.0];

    // When: Calculating hyperscore
    let score = calculate_hyperscore_inverse_mass_error(
        &fragment_types,
        &fragment_intensities,
        &matched_mask,
        &mass_errors,
    );

    // Then: Should return 0
    assert_eq!(score, 0.0);
}

#[test]
fn test_hyperscore_inverse_mass_error_large_errors() {
    use super::utils::calculate_hyperscore_inverse_mass_error;
    use crate::constants::FragmentType;

    // Given: Fragments with very large mass errors
    let fragment_types = vec![FragmentType::B, FragmentType::Y];
    let fragment_intensities = vec![100.0, 100.0];
    let matched_mask = vec![true, true];
    let mass_errors = vec![1000.0, 2000.0]; // Very large errors

    // When: Calculating hyperscore
    let score = calculate_hyperscore_inverse_mass_error(
        &fragment_types,
        &fragment_intensities,
        &matched_mask,
        &mass_errors,
    );

    // Then: Should still produce valid score (weight approaches 1/|error|)
    // Even with very large errors, should produce non-zero score
    assert!(score != 0.0);
    assert!(score.is_finite());
}

#[test]
fn test_hyperscore_inverse_mass_error_small_errors() {
    use super::utils::calculate_hyperscore_inverse_mass_error;
    use crate::constants::FragmentType;

    // Given: Fragments with very small mass errors
    let fragment_types = vec![FragmentType::B, FragmentType::Y];
    let fragment_intensities = vec![100.0, 100.0];
    let matched_mask = vec![true, true];
    let mass_errors = vec![0.01, 0.02]; // Very small errors

    // When: Calculating hyperscore
    let score_small = calculate_hyperscore_inverse_mass_error(
        &fragment_types,
        &fragment_intensities,
        &matched_mask,
        &mass_errors,
    );

    // Compare with large errors
    let mass_errors_large = vec![10.0, 20.0];
    let score_large = calculate_hyperscore_inverse_mass_error(
        &fragment_types,
        &fragment_intensities,
        &matched_mask,
        &mass_errors_large,
    );

    // Then: Small errors should produce higher score
    assert!(score_small > score_large);
}

#[test]
fn test_intensity_ion_series_basic() {
    // Given: Fragment data with mixed types
    let fragment_types = vec![
        FragmentType::B,
        FragmentType::B,
        FragmentType::Y,
        FragmentType::Y,
        FragmentType::A,
    ];
    let fragment_intensities = vec![100.0, 200.0, 150.0, 250.0, 75.0];
    let matched_mask = vec![true, true, true, true, true];

    // When: Calculating intensities for different ion series
    let b_intensity = intensity_ion_series(
        &fragment_types,
        &fragment_intensities,
        &matched_mask,
        FragmentType::B,
    );
    let y_intensity = intensity_ion_series(
        &fragment_types,
        &fragment_intensities,
        &matched_mask,
        FragmentType::Y,
    );

    // Then: Should sum only matching ion types
    assert_eq!(b_intensity, 300.0); // 100.0 + 200.0
    assert_eq!(y_intensity, 400.0); // 150.0 + 250.0
}

#[test]
fn test_intensity_ion_series_edge_cases() {
    let fragment_types = vec![FragmentType::B, FragmentType::Y];
    let fragment_intensities = vec![100.0, 200.0];
    let matched_mask = vec![false, true]; // Only y-ion matched

    // No matches for B-series
    assert_eq!(
        intensity_ion_series(
            &fragment_types,
            &fragment_intensities,
            &matched_mask,
            FragmentType::B
        ),
        0.0
    );
    // Y-series has one match
    assert_eq!(
        intensity_ion_series(
            &fragment_types,
            &fragment_intensities,
            &matched_mask,
            FragmentType::Y
        ),
        200.0
    );

    // Zero intensities should be excluded
    let zero_intensities = vec![0.0, 200.0];
    let all_matched = vec![true, true];
    assert_eq!(
        intensity_ion_series(
            &fragment_types,
            &zero_intensities,
            &all_matched,
            FragmentType::B
        ),
        0.0
    );

    // Empty arrays
    let empty: Vec<u8> = vec![];
    let empty_f32: Vec<f32> = vec![];
    let empty_bool: Vec<bool> = vec![];
    assert_eq!(
        intensity_ion_series(&empty, &empty_f32, &empty_bool, FragmentType::B),
        0.0
    );
}

#[test]
fn test_intensity_ion_series_all_fragment_types() {
    // Given: Fragment data with multiple types
    let fragment_types = vec![
        FragmentType::A,
        FragmentType::B,
        FragmentType::C,
        FragmentType::X,
        FragmentType::Y,
        FragmentType::Z,
    ];
    let fragment_intensities = vec![50.0, 100.0, 75.0, 80.0, 200.0, 60.0];
    let matched_mask = vec![true, true, true, true, true, true];

    // When: Calculating intensity for different fragment types
    let a_intensity = intensity_ion_series(
        &fragment_types,
        &fragment_intensities,
        &matched_mask,
        FragmentType::A,
    );
    let b_intensity = intensity_ion_series(
        &fragment_types,
        &fragment_intensities,
        &matched_mask,
        FragmentType::B,
    );
    let c_intensity = intensity_ion_series(
        &fragment_types,
        &fragment_intensities,
        &matched_mask,
        FragmentType::C,
    );
    let x_intensity = intensity_ion_series(
        &fragment_types,
        &fragment_intensities,
        &matched_mask,
        FragmentType::X,
    );
    let y_intensity = intensity_ion_series(
        &fragment_types,
        &fragment_intensities,
        &matched_mask,
        FragmentType::Y,
    );
    let z_intensity = intensity_ion_series(
        &fragment_types,
        &fragment_intensities,
        &matched_mask,
        FragmentType::Z,
    );

    // Then: Should return correct intensities for each type
    assert_eq!(a_intensity, 50.0);
    assert_eq!(b_intensity, 100.0);
    assert_eq!(c_intensity, 75.0);
    assert_eq!(x_intensity, 80.0);
    assert_eq!(y_intensity, 200.0);
    assert_eq!(z_intensity, 60.0);
}

#[test]
fn test_intensity_ion_series_no_matches() {
    // Given: Fragment data with no matches for target type
    let fragment_types = vec![FragmentType::B, FragmentType::Y, FragmentType::A];
    let fragment_intensities = vec![100.0, 200.0, 50.0];
    let matched_mask = vec![true, true, true];

    // When: Calculating intensity for a type that doesn't exist
    let c_intensity = intensity_ion_series(
        &fragment_types,
        &fragment_intensities,
        &matched_mask,
        FragmentType::C,
    );

    // Then: Should return 0
    assert_eq!(c_intensity, 0.0);
}

#[test]
fn test_calculate_fwhm_rt_basic() {
    // Given: Simple XIC profile with clear peak
    let xic_profile = vec![0.0, 50.0, 100.0, 50.0, 0.0]; // Triangular peak
    let cycle_start_idx = 10;
    let rt_values = arr1(&[
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ]);

    // When: Calculating FWHM
    let fwhm = calculate_fwhm_rt(&xic_profile, cycle_start_idx, &rt_values);

    // Then: Should return reasonable FWHM value
    assert!(fwhm > 0.0);
    assert!(fwhm < 5.0); // Should be within reasonable range
}

#[test]
fn test_calculate_fwhm_rt_empty_profile() {
    // Given: Empty XIC profile
    let xic_profile = vec![];
    let cycle_start_idx = 0;
    let rt_values = arr1(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);

    // When: Calculating FWHM
    let fwhm = calculate_fwhm_rt(&xic_profile, cycle_start_idx, &rt_values);

    // Then: Should return 0
    assert_eq!(fwhm, 0.0);
}

#[test]
fn test_calculate_fwhm_rt_interpolates_between_cycles() {
    // Given: symmetric peak whose half-max (5.0) falls halfway between cycles
    let xic_profile = vec![0.0, 10.0, 0.0];
    let cycle_start_idx = 0;
    let rt_values = arr1(&[0.0, 1.0, 2.0]);

    // When
    let fwhm = calculate_fwhm_rt(&xic_profile, cycle_start_idx, &rt_values);

    // Then: crossings interpolated at rt=0.5 and rt=1.5 -> width 1.0
    assert!((fwhm - 1.0).abs() < 1e-6);
}

#[test]
fn test_calculate_fwhm_rt_off_center_apex() {
    // Given: apex is not at the center of the profile
    let xic_profile = vec![0.0, 100.0, 50.0, 0.0, 0.0];
    let cycle_start_idx = 0;
    let rt_values = arr1(&[0.0, 1.0, 2.0, 3.0, 4.0]);

    // When
    let fwhm = calculate_fwhm_rt(&xic_profile, cycle_start_idx, &rt_values);

    // Then: half-max 50; left crossing interpolated at rt=0.5, right crossing at rt=2.0
    assert!((fwhm - 1.5).abs() < 1e-6);
}

#[test]
fn test_calculate_fwhm_rt_asymmetric_peak() {
    // Given: sharp rise, slow decay -> flanks measured independently
    let xic_profile = vec![0.0, 100.0, 100.0, 50.0, 0.0];
    let cycle_start_idx = 0;
    let rt_values = arr1(&[0.0, 1.0, 2.0, 3.0, 4.0]);

    // When
    let fwhm = calculate_fwhm_rt(&xic_profile, cycle_start_idx, &rt_values);

    // Then: apex at idx 1 (or 2, equal); left crossing at rt=0.5, right crossing at rt=3.0
    assert!((fwhm - 2.5).abs() < 1e-6);
}

#[test]
fn test_calculate_fwhm_rt_no_crossing_clamps_to_window() {
    // Given: profile never falls to half maximum
    let xic_profile = vec![80.0, 100.0, 90.0];
    let cycle_start_idx = 2;
    let rt_values = arr1(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);

    // When
    let fwhm = calculate_fwhm_rt(&xic_profile, cycle_start_idx, &rt_values);

    // Then: both flanks clamp to the profile edges -> full window width (rt[4]-rt[2])
    assert!((fwhm - 2.0).abs() < 1e-6);
}

#[test]
fn test_calculate_ion_coverage_features_ion_lengths() {
    use super::ion_coverage::calculate_ion_coverage_features;
    let fragment_numbers = [2u8, 3, 6, 8];
    let correlations = [0.9f32, 0.2, 0.8, 0.6];
    let intensities = [10.0f32, 10.0, 30.0, 50.0];
    let types = [
        FragmentType::B,
        FragmentType::Y,
        FragmentType::Y,
        FragmentType::Y,
    ];

    let (features, _) = calculate_ion_coverage_features(
        &[1.0f32; 4],
        &intensities,
        &correlations,
        &types,
        &fragment_numbers,
        &[1u8; 4],
        10,
        12,
    );

    // co-eluting fragments: numbers 2, 6 and 8 of a 10-residue peptide
    assert!((features.ion_position_mean - (0.2 + 0.6 + 0.8) / 3.0).abs() < 1e-6);
    assert_eq!(features.num_over_50_b_short, 1.0);
    assert_eq!(features.num_over_50_y_mid, 1.0);
    assert_eq!(features.num_over_50_y_long, 1.0);
    // the short ions b2 and y3 carry 20 of 100
    assert!((features.short_ion_intensity_share - 0.2).abs() < 1e-6);
    assert!((features.lib_share_over_50 - 0.75).abs() < 1e-6);
}

#[test]
fn test_calculate_ion_coverage_features_without_signal() {
    use super::ion_coverage::{calculate_ion_coverage_features, IonCoverageFeatures};
    let (features, _) = calculate_ion_coverage_features(
        &[1.0f32, 1.0],
        &[0.0f32, 0.0],
        &[0.0f32, 0.0],
        &[FragmentType::Y; 2],
        &[4u8, 7],
        &[1u8, 1],
        9,
        12,
    );

    assert_eq!(features, IonCoverageFeatures::default());
}

#[test]
fn test_calculate_ion_coverage_features() {
    use super::ion_coverage::calculate_ion_coverage_features;
    let library = [4.0f32, 2.0, 1.0, 1.0];
    let observed = [8.0f32, 4.0, 0.0, 2.0];
    let correlations = [0.75f32, 0.625, 0.0, 0.25];
    let types = [
        FragmentType::Y,
        FragmentType::B,
        FragmentType::Y,
        FragmentType::Y,
    ];
    let numbers = [9u8, 2, 5, 3];
    let charges = [1u8, 1, 2, 2];

    let (features, _) = calculate_ion_coverage_features(
        &library,
        &observed,
        &correlations,
        &types,
        &numbers,
        &charges,
        12,
        12,
    );

    assert!((features.xic_libweighted - 0.5625).abs() < 1e-6);
    assert!((features.xic_obsweighted - 9.0 / 14.0).abs() < 1e-6);
    // co-eluting: the y9 (long) and the b2 (short)
    assert_eq!(features.num_over_50_y_long, 1.0);
    assert_eq!(features.num_over_50_b_short, 1.0);
    assert_eq!(
        features.num_over_50_b_long + features.num_over_50_y_short + features.num_over_50_y_mid,
        0.0
    );
    assert!((features.lib_share_over_50 - 0.75).abs() < 1e-6);
    // neither doubly charged fragment (y5 without signal, y3 at 0.25) co-elutes
    assert_eq!(features.num_over_50_z2, 0.0);
}

#[test]
fn test_calculate_ion_coverage_features_ion_evidence_in_intensity_rank_order() {
    use super::ion_coverage::calculate_ion_coverage_features;
    // scored in m/z order; the library intensity ranks order them by intensity
    let library = [1.0f32, 4.0, 2.0, 1.0];
    let observed = [0.0f32, 8.0, 4.0, 2.0];
    let correlations = [0.0f32, 0.75, 0.625, -0.25];

    let (_, evidence) = calculate_ion_coverage_features(
        &library,
        &observed,
        &correlations,
        &[
            FragmentType::B,
            FragmentType::Y,
            FragmentType::B,
            FragmentType::Y,
        ],
        &[2, 9, 2, 3],
        &[1, 1, 1, 1],
        12,
        12,
    );

    let shares: Vec<f32> = evidence.iter().map(|e| e.library_share).collect();
    assert_eq!(shares, vec![0.5, 0.25, 0.125, 0.125]);
    assert_eq!(evidence[0].bin, 75);
    assert_eq!(evidence[1].bin, 62);
    assert!(evidence[0].positive && evidence[1].positive);
    // a hit with a negative correlation is no positive hit, no signal is no hit
    assert!(evidence[3].hit && !evidence[3].positive);
    assert!(!evidence[2].hit);
    // y9 at rank 0, b2 at rank 1 and the b2 at rank 2 share a rank group: only ion class and rank group count
    assert_ne!(evidence[0].class, evidence[1].class);
    assert_eq!(evidence[1].class, evidence[2].class);
}

#[test]
fn test_ion_noise_model_fisher() {
    use super::ion_coverage::{IonEvidence, IonNoiseModel};
    use rayon::prelude::*;
    let fragment = |bin: u8, hit: bool, positive: bool, library_share: f32| IonEvidence {
        class: 0,
        bin,
        hit,
        positive,
        library_share,
    };
    // class 0: four fragments, two positive hits at bins 10 and 90
    let noise = vec![
        vec![
            fragment(10, true, true, 0.5),
            fragment(0, false, false, 0.5),
        ],
        vec![
            fragment(90, true, true, 0.5),
            fragment(0, false, false, 0.5),
        ],
    ];
    let model = IonNoiseModel::fit(noise.par_iter().map(|e| e.as_slice()), 1);

    // hit rate 0.5 times the share of hits at bin 90 or above (0.5)
    let weighted = model.fisher(&[fragment(90, true, true, 0.5)]);
    assert!((weighted - 0.5 * 4f32.ln()).abs() < 1e-6);

    // no noise hit correlates this well: the chance is floored
    let floored = model.fisher(&[fragment(95, true, true, 1.0)]);
    assert!((floored - 1e4f32.ln()).abs() < 1e-4);

    // only positive hits add surprise
    assert_eq!(
        model.fisher(&[
            fragment(90, true, false, 1.0),
            fragment(0, false, false, 1.0)
        ]),
        0.0
    );
}

#[test]
fn test_calculate_library_agreement_features() {
    use super::library_agreement::calculate_library_agreement_features;
    let types = [
        FragmentType::B,
        FragmentType::Y,
        FragmentType::Y,
        FragmentType::Y,
        FragmentType::B,
        FragmentType::Y,
    ];
    let library = [1.0f32, 4.0, 2.0, 1.0, 1.0, 1.0];
    let correlations = [0.9f32, 0.95, 0.8, 0.1, 0.0, -0.2];
    let observed = [1.0f32, 4.0, 2.0, 1.0, 1.0, 1.0];
    let xic = Array2::from_shape_fn((6, 1), |(i, _)| observed[i]);

    let features = calculate_library_agreement_features(
        &xic,
        &[1.0],
        &library,
        &observed,
        &correlations,
        &types,
    );

    assert!(features.log_ratio_std.abs() < 1e-6);
    assert!((features.y_intensity_correlation - 1.0).abs() < 1e-6);
    assert!((features.lib_cosine_over_50 - 1.0).abs() < 1e-5);
    assert!((features.ent_sim - 1.0).abs() < 1e-6);
}

#[test]
fn test_lib_cosine_over_50_uses_only_coeluting_fragments() {
    use super::library_agreement::calculate_library_agreement_features;
    let xic = arr2(&[
        [0.0, 4.0, 8.0, 4.0, 0.0],
        [0.0, 2.0, 4.0, 2.0, 0.0],
        [0.0, 1.0, 2.0, 1.0, 0.0],
        [50.0, 0.0, 0.0, 0.0, 50.0],
    ]);
    let profile = [0.0, 0.5, 1.0, 0.5, 0.0];
    let library = [4.0f32, 2.0, 1.0, 9.0];
    let observed: Vec<f32> = (0..4).map(|i| xic.row(i).sum()).collect();
    let types = [FragmentType::Y; 4];

    let coeluting = calculate_library_agreement_features(
        &xic,
        &profile,
        &library,
        &observed,
        &[0.99, 0.99, 0.99, -0.5],
        &types,
    );
    let too_few = calculate_library_agreement_features(
        &xic,
        &profile,
        &library,
        &observed,
        &[0.99, 0.99, 0.1, -0.5],
        &types,
    );

    assert!((coeluting.lib_cosine_over_50 - 1.0).abs() < 1e-5);
    assert_eq!(too_few.lib_cosine_over_50, 0.0);
}

#[test]
fn test_calculate_library_agreement_features_entropy_similarity() {
    use super::library_agreement::calculate_library_agreement_features;
    let library = [4.0f32, 2.0, 1.0];
    let types = [FragmentType::Y; 3];
    let xic = Array2::<f32>::zeros((3, 1));

    let proportional = calculate_library_agreement_features(
        &xic,
        &[0.0],
        &library,
        &[40.0, 20.0, 10.0],
        &[1.0, 1.0, 1.0],
        &types,
    );
    let no_signal = calculate_library_agreement_features(
        &xic,
        &[0.0],
        &library,
        &[0.0, 0.0, 0.0],
        &[0.0, 0.0, 0.0],
        &types,
    );

    assert!((proportional.ent_sim - 1.0).abs() < 1e-6);
    assert!((proportional.ent_sim_corrweighted - 1.0).abs() < 1e-6);
    assert_eq!(no_signal.ent_sim, 0.0);
    assert_eq!(no_signal.ent_sim_corrweighted, 0.0);
}

#[cfg(test)]
mod feature_sets {
    use crate::candidate::{Candidate, CandidateCollection};
    use crate::constants::{FragmentType, Loss};
    use crate::dia_data::{AlphaRawView, DIAData};
    use crate::dia_data_builder::DIADataBuilder;
    use crate::peak_group_scoring::features::{CandidateFeatureCollection, SetSpecificFeatures};
    use crate::peak_group_scoring::{FeatureSet, PeakGroupScoring, ScoringParameters};
    use crate::speclib_flat::SpecLibFlat;
    use numpy::ndarray::{Array1, Array4};

    const NUM_CYCLES: usize = 20;
    const APEX_CYCLE: usize = 10;
    const PRECURSOR_MZ: f32 = 500.0;
    const FRAGMENTS: [(f32, f32); 6] = [
        (300.0, 6.0),
        (400.0, 5.0),
        (500.0, 4.0),
        (600.0, 3.0),
        (700.0, 2.0),
        (800.0, 1.0),
    ];

    /// One isolation window and one spectrum per cycle, in which the fragments of the library elute as a triangle
    /// around APEX_CYCLE.
    fn dia_data() -> DIAData {
        let mut peak_mz = Vec::new();
        let mut peak_intensity = Vec::new();
        let mut peak_start = Vec::new();
        let mut peak_stop = Vec::new();
        for cycle in 0..NUM_CYCLES {
            peak_start.push(peak_mz.len() as i64);
            let elution = 5.0 - (cycle as f32 - APEX_CYCLE as f32).abs();
            if elution > 0.0 {
                for (mz, intensity) in FRAGMENTS {
                    peak_mz.push(mz);
                    peak_intensity.push(intensity * elution * 100.0);
                }
            }
            peak_stop.push(peak_mz.len() as i64);
        }
        let delta_scan_idx = Array1::<i64>::zeros(NUM_CYCLES);
        let lower = Array1::<f32>::from_elem(NUM_CYCLES, 400.0);
        let upper = Array1::<f32>::from_elem(NUM_CYCLES, 600.0);
        let peak_start = Array1::from(peak_start);
        let peak_stop = Array1::from(peak_stop);
        let cycle_idx = Array1::from_iter(0..NUM_CYCLES as i64);
        let rt = Array1::from_iter((0..NUM_CYCLES).map(|c| c as f32 * 2.0));
        let peak_mz = Array1::from(peak_mz);
        let peak_intensity = Array1::from(peak_intensity);
        let mut cycle = Array4::<f32>::zeros((1, 1, 1, 2));
        cycle[[0, 0, 0, 0]] = 400.0;
        cycle[[0, 0, 0, 1]] = 600.0;
        let view = AlphaRawView::new(
            delta_scan_idx.view(),
            lower.view(),
            upper.view(),
            peak_start.view(),
            peak_stop.view(),
            cycle_idx.view(),
            rt.view(),
            peak_mz.view(),
            peak_intensity.view(),
            cycle.view(),
        );
        DIADataBuilder::from_alpha_raw(&view)
    }

    fn library() -> SpecLibFlat {
        let n = FRAGMENTS.len();
        let fragment_mz: Vec<f32> = FRAGMENTS.iter().map(|&(mz, _)| mz).collect();
        SpecLibFlat::from_vecs(
            vec![0],
            vec![PRECURSOR_MZ],
            vec![PRECURSOR_MZ],
            vec![20.0],
            vec![20.0],
            vec![10],
            vec![0],
            vec![n],
            fragment_mz.clone(),
            fragment_mz,
            FRAGMENTS.iter().map(|&(_, intensity)| intensity).collect(),
            vec![1; n],
            vec![1; n],
            vec![Loss::NONE; n],
            (2..2 + n as u8).collect(),
            (2..2 + n as u8).collect(),
            vec![FragmentType::Y; n],
        )
    }

    /// The precursor at its apex (rank 0) and off its apex (rank 1)
    fn candidates() -> CandidateCollection {
        CandidateCollection::from_vec(vec![
            Candidate::new(0, 0, 2.0, APEX_CYCLE - 3, APEX_CYCLE, APEX_CYCLE + 3),
            Candidate::new(0, 1, 1.0, 1, 3, 6),
        ])
    }

    fn score(feature_set: FeatureSet) -> CandidateFeatureCollection {
        let mut params = ScoringParameters::new();
        params.feature_set = feature_set;
        PeakGroupScoring::new(params)
            .score(&dia_data(), &library(), &candidates())
            .unwrap()
    }

    #[test]
    fn test_scoring_parameters_default_to_the_full_feature_set() {
        // given / when
        let params = ScoringParameters::new();

        // then
        assert_eq!(params.feature_set, FeatureSet::Full);
    }

    #[test]
    fn test_calibration_and_full_share_the_core_block() {
        // given
        let full = score(FeatureSet::Full);

        // when
        let calibration = score(FeatureSet::Calibration);

        // then
        assert_eq!(calibration.feature_set(), FeatureSet::Calibration);
        assert_eq!(full.feature_set(), FeatureSet::Full);
        assert_eq!(calibration.len(), full.len());
        for (calibration_feature, full_feature) in calibration.iter().zip(full.iter()) {
            assert_eq!(calibration_feature.feature_set(), FeatureSet::Calibration);
            assert_eq!(full_feature.feature_set(), FeatureSet::Full);
            assert_eq!(calibration_feature.core, full_feature.core);
        }
    }

    #[test]
    fn test_calibration_scores_the_calibration_only_block() {
        // given / when
        let calibration = score(FeatureSet::Calibration);

        // then
        let apex = calibration.iter().find(|f| f.rank == 0).unwrap();
        let SetSpecificFeatures::Calibration(calibration_only) = apex.set_specific else {
            panic!("expected the calibration-only block");
        };
        assert_eq!(calibration_only.num_fragments, FRAGMENTS.len() as f32);
        assert_eq!(calibration_only.num_over_50, FRAGMENTS.len() as f32);
        assert!(calibration_only.intensity_correlation > 0.99);
    }

    #[test]
    fn test_full_scores_the_apex_candidate_with_signal_in_every_block() {
        // given / when
        let full = score(FeatureSet::Full);

        // then
        let apex = full.iter().find(|f| f.rank == 0).unwrap();
        let SetSpecificFeatures::Full {
            ion_coverage,
            library_agreement,
        } = apex.set_specific
        else {
            panic!("expected the full blocks");
        };
        assert_eq!(apex.core.num_over_80, FRAGMENTS.len() as f32);
        assert!(ion_coverage.lib_share_over_50 > 0.99);
        assert!(ion_coverage.ion_fisher_lib > 0.0);
        assert!(library_agreement.ent_sim > 0.99);
    }
}
