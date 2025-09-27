use super::*;
use numpy::ndarray::arr2;

#[test]
fn test_weighted_mean_nonzero_axis_1_basic() {
    // Given: Simple 2x3 arrays with known values
    let values = arr2(&[[10.0, 20.0, 30.0], [5.0, 15.0, 25.0]]);
    let weights = arr2(&[
        [1.0, 2.0, 3.0],
        [4.0, 0.0, 2.0], // Middle weight is 0
    ]);

    // When: Calculating weighted mean along axis 1
    let result = weighted_mean_nonzero_axis_1(&values, &weights);

    // Then: Should get correct weighted averages
    // Row 0: (10*1 + 20*2 + 30*3) / (1+2+3) = 140/6 = 23.333...
    assert!((result[0] - 23.333334).abs() < 1e-5);
    // Row 1: (5*4 + 25*2) / (4+2) = 70/6 = 11.666... (15 is excluded due to 0 weight)
    assert!((result[1] - 11.666667).abs() < 1e-5);
}

#[test]
fn test_weighted_mean_nonzero_axis_1_all_zeros() {
    // Given: Arrays where all weights in a row are zero
    let values = arr2(&[[10.0, 20.0, 30.0], [5.0, 15.0, 25.0]]);
    let weights = arr2(&[
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0], // All weights are zero
    ]);

    // When: Calculating weighted mean
    let result = weighted_mean_nonzero_axis_1(&values, &weights);

    // Then: Row with all zero weights should return 0
    assert!((result[0] - 23.333334).abs() < 1e-5);
    assert_eq!(result[1], 0.0);
}

#[test]
fn test_weighted_mean_nonzero_axis_1_single_value() {
    // Given: 1x1 arrays
    let values = arr2(&[[42.0]]);
    let weights = arr2(&[[2.5]]);

    // When: Calculating weighted mean
    let result = weighted_mean_nonzero_axis_1(&values, &weights);

    // Then: Should return the single value
    assert_eq!(result[0], 42.0);
}

#[test]
fn test_weighted_mean_nonzero_axis_1_negative_values() {
    // Given: Arrays with negative values
    let values = arr2(&[[-10.0, 20.0, -30.0], [5.0, -15.0, 25.0]]);
    let weights = arr2(&[[1.0, 2.0, 1.0], [1.0, 1.0, 1.0]]);

    // When: Calculating weighted mean
    let result = weighted_mean_nonzero_axis_1(&values, &weights);

    // Then: Should handle negative values correctly
    // Row 0: (-10*1 + 20*2 + -30*1) / (1+2+1) = 0/4 = 0
    assert_eq!(result[0], 0.0);
    // Row 1: (5*1 + -15*1 + 25*1) / (1+1+1) = 15/3 = 5
    assert_eq!(result[1], 5.0);
}

#[test]
fn test_weighted_mean_nonzero_axis_1_mixed_zeros() {
    // Given: Arrays with some zero weights mixed in
    let values = arr2(&[[100.0, 200.0, 300.0, 400.0], [10.0, 20.0, 30.0, 40.0]]);
    let weights = arr2(&[
        [1.0, 0.0, 3.0, 0.0], // Alternating zeros
        [0.0, 2.0, 0.0, 4.0], // Alternating zeros
    ]);

    // When: Calculating weighted mean
    let result = weighted_mean_nonzero_axis_1(&values, &weights);

    // Then: Should only use non-zero weighted values
    // Row 0: (100*1 + 300*3) / (1+3) = 1000/4 = 250
    assert_eq!(result[0], 250.0);
    // Row 1: (20*2 + 40*4) / (2+4) = 200/6 = 33.333...
    assert!((result[1] - 33.333332).abs() < 1e-5);
}

#[test]
fn test_calculate_weighted_mean_absolute_error_basic() {
    // Given: Mass errors and intensity weights
    let mass_errors = vec![10.0, -20.0, 30.0, -40.0];
    let intensity_weights = vec![1.0, 2.0, 3.0, 4.0];

    // When: Calculating weighted mean absolute error
    let result = calculate_weighted_mean_absolute_error(&mass_errors, &intensity_weights);

    // Then: Should get correct weighted mean of absolute values
    // (|10|*1 + |-20|*2 + |30|*3 + |-40|*4) / (1+2+3+4) = (10+40+90+160) / 10 = 30
    assert!((result - 30.0).abs() < 1e-5);
}

#[test]
fn test_calculate_weighted_mean_absolute_error_zero_weights() {
    // Given: Some weights are zero
    let mass_errors = vec![10.0, -20.0, 30.0, -40.0];
    let intensity_weights = vec![1.0, 0.0, 3.0, 0.0];

    // When: Calculating weighted mean absolute error
    let result = calculate_weighted_mean_absolute_error(&mass_errors, &intensity_weights);

    // Then: Should ignore entries with zero weight
    // (|10|*1 + |30|*3) / (1+3) = (10+90) / 4 = 25
    assert!((result - 25.0).abs() < 1e-5);
}

#[test]
fn test_calculate_weighted_mean_absolute_error_zero_mass_errors() {
    // Given: Some mass errors are zero (no signal)
    let mass_errors = vec![10.0, 0.0, 30.0, 0.0];
    let intensity_weights = vec![1.0, 2.0, 3.0, 4.0];

    // When: Calculating weighted mean absolute error
    let result = calculate_weighted_mean_absolute_error(&mass_errors, &intensity_weights);

    // Then: Should ignore entries with zero mass error
    // (|10|*1 + |30|*3) / (1+3) = (10+90) / 4 = 25
    assert!((result - 25.0).abs() < 1e-5);
}

#[test]
fn test_calculate_weighted_mean_absolute_error_all_zeros() {
    // Given: All weights are zero
    let mass_errors = vec![10.0, -20.0, 30.0];
    let intensity_weights = vec![0.0, 0.0, 0.0];

    // When: Calculating weighted mean absolute error
    let result = calculate_weighted_mean_absolute_error(&mass_errors, &intensity_weights);

    // Then: Should return 0
    assert_eq!(result, 0.0);
}

#[test]
fn test_calculate_weighted_mean_absolute_error_empty() {
    // Given: Empty arrays
    let mass_errors: Vec<f32> = vec![];
    let intensity_weights: Vec<f32> = vec![];

    // When: Calculating weighted mean absolute error
    let result = calculate_weighted_mean_absolute_error(&mass_errors, &intensity_weights);

    // Then: Should return 0
    assert_eq!(result, 0.0);
}

#[test]
fn test_calculate_weighted_mean_absolute_error_mismatched_lengths() {
    // Given: Arrays with different lengths
    let mass_errors = vec![10.0, -20.0];
    let intensity_weights = vec![1.0, 2.0, 3.0];

    // When: Calculating weighted mean absolute error
    let result = calculate_weighted_mean_absolute_error(&mass_errors, &intensity_weights);

    // Then: Should return 0
    assert_eq!(result, 0.0);
}

#[test]
fn test_calculate_weighted_mean_absolute_error_single_value() {
    // Given: Single value arrays
    let mass_errors = vec![25.0];
    let intensity_weights = vec![2.0];

    // When: Calculating weighted mean absolute error
    let result = calculate_weighted_mean_absolute_error(&mass_errors, &intensity_weights);

    // Then: Should return the absolute value
    assert_eq!(result, 25.0);
}

#[test]
fn test_calculate_weighted_mean_absolute_error_negative_weights() {
    // Given: Negative weights (should be treated as invalid and ignored)
    let mass_errors = vec![10.0, 20.0, 30.0];
    let intensity_weights = vec![1.0, -2.0, 3.0];

    // When: Calculating weighted mean absolute error
    let result = calculate_weighted_mean_absolute_error(&mass_errors, &intensity_weights);

    // Then: Should only use positive weights
    // (|10|*1 + |30|*3) / (1+3) = (10+90) / 4 = 25
    assert!((result - 25.0).abs() < 1e-5);
}

#[test]
fn test_calculate_fragment_mz_and_errors() {
    // Given: Dense m/z and XIC matrices with known values
    let dense_mz = arr2(&[
        [200.01, 200.02, 200.03], // Fragment 0 m/z values
        [300.05, 300.10, 300.15], // Fragment 1 m/z values
        [0.0, 0.0, 0.0],          // Fragment 2 no signal
    ]);
    let dense_xic = arr2(&[
        [1000.0, 2000.0, 3000.0], // Fragment 0 intensities
        [500.0, 1000.0, 500.0],   // Fragment 1 intensities
        [0.0, 0.0, 0.0],          // Fragment 2 no signal
    ]);
    let mz_library = vec![200.0, 300.0, 400.0];

    // When: Calculating fragment m/z and errors
    let (mz_observed, mass_errors) =
        calculate_fragment_mz_and_errors(&dense_mz, &dense_xic, &mz_library);

    // Then: Should get correct weighted averages and mass errors
    // Fragment 0: (200.01*1000 + 200.02*2000 + 200.03*3000) / 6000 = 200.02333...
    assert!((mz_observed[0] - 200.02333).abs() < 1e-5);
    // Mass error: (200.02333 - 200.0) / 200.0 * 1e6 = 116.67 ppm
    assert!((mass_errors[0] - 116.67).abs() < 0.1);

    // Fragment 1: (300.05*500 + 300.10*1000 + 300.15*500) / 2000 = 300.1
    assert!((mz_observed[1] - 300.1).abs() < 1e-5);
    // Mass error: (300.1 - 300.0) / 300.0 * 1e6 = 333.33 ppm
    assert!((mass_errors[1] - 333.33).abs() < 0.1);

    // Fragment 2: No signal
    assert_eq!(mz_observed[2], 0.0);
    assert_eq!(mass_errors[2], 0.0);
}

#[test]
fn test_calculate_fragment_mz_and_errors_edge_cases() {
    // Given: Edge case with single fragment and cycle
    let dense_mz = arr2(&[[250.005]]);
    let dense_xic = arr2(&[[1500.0]]);
    let mz_library = vec![250.0];

    // When: Calculating fragment m/z and errors
    let (mz_observed, mass_errors) =
        calculate_fragment_mz_and_errors(&dense_mz, &dense_xic, &mz_library);

    // Then: Should handle single value correctly
    assert_eq!(mz_observed[0], 250.005);
    // Mass error: (250.005 - 250.0) / 250.0 * 1e6 = 20 ppm
    assert!((mass_errors[0] - 20.0).abs() < 0.1);
}

#[test]
fn test_calculate_fragment_mz_and_errors_partial_signal() {
    // Given: Some cycles have signal, others don't
    let dense_mz = arr2(&[
        [150.01, 0.0, 150.03, 0.0], // Fragment with partial signal
        [0.0, 200.02, 0.0, 200.04], // Different pattern
    ]);
    let dense_xic = arr2(&[
        [1000.0, 0.0, 2000.0, 0.0], // Matching intensities
        [0.0, 3000.0, 0.0, 1000.0], // Different pattern
    ]);
    let mz_library = vec![150.0, 200.0];

    // When: Calculating fragment m/z and errors
    let (mz_observed, mass_errors) =
        calculate_fragment_mz_and_errors(&dense_mz, &dense_xic, &mz_library);

    // Then: Should only use non-zero signals
    // Fragment 0: (150.01*1000 + 150.03*2000) / 3000 = 150.02333...
    assert!((mz_observed[0] - 150.02333).abs() < 1e-5);
    assert!((mass_errors[0] - 155.56).abs() < 0.2); // (150.02333 - 150.0) / 150.0 * 1e6
                                                    // Fragment 1: (200.02*3000 + 200.04*1000) / 4000 = 200.025
    assert!((mz_observed[1] - 200.025).abs() < 1e-5);
    assert!((mass_errors[1] - 125.0).abs() < 0.1); // (200.025 - 200.0) / 200.0 * 1e6
}

#[test]
fn test_calculate_median() {
    assert_eq!(calculate_median(&[3.0, 1.0, 4.0, 1.0, 5.0]), 3.0); // odd length
    assert_eq!(calculate_median(&[2.0, 4.0, 1.0, 3.0]), 2.5); // even length
    assert_eq!(calculate_median(&[42.0]), 42.0); // single value
    assert_eq!(calculate_median(&[]), 0.0); // empty
    assert_eq!(calculate_median(&[-3.0, 1.0, -1.0, 3.0, 0.0]), 0.0); // negative values
    assert!(calculate_median(&[1.0, f32::NAN, 3.0]).is_nan()); // NaN handling
}

#[test]
fn test_calculate_std() {
    assert!((calculate_std(&[1.0, 2.0, 3.0, 4.0, 5.0]) - 1.5811).abs() < 1e-4); // basic case
    assert_eq!(calculate_std(&[42.0]), 0.0); // single value
    assert_eq!(calculate_std(&[]), 0.0); // empty
    assert!((calculate_std(&[1.0, 5.0]) - 2.8284).abs() < 1e-4); // two values
    assert_eq!(calculate_std(&[7.0, 7.0, 7.0]), 0.0); // identical values
    assert_eq!(calculate_std(&[-2.0, 0.0, 2.0]), 2.0); // negative values
}

#[test]
fn test_create_ranked_mask_basic_top_k() {
    // Select top 2 from [1.0, 3.0, 2.0] -> should select 3.0, 2.0 (ranks 0-1)
    assert_eq!(
        create_ranked_mask(&[1.0, 3.0, 2.0], 0, 2),
        [false, true, true]
    );

    // Select top 2 from [5.0, 1.0, 3.0, 2.0] -> should select 5.0, 3.0 (ranks 0-1)
    assert_eq!(
        create_ranked_mask(&[5.0, 1.0, 3.0, 2.0], 0, 2),
        [true, false, true, false]
    );
}

#[test]
fn test_create_ranked_mask_ties() {
    // With ties, select first occurrences: [1.0, 1.0, 1.0], ranks 0-1 -> first two
    assert_eq!(
        create_ranked_mask(&[1.0, 1.0, 1.0], 0, 2),
        [true, true, false]
    );

    // Mixed values with ties: [3.0, 1.0, 3.0, 2.0], ranks 0-1 -> first 3.0 and second 3.0
    assert_eq!(
        create_ranked_mask(&[3.0, 1.0, 3.0, 2.0], 0, 2),
        [true, false, true, false]
    );
}

#[test]
fn test_create_ranked_mask_k_zero() {
    // When r1 >= r2, should return all false
    assert_eq!(
        create_ranked_mask(&[1.0, 2.0, 3.0], 0, 0),
        [false, false, false]
    );
    assert_eq!(
        create_ranked_mask(&[1.0, 2.0, 3.0], 2, 1),
        [false, false, false]
    );
}

#[test]
fn test_create_ranked_mask_empty_array() {
    assert_eq!(create_ranked_mask(&[], 0, 5), Vec::<bool>::new());
}

#[test]
fn test_create_ranked_mask_k_exceeds_length() {
    // r2=5 but only 3 elements -> select all (ranks 0-2)
    assert_eq!(
        create_ranked_mask(&[1.0, 2.0, 3.0], 0, 5),
        [true, true, true]
    );
    // ranks 1-5 but only 3 elements -> select ranks 1-2
    assert_eq!(
        create_ranked_mask(&[1.0, 2.0, 3.0], 1, 5),
        [true, true, false]
    );
}

#[test]
fn test_create_ranked_mask_single_element() {
    assert_eq!(create_ranked_mask(&[42.0], 0, 1), [true]);
    assert_eq!(create_ranked_mask(&[42.0], 0, 0), [false]);
    assert_eq!(create_ranked_mask(&[42.0], 1, 2), [false]); // rank 1 doesn't exist
}

#[test]
fn test_create_ranked_mask_descending_order() {
    // Already sorted descending: [5.0, 4.0, 3.0, 2.0, 1.0], ranks 0-2
    assert_eq!(
        create_ranked_mask(&[5.0, 4.0, 3.0, 2.0, 1.0], 0, 3),
        [true, true, true, false, false]
    );
}

#[test]
fn test_create_ranked_mask_ascending_order() {
    // Ascending order: [1.0, 2.0, 3.0, 4.0, 5.0], ranks 0-2 -> select 5.0, 4.0, 3.0
    assert_eq!(
        create_ranked_mask(&[1.0, 2.0, 3.0, 4.0, 5.0], 0, 3),
        [false, false, true, true, true]
    );
}

#[test]
fn test_create_ranked_mask_negative_values() {
    // Include negative values: [-1.0, 2.0, -3.0, 4.0], ranks 0-1 -> select 4.0, 2.0
    assert_eq!(
        create_ranked_mask(&[-1.0, 2.0, -3.0, 4.0], 0, 2),
        [false, true, false, true]
    );
}

#[test]
fn test_create_ranked_mask_ranges() {
    // Test different rank ranges with [5.0, 1.0, 3.0, 2.0, 4.0]
    // Sorted by rank: [5.0(0), 4.0(1), 3.0(2), 2.0(3), 1.0(4)]
    let values = [5.0, 1.0, 3.0, 2.0, 4.0];

    // Ranks 0-1 (top 2): 5.0, 4.0
    assert_eq!(
        create_ranked_mask(&values, 0, 2),
        [true, false, false, false, true]
    );

    // Ranks 2-3 (next 2): 3.0, 2.0
    assert_eq!(
        create_ranked_mask(&values, 2, 4),
        [false, false, true, true, false]
    );

    // Rank 4 (last): 1.0
    assert_eq!(
        create_ranked_mask(&values, 4, 5),
        [false, true, false, false, false]
    );
}

#[test]
fn test_count_values_above_no_mask() {
    assert_eq!(count_values_above(&[1.0, 2.0, 3.0, 4.0], 2.5, None), 2); // 3.0, 4.0
    assert_eq!(count_values_above(&[1.0, 2.0, 3.0, 4.0], 0.0, None), 4); // all values
    assert_eq!(count_values_above(&[1.0, 2.0, 3.0, 4.0], 5.0, None), 0); // no values
    assert_eq!(count_values_above(&[], 1.0, None), 0); // empty slice
    assert_eq!(count_values_above(&[2.0, 2.0, 2.0], 2.0, None), 0); // exact threshold
}

#[test]
fn test_count_values_above_with_mask() {
    let values = &[1.0, 2.0, 3.0, 4.0];
    let mask = &[true, false, true, false];
    assert_eq!(count_values_above(values, 1.5, Some(mask)), 1); // only 3.0 passes (2.0 masked out)

    let all_true = &[true, true, true, true];
    assert_eq!(count_values_above(values, 1.5, Some(all_true)), 3); // 2.0, 3.0, 4.0

    let all_false = &[false, false, false, false];
    assert_eq!(count_values_above(values, 0.0, Some(all_false)), 0); // all masked out
}

#[test]
fn test_count_values_above_mismatched_lengths() {
    let values = &[1.0, 2.0, 3.0];
    let short_mask = &[true, false];
    assert_eq!(count_values_above(values, 0.0, Some(short_mask)), 0); // mismatched lengths
}
