use super::*;

// Test fixtures for common test parameters
struct TestParams {
    resolution_ppm: f32,
    mz_start: f32,
    mz_end: f32,
}

fn standard_params() -> TestParams {
    TestParams {
        resolution_ppm: 1.0,
        mz_start: 100.0,
        mz_end: 1000.0,
    }
}

fn high_resolution_params() -> TestParams {
    TestParams {
        resolution_ppm: 0.5,
        mz_start: 200.0,
        mz_end: 300.0,
    }
}

fn low_resolution_params() -> TestParams {
    TestParams {
        resolution_ppm: 2.0,
        mz_start: 500.0,
        mz_end: 600.0,
    }
}

fn small_range_params() -> TestParams {
    TestParams {
        resolution_ppm: 1.0,
        mz_start: 100.0,
        mz_end: 101.0,
    }
}

fn clamping_test_params() -> TestParams {
    TestParams {
        resolution_ppm: 1.0,
        mz_start: 10.0, // Below 50, should be clamped
        mz_end: 100.0,
    }
}

fn very_high_resolution_params() -> TestParams {
    TestParams {
        resolution_ppm: 0.1,
        mz_start: 500.0,
        mz_end: 600.0,
    }
}

fn very_low_resolution_params() -> TestParams {
    TestParams {
        resolution_ppm: 10.0,
        mz_start: 500.0,
        mz_end: 600.0,
    }
}

// Helper function to verify geometric progression
fn verify_geometric_progression(result: &Array1<f32>, expected_ratio: f32) {
    for i in 1..std::cmp::min(10, result.len()) {
        let actual_ratio = result[i] / result[i - 1];
        assert!(
            (actual_ratio - expected_ratio).abs() < 1e-6,
            "Geometric progression broken at index {}: ratio={}, expected={}",
            i,
            actual_ratio,
            expected_ratio
        );
    }
}

#[test]
fn test_ppm_index_len() {
    let result = ppm_index(1.0, 100.0, 2000.0);
    assert_eq!(result.len(), 2995974);
}

#[test]
fn test_ppm_index_standard_case() {
    let params = standard_params();
    let result = ppm_index(params.resolution_ppm, params.mz_start, params.mz_end);

    // Verify basic properties
    assert!(!result.is_empty());
    assert!(result[0] >= params.mz_start);
    assert!(result[result.len() - 1] <= params.mz_end * 1.01);

    // Verify geometric progression
    let expected_ratio = 1.0 + (params.resolution_ppm / 1e6);
    verify_geometric_progression(&result, expected_ratio);
}

#[test]
fn test_ppm_index_high_resolution() {
    let params = high_resolution_params();
    let result = ppm_index(params.resolution_ppm, params.mz_start, params.mz_end);

    assert!(!result.is_empty());
    assert!(result[0] >= params.mz_start);
    assert!(result[result.len() - 1] <= params.mz_end * 1.01);

    // Should have more points due to smaller step size
    assert!(result.len() > 100);

    let expected_ratio = 1.0 + (params.resolution_ppm / 1e6);
    verify_geometric_progression(&result, expected_ratio);
}

#[test]
fn test_ppm_index_low_resolution() {
    let params = low_resolution_params();
    let result = ppm_index(params.resolution_ppm, params.mz_start, params.mz_end);

    assert!(!result.is_empty());
    assert!(result[0] >= params.mz_start);
    assert!(result[result.len() - 1] <= params.mz_end * 1.01);

    let expected_ratio = 1.0 + (params.resolution_ppm / 1e6);
    verify_geometric_progression(&result, expected_ratio);
}

#[test]
fn test_ppm_index_small_range() {
    let params = small_range_params();
    let result = ppm_index(params.resolution_ppm, params.mz_start, params.mz_end);

    assert!(!result.is_empty());
    assert!(result[0] >= params.mz_start);
    // Allow slight overshoot due to geometric progression
    assert!(result[result.len() - 1] <= params.mz_end + 0.1);
}

#[test]
fn test_ppm_index_mz_start_clamping() {
    let params = clamping_test_params();
    let result = ppm_index(params.resolution_ppm, params.mz_start, params.mz_end);

    // Should be clamped to 50.0 even though mz_start is 10.0
    assert_eq!(result[0], 50.0);
}

#[test]
fn test_ppm_index_very_high_resolution() {
    let params = very_high_resolution_params();
    let result = ppm_index(params.resolution_ppm, params.mz_start, params.mz_end);

    assert!(!result.is_empty());
    assert!(result.len() > 1000); // Should have many points due to very small steps

    let expected_ratio = 1.0 + (params.resolution_ppm / 1e6);
    verify_geometric_progression(&result, expected_ratio);
}

#[test]
fn test_ppm_index_very_low_resolution() {
    let params = very_low_resolution_params();
    let result = ppm_index(params.resolution_ppm, params.mz_start, params.mz_end);

    assert!(!result.is_empty());
    assert!(result.len() > 1);

    let expected_ratio = 1.0 + (params.resolution_ppm / 1e6);
    verify_geometric_progression(&result, expected_ratio);
}

mod find_closest_index_tests {
    use super::*;

    #[test]
    fn exact_match_and_edges() {
        let mz_index = MZIndex::new();

        // Exact match
        let mz = mz_index.mz[100];
        assert_eq!(mz_index.find_closest_index(mz), 100);

        // First element
        assert_eq!(mz_index.find_closest_index(mz_index.mz[0]), 0);

        // Last element
        assert_eq!(
            mz_index.find_closest_index(mz_index.mz[mz_index.len() - 1]),
            mz_index.len() - 1
        );
    }

    #[test]
    fn out_of_range_values() {
        let mz_index = MZIndex::new();

        // Below range
        assert_eq!(mz_index.find_closest_index(0.0), 0);

        // Above range
        assert_eq!(mz_index.find_closest_index(3000.0), mz_index.len() - 1);
    }

    #[test]
    fn between_values() {
        let mz_index = MZIndex::new();
        let mz_between = (mz_index.mz[100] + mz_index.mz[101]) / 2.0;
        let closest = mz_index.find_closest_index(mz_between);
        assert!(closest == 100 || closest == 101);
    }
}

mod mz_range_indices_tests {
    use super::*;

    #[test]
    fn standard_ranges() {
        let mz_index = MZIndex::new();

        // Standard range
        let start = 100;
        let end = 105;
        let indices: Vec<usize> = mz_index
            .mz_range_indices(mz_index.mz[start], mz_index.mz[end])
            .collect();
        // Should include the end value now
        assert_eq!(indices, (start..=end).collect::<Vec<_>>());

        // Single element
        let idx = 100;
        let indices: Vec<usize> = mz_index
            .mz_range_indices(mz_index.mz[idx], mz_index.mz[idx])
            .collect();
        assert_eq!(indices, vec![idx]);
    }

    #[test]
    fn edge_cases() {
        let mz_index = MZIndex::new();

        // Below range
        let below_min = mz_index.mz[0] - 10.0;
        let indices: Vec<usize> = mz_index.mz_range_indices(below_min, below_min).collect();
        assert!(indices.is_empty());

        // Above range
        let max_mz = mz_index.mz[mz_index.len() - 1] + 10.0;
        let indices: Vec<usize> = mz_index.mz_range_indices(max_mz, max_mz + 10.0).collect();
        assert!(indices.is_empty());

        // Full range
        let below_min = mz_index.mz[0] - 10.0;
        let above_max = mz_index.mz[mz_index.len() - 1] + 10.0;
        let indices: Vec<usize> = mz_index.mz_range_indices(below_min, above_max).collect();
        assert_eq!(indices, (0..mz_index.len()).collect::<Vec<_>>());
    }

    #[test]
    fn partial_ranges() {
        let mz_index = MZIndex::new();

        // Lower out of bounds, upper in bounds
        let below_min = mz_index.mz[0] - 10.0;
        let target_idx = 50;
        let indices: Vec<usize> = mz_index
            .mz_range_indices(below_min, mz_index.mz[target_idx])
            .collect();
        assert_eq!(indices, (0..=target_idx).collect::<Vec<_>>());

        // Lower in bounds, upper out of bounds
        let start_idx = mz_index.len() - 50;
        let above_max = mz_index.mz[mz_index.len() - 1] + 10.0;
        let indices: Vec<usize> = mz_index
            .mz_range_indices(mz_index.mz[start_idx], above_max)
            .collect();
        assert_eq!(indices, (start_idx..mz_index.len()).collect::<Vec<_>>());
    }
}

#[test]
fn test_global_mz_index_singleton() {
    // Test that global() returns the same instance
    let index1 = MZIndex::global();
    let index2 = MZIndex::global();

    // Should be the same reference (same address)
    assert_eq!(index1 as *const _, index2 as *const _);

    // Should have the expected properties
    assert_eq!(index1.mz[0], MZ_START.max(50.0));
    assert!(index1.mz[index1.len() - 1] <= MZ_END * 1.01);
    assert!(!index1.mz.is_empty());

    // Test basic functionality works
    let closest = index1.find_closest_index(500.0);
    assert!(closest < index1.len());

    // Test range functionality works
    let range_indices: Vec<usize> = index1.mz_range_indices(400.0, 600.0).collect();
    assert!(!range_indices.is_empty());
}
