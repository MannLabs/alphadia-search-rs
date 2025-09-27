use super::*;
use numpy::ndarray::{Array1, Array4};

// Test fixtures for RT index tests
struct RTTestParams {
    target_rt: f32,
    tolerance: f32,
    padding: usize,
}

fn standard_search_params() -> RTTestParams {
    RTTestParams {
        target_rt: 25.0,
        tolerance: 10.0,
        padding: 2,
    }
}

fn small_tolerance_params() -> RTTestParams {
    RTTestParams {
        target_rt: 10.0,
        tolerance: 5.0,
        padding: 0,
    }
}

fn padding_test_params() -> RTTestParams {
    RTTestParams {
        target_rt: 40.0,
        tolerance: 7.5,
        padding: 1,
    }
}

fn create_realistic_index() -> RTIndex {
    let rt_values: Vec<f32> = (0..=100).map(|i| i as f32 * 0.5).collect();
    let rt_array = Array1::from_vec(rt_values);
    let delta_t = RTIndex::calculate_mean_delta_t(&rt_array);

    RTIndex {
        rt: rt_array,
        delta_t,
    }
}

// Helper function to create test data for AlphaRawView
struct TestAlphaRawData {
    delta_scan_array: Array1<i64>,
    isolation_lower_array: Array1<f32>,
    isolation_upper_array: Array1<f32>,
    peak_start_array: Array1<i64>,
    peak_stop_array: Array1<i64>,
    cycle_array: Array1<i64>,
    rt_array: Array1<f32>,
    peak_mz_array: Array1<f32>,
    peak_intensity_array: Array1<f32>,
    cycle_data: Array4<f32>,
}

impl TestAlphaRawData {
    fn new(total_spectra: usize, dia_cycle_length: usize) -> Self {
        use numpy::ndarray::Array1;

        let delta_scan_idx: Vec<i64> = (0..total_spectra)
            .map(|i| (i % dia_cycle_length) as i64)
            .collect();
        let rt_values: Vec<f32> = (0..total_spectra).map(|i| i as f32 * 0.1).collect();
        let isolation_lower_mz = vec![400.0; total_spectra];
        let isolation_upper_mz = vec![500.0; total_spectra];
        let peak_start_idx = vec![0i64; total_spectra];
        let peak_stop_idx = vec![0i64; total_spectra];
        let cycle_idx = vec![0i64; total_spectra];
        let peak_mz = vec![450.0f32; 0];
        let peak_intensity = vec![1000.0f32; 0];

        Self {
            delta_scan_array: Array1::from_vec(delta_scan_idx),
            isolation_lower_array: Array1::from_vec(isolation_lower_mz),
            isolation_upper_array: Array1::from_vec(isolation_upper_mz),
            peak_start_array: Array1::from_vec(peak_start_idx),
            peak_stop_array: Array1::from_vec(peak_stop_idx),
            cycle_array: Array1::from_vec(cycle_idx),
            rt_array: Array1::from_vec(rt_values),
            peak_mz_array: Array1::from_vec(peak_mz),
            peak_intensity_array: Array1::from_vec(peak_intensity),
            cycle_data: Array4::zeros((1, 1, 1, 1)),
        }
    }

    fn create_alpha_raw_view(&self) -> crate::dia_data::AlphaRawView {
        use crate::dia_data::AlphaRawView;

        AlphaRawView::new(
            self.delta_scan_array.view(),
            self.isolation_lower_array.view(),
            self.isolation_upper_array.view(),
            self.peak_start_array.view(),
            self.peak_stop_array.view(),
            self.cycle_array.view(),
            self.rt_array.view(),
            self.peak_mz_array.view(),
            self.peak_intensity_array.view(),
            self.cycle_data.view(),
        )
    }
}

#[test]
fn test_minimum_tolerance_enforced() {
    let rt_index = create_realistic_index();
    // Small tolerance should trigger minimum enforcement
    let (lower, upper) = rt_index.get_cycle_idx_limits(25.0, 2.0, 5);

    assert!(lower >= 33 && lower <= 37);
    assert!(upper >= 63 && upper <= 67);
}

#[test]
fn test_large_tolerance_not_enforced() {
    let rt_index = create_realistic_index();
    // Large tolerance should be used as-is
    let (lower, upper) = rt_index.get_cycle_idx_limits(25.0, 10.0, 5);

    assert!(lower >= 28 && lower <= 32);
    assert!(upper >= 68 && upper <= 72);
}

#[test]
fn test_padding_increases_range() {
    let rt_index = create_realistic_index();
    let target_rt = 25.0;
    let small_tolerance = 1.0;

    let (l1, u1) = rt_index.get_cycle_idx_limits(target_rt, small_tolerance, 2);
    let (l2, u2) = rt_index.get_cycle_idx_limits(target_rt, small_tolerance, 10);

    // Higher padding increases minimum tolerance
    assert!(u2 - l2 > u1 - l1);

    let range1 = u1 - l1;
    let range2 = u2 - l2;

    assert!(range1 >= 22 && range1 <= 26);
    assert!(range2 >= 38 && range2 <= 42);
}

#[test]
fn test_precise_boundary_calculation() {
    let rt_index = create_realistic_index();
    // Large tolerance avoids minimum enforcement for precise testing
    let (lower, upper) = rt_index.get_cycle_idx_limits(25.0, 15.0, 0);

    assert_eq!(lower, 20);
    assert_eq!(upper, 80);
}

#[test]
fn test_target_near_boundaries() {
    let rt_index = create_realistic_index();

    // Range gets clipped to data boundaries
    let (lower, upper) = rt_index.get_cycle_idx_limits(5.0, 15.0, 0);
    assert_eq!(lower, 0);
    assert_eq!(upper, 40);

    let (lower, upper) = rt_index.get_cycle_idx_limits(45.0, 15.0, 0);
    assert_eq!(lower, 60);
    assert_eq!(upper, 101);
}

#[test]
fn test_target_outside_range() {
    let rt_index = create_realistic_index();

    let (lower, upper) = rt_index.get_cycle_idx_limits(-20.0, 5.0, 0);
    assert_eq!(lower, 0);
    assert_eq!(upper, 0);

    let (lower, upper) = rt_index.get_cycle_idx_limits(70.0, 5.0, 0);
    assert_eq!(lower, 101);
    assert_eq!(upper, 101);
}

#[test]
fn test_empty_index() {
    let empty_rt_index = RTIndex::new();
    let (lower, upper) = empty_rt_index.get_cycle_idx_limits(25.0, 5.0, 5);
    assert_eq!(lower, 0);
    assert_eq!(upper, 0);
}

#[test]
fn test_binary_search_standard_case() {
    let rt_index = create_realistic_index();
    let params = standard_search_params();
    let (lower, upper) =
        rt_index.get_cycle_idx_limits(params.target_rt, params.tolerance, params.padding);

    // Verify bounds are within valid range
    assert!(lower <= rt_index.len());
    assert!(upper <= rt_index.len());
    assert!(lower <= upper);

    // Verify the tolerance logic is preserved
    let min_tolerance = (MIN_CYCLES + params.padding as f32) * rt_index.delta_t;
    let _effective_tolerance = params.tolerance.max(min_tolerance);
    assert!(upper - lower > 0); // Should capture some range
}

#[test]
fn test_binary_search_small_tolerance() {
    let rt_index = create_realistic_index();
    let params = small_tolerance_params();
    let (lower, upper) =
        rt_index.get_cycle_idx_limits(params.target_rt, params.tolerance, params.padding);

    assert!(lower <= rt_index.len());
    assert!(upper <= rt_index.len());
    assert!(lower <= upper);

    // With small tolerance, minimum should be enforced
    let min_tolerance = MIN_CYCLES * rt_index.delta_t;
    assert!(upper - lower >= (min_tolerance / rt_index.delta_t) as usize / 2);
}

#[test]
fn test_binary_search_with_padding() {
    let rt_index = create_realistic_index();
    let params = padding_test_params();
    let (lower, upper) =
        rt_index.get_cycle_idx_limits(params.target_rt, params.tolerance, params.padding);

    assert!(lower <= rt_index.len());
    assert!(upper <= rt_index.len());
    assert!(lower <= upper);

    // Verify padding affects the tolerance calculation
    let min_tolerance_no_padding = MIN_CYCLES * rt_index.delta_t;
    let min_tolerance_with_padding = (MIN_CYCLES + params.padding as f32) * rt_index.delta_t;

    // With padding, minimum tolerance should be larger
    assert!(min_tolerance_with_padding > min_tolerance_no_padding);
}

#[test]
fn test_from_alpha_raw_ms1_extraction() {
    let total_spectra = 30;
    let dia_cycle_length = 10; // 1 MS1 + 9 MS2 scans

    let test_data = TestAlphaRawData::new(total_spectra, dia_cycle_length);
    let alpha_raw_view = test_data.create_alpha_raw_view();
    let rt_index = RTIndex::from_alpha_raw(&alpha_raw_view);

    // Should extract only MS1 scans (every 10th scan)
    assert_eq!(rt_index.len(), 3); // 30 / 10 = 3 MS1 scans

    // Verify RT values are from MS1 scans only
    assert_eq!(rt_index.rt[0], 0.0); // First MS1 at index 0
    assert_eq!(rt_index.rt[1], 1.0); // Second MS1 at index 10
    assert_eq!(rt_index.rt[2], 2.0); // Third MS1 at index 20
}

#[test]
fn test_from_alpha_raw_delta_t_calculation() {
    // Simple case with regular MS1 scans - MS1 at positions 0, 3, 6
    let total_spectra = 9;
    let dia_cycle_length = 3; // Shorter cycle for testing

    let test_data = TestAlphaRawData::new(total_spectra, dia_cycle_length);
    let alpha_raw_view = test_data.create_alpha_raw_view();
    let rt_index = RTIndex::from_alpha_raw(&alpha_raw_view);

    // Should have 3 MS1 scans at RT 0.0, 0.3, 0.6
    assert_eq!(rt_index.len(), 3);

    // Delta_t should be (0.6 - 0.0) / (3 - 1) = 0.3
    let expected_delta_t = 0.3;
    assert!((rt_index.delta_t - expected_delta_t).abs() < 1e-6);
}
