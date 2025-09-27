use super::*;
use crate::dia_data::AlphaRawView;
use numpy::ndarray::{ArrayView1, ArrayView4};

fn create_mock_alpha_raw_view<'a>(
    spectrum_delta_scan_idx: &'a [i64],
    isolation_lower_mz: &'a [f32],
    isolation_upper_mz: &'a [f32],
    spectrum_peak_start_idx: &'a [i64],
    spectrum_peak_stop_idx: &'a [i64],
    spectrum_cycle_idx: &'a [i64],
    spectrum_rt: &'a [f32],
    peak_mz: &'a [f32],
    peak_intensity: &'a [f32],
    cycle: &'a [f32],
) -> AlphaRawView<'a> {
    AlphaRawView {
        spectrum_delta_scan_idx: ArrayView1::from(spectrum_delta_scan_idx),
        isolation_lower_mz: ArrayView1::from(isolation_lower_mz),
        isolation_upper_mz: ArrayView1::from(isolation_upper_mz),
        spectrum_peak_start_idx: ArrayView1::from(spectrum_peak_start_idx),
        spectrum_peak_stop_idx: ArrayView1::from(spectrum_peak_stop_idx),
        spectrum_cycle_idx: ArrayView1::from(spectrum_cycle_idx),
        spectrum_rt: ArrayView1::from(spectrum_rt),
        peak_mz: ArrayView1::from(peak_mz),
        peak_intensity: ArrayView1::from(peak_intensity),
        cycle: ArrayView4::from_shape([1, 1, 1, 1], cycle).unwrap(),
    }
}

#[test]
fn test_optimized_builder_basic_functionality() {
    // Test data with 3 observations
    let spectrum_delta_scan_idx = [0i64, 0, 1, 1, 2, 2];
    let isolation_lower_mz = [100.0f32, 100.0, 200.0, 200.0, 300.0, 300.0];
    let isolation_upper_mz = [125.0f32, 125.0, 225.0, 225.0, 325.0, 325.0];
    let spectrum_peak_start_idx = [0i64, 2, 4, 6, 8, 10];
    let spectrum_peak_stop_idx = [2i64, 4, 6, 8, 10, 12];
    let spectrum_cycle_idx = [0i64, 1, 0, 1, 0, 1];
    let spectrum_rt = [1.0f32, 1.1, 2.0, 2.1, 3.0, 3.1];

    // Peak data - 2 peaks per spectrum
    let peak_mz = [
        110.0f32, 115.0, // spectrum 0
        111.0, 116.0, // spectrum 1
        210.0, 215.0, // spectrum 2
        211.0, 216.0, // spectrum 3
        310.0, 315.0, // spectrum 4
        311.0, 316.0, // spectrum 5
    ];
    let peak_intensity = [
        1000.0f32, 1100.0, // spectrum 0
        1200.0, 1300.0, // spectrum 1
        2000.0, 2100.0, // spectrum 2
        2200.0, 2300.0, // spectrum 3
        3000.0, 3100.0, // spectrum 4
        3200.0, 3300.0, // spectrum 5
    ];

    let cycle_data = [8.0f32];
    let alpha_raw_view = create_mock_alpha_raw_view(
        &spectrum_delta_scan_idx,
        &isolation_lower_mz,
        &isolation_upper_mz,
        &spectrum_peak_start_idx,
        &spectrum_peak_stop_idx,
        &spectrum_cycle_idx,
        &spectrum_rt,
        &peak_mz,
        &peak_intensity,
        &cycle_data,
    );

    let dia_data = DIADataBuilder::from_alpha_raw(&alpha_raw_view);

    // Should have 3 observations (delta_scan_idx 0, 1, 2)
    assert_eq!(dia_data.num_observations(), 3);

    // Check that indices were built
    assert!(MZIndex::global().len() > 0);
    assert!(dia_data.rt_index.rt.len() > 0);
}

#[test]
fn test_observation_isolation_windows() {
    let spectrum_delta_scan_idx = [0i64, 0, 1, 1, 2, 2];
    let isolation_lower_mz = [100.0f32, 100.0, 200.0, 200.0, 300.0, 300.0];
    let isolation_upper_mz = [125.0f32, 125.0, 225.0, 225.0, 325.0, 325.0];
    let spectrum_peak_start_idx = [0i64, 2, 4, 6, 8, 10];
    let spectrum_peak_stop_idx = [2i64, 4, 6, 8, 10, 12];
    let spectrum_cycle_idx = [0i64, 1, 0, 1, 0, 1];
    let spectrum_rt = [1.0f32, 1.1, 2.0, 2.1, 3.0, 3.1];
    let peak_mz = [
        110.0f32, 115.0, 111.0, 116.0, 210.0, 215.0, 211.0, 216.0, 310.0, 315.0, 311.0, 316.0,
    ];
    let peak_intensity = [
        1000.0f32, 1100.0, 1200.0, 1300.0, 2000.0, 2100.0, 2200.0, 2300.0, 3000.0, 3100.0, 3200.0,
        3300.0,
    ];

    let cycle_data = [8.0f32];
    let alpha_raw_view = create_mock_alpha_raw_view(
        &spectrum_delta_scan_idx,
        &isolation_lower_mz,
        &isolation_upper_mz,
        &spectrum_peak_start_idx,
        &spectrum_peak_stop_idx,
        &spectrum_cycle_idx,
        &spectrum_rt,
        &peak_mz,
        &peak_intensity,
        &cycle_data,
    );

    let dia_data = DIADataBuilder::from_alpha_raw(&alpha_raw_view);

    // Test isolation windows for each observation
    let obs0 = &dia_data.quadrupole_observations[0];
    assert_eq!(obs0.isolation_window[0], 100.0);
    assert_eq!(obs0.isolation_window[1], 125.0);

    let obs1 = &dia_data.quadrupole_observations[1];
    assert_eq!(obs1.isolation_window[0], 200.0);
    assert_eq!(obs1.isolation_window[1], 225.0);

    let obs2 = &dia_data.quadrupole_observations[2];
    assert_eq!(obs2.isolation_window[0], 300.0);
    assert_eq!(obs2.isolation_window[1], 325.0);
}

#[test]
fn test_valid_observations() {
    let spectrum_delta_scan_idx = [0i64, 0, 1, 1, 2, 2];
    let isolation_lower_mz = [100.0f32, 100.0, 200.0, 200.0, 300.0, 300.0];
    let isolation_upper_mz = [125.0f32, 125.0, 225.0, 225.0, 325.0, 325.0];
    let spectrum_peak_start_idx = [0i64, 2, 4, 6, 8, 10];
    let spectrum_peak_stop_idx = [2i64, 4, 6, 8, 10, 12];
    let spectrum_cycle_idx = [0i64, 1, 0, 1, 0, 1];
    let spectrum_rt = [1.0f32, 1.1, 2.0, 2.1, 3.0, 3.1];
    let peak_mz = [
        110.0f32, 115.0, 111.0, 116.0, 210.0, 215.0, 211.0, 216.0, 310.0, 315.0, 311.0, 316.0,
    ];
    let peak_intensity = [
        1000.0f32, 1100.0, 1200.0, 1300.0, 2000.0, 2100.0, 2200.0, 2300.0, 3000.0, 3100.0, 3200.0,
        3300.0,
    ];

    let cycle_data = [8.0f32];
    let alpha_raw_view = create_mock_alpha_raw_view(
        &spectrum_delta_scan_idx,
        &isolation_lower_mz,
        &isolation_upper_mz,
        &spectrum_peak_start_idx,
        &spectrum_peak_stop_idx,
        &spectrum_cycle_idx,
        &spectrum_rt,
        &peak_mz,
        &peak_intensity,
        &cycle_data,
    );

    let dia_data = DIADataBuilder::from_alpha_raw(&alpha_raw_view);

    // Test precursor matching
    let valid_for_112 = dia_data.get_valid_observations(112.0);
    assert_eq!(valid_for_112, vec![0]); // Should match observation 0 (100-125)

    let valid_for_212 = dia_data.get_valid_observations(212.0);
    assert_eq!(valid_for_212, vec![1]); // Should match observation 1 (200-225)

    let valid_for_312 = dia_data.get_valid_observations(312.0);
    assert_eq!(valid_for_312, vec![2]); // Should match observation 2 (300-325)

    let valid_for_50 = dia_data.get_valid_observations(50.0);
    assert_eq!(valid_for_50, Vec::<usize>::new()); // Should match no observations
}

#[test]
fn test_parallel_building_deterministic() {
    let spectrum_delta_scan_idx = [0i64, 0, 1, 1];
    let isolation_lower_mz = [100.0f32, 100.0, 200.0, 200.0];
    let isolation_upper_mz = [125.0f32, 125.0, 225.0, 225.0];
    let spectrum_peak_start_idx = [0i64, 2, 4, 6];
    let spectrum_peak_stop_idx = [2i64, 4, 6, 8];
    let spectrum_cycle_idx = [0i64, 1, 0, 1];
    let spectrum_rt = [1.0f32, 1.1, 2.0, 2.1];
    let peak_mz = [110.0f32, 115.0, 111.0, 116.0, 210.0, 215.0, 211.0, 216.0];
    let peak_intensity = [
        1000.0f32, 1100.0, 1200.0, 1300.0, 2000.0, 2100.0, 2200.0, 2300.0,
    ];

    let cycle_data = [8.0f32];
    let alpha_raw_view = create_mock_alpha_raw_view(
        &spectrum_delta_scan_idx,
        &isolation_lower_mz,
        &isolation_upper_mz,
        &spectrum_peak_start_idx,
        &spectrum_peak_stop_idx,
        &spectrum_cycle_idx,
        &spectrum_rt,
        &peak_mz,
        &peak_intensity,
        &cycle_data,
    );

    // Build multiple times to ensure deterministic results
    let dia_data1 = DIADataBuilder::from_alpha_raw(&alpha_raw_view);
    let dia_data2 = DIADataBuilder::from_alpha_raw(&alpha_raw_view);

    // Results should be identical
    assert_eq!(dia_data1.num_observations(), dia_data2.num_observations());

    for i in 0..dia_data1.num_observations() {
        let obs1 = &dia_data1.quadrupole_observations[i];
        let obs2 = &dia_data2.quadrupole_observations[i];

        assert_eq!(obs1.isolation_window, obs2.isolation_window);
    }
}

#[test]
fn test_cycle_ordering_preservation() {
    // Test data specifically designed to test cycle ordering within mz_idx
    // Uses non-sequential cycles to ensure ordering is preserved
    let spectrum_delta_scan_idx = [0i64, 0, 0];
    let isolation_lower_mz = [100.0f32, 100.0, 100.0];
    let isolation_upper_mz = [150.0f32, 150.0, 150.0];
    let spectrum_peak_start_idx = [0i64, 1, 2];
    let spectrum_peak_stop_idx = [1i64, 2, 3];
    let spectrum_cycle_idx = [2i64, 0, 1]; // Non-sequential: 2, 0, 1
    let spectrum_rt = [1.0f32, 1.1, 1.2];

    // All peaks have same m/z to test ordering within same mz_idx
    let peak_mz = [120.0f32, 120.0, 120.0];
    let peak_intensity = [1000.0f32, 2000.0, 3000.0];

    let cycle_data = [8.0f32];
    let alpha_raw_view = create_mock_alpha_raw_view(
        &spectrum_delta_scan_idx,
        &isolation_lower_mz,
        &isolation_upper_mz,
        &spectrum_peak_start_idx,
        &spectrum_peak_stop_idx,
        &spectrum_cycle_idx,
        &spectrum_rt,
        &peak_mz,
        &peak_intensity,
        &cycle_data,
    );

    let dia_data = DIADataBuilder::from_alpha_raw(&alpha_raw_view);

    // Find the mz_idx for our test m/z
    let mz_idx = MZIndex::global().find_closest_index(120.0);
    let obs = &dia_data.quadrupole_observations[0];
    let (cycles, intensities) = obs.get_slice_data(mz_idx);

    // Cycles should be in temporal order: [0, 1, 2] not [2, 0, 1]
    // This verifies the sort_by_key fix is working
    assert_eq!(cycles.len(), 3);
    assert_eq!(cycles[0], 0); // First cycle chronologically
    assert_eq!(cycles[1], 1); // Second cycle chronologically
    assert_eq!(cycles[2], 2); // Third cycle chronologically

    // Intensities should follow the cycle ordering
    assert_eq!(intensities[0], 2000.0); // Intensity from cycle 0
    assert_eq!(intensities[1], 3000.0); // Intensity from cycle 1
    assert_eq!(intensities[2], 1000.0); // Intensity from cycle 2
}
