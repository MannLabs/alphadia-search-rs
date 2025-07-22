use super::*;
use numpy::ndarray::ArrayView1;

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
    }
}

#[test]
fn test_build_quadrupole_observation() {
    // Create a simple mz_index
    let mz_index = MZIndex::new();

    // Prepare test data
    let spectrum_delta_scan_idx = [0, 0, 1, 1];
    let isolation_lower_mz = [100.0, 100.0, 200.0, 200.0];
    let isolation_upper_mz = [150.0, 150.0, 250.0, 250.0];
    let spectrum_peak_start_idx = [0, 2, 4, 6];
    let spectrum_peak_stop_idx = [2, 4, 6, 8];
    let spectrum_cycle_idx = [0, 1, 0, 1];
    let spectrum_rt = [1.0, 2.0, 3.0, 4.0];
    let peak_mz = [120.0, 130.0, 140.0, 145.0, 210.0, 220.0, 230.0, 240.0];
    let peak_intensity = [
        1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0,
    ];

    // Create mock AlphaRawView
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
    );

    // Build a quadrupole observation for delta_scan_idx = 0
    let quad_obs = DIADataBuilder::build_quadrupole_observation(&alpha_raw_view, 0, &mz_index);

    // Validate the quadrupole observation
    assert_eq!(quad_obs.isolation_window, [100.0, 150.0]);
    assert_eq!(quad_obs.num_cycles, 2);

    // Build another quadrupole observation for delta_scan_idx = 1
    let quad_obs = DIADataBuilder::build_quadrupole_observation(&alpha_raw_view, 1, &mz_index);

    // Validate the quadrupole observation
    assert_eq!(quad_obs.isolation_window, [200.0, 250.0]);
    assert_eq!(quad_obs.num_cycles, 2);
}

#[test]
fn test_from_alpha_raw() {
    // Prepare test data
    let spectrum_delta_scan_idx = [0, 0, 1, 1];
    let isolation_lower_mz = [100.0, 100.0, 200.0, 200.0];
    let isolation_upper_mz = [150.0, 150.0, 250.0, 250.0];
    let spectrum_peak_start_idx = [0, 2, 4, 6];
    let spectrum_peak_stop_idx = [2, 4, 6, 8];
    let spectrum_cycle_idx = [0, 1, 0, 1];
    let spectrum_rt = [1.0, 2.0, 3.0, 4.0];
    let peak_mz = [120.0, 130.0, 140.0, 145.0, 210.0, 220.0, 230.0, 240.0];
    let peak_intensity = [
        1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0,
    ];

    // Create mock AlphaRawView
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
    );

    // Build DIA data
    let dia_data = DIADataBuilder::from_alpha_raw(&alpha_raw_view);

    // Validate the DIA data
    assert_eq!(dia_data.num_observations(), 2);
    assert_eq!(
        dia_data.quadrupole_observations[0].isolation_window,
        [100.0, 150.0]
    );
    assert_eq!(
        dia_data.quadrupole_observations[1].isolation_window,
        [200.0, 250.0]
    );

    // Test get_valid_observations
    let valid_obs = dia_data.get_valid_observations(125.0);
    assert_eq!(valid_obs, vec![0]);

    let valid_obs = dia_data.get_valid_observations(225.0);
    assert_eq!(valid_obs, vec![1]);

    // No valid observations for mz outside any isolation window
    let valid_obs = dia_data.get_valid_observations(175.0);
    assert_eq!(valid_obs, Vec::<usize>::new());
}
