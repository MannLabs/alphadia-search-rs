use super::*;
use crate::dia_data::AlphaRawView;
use numpy::ndarray::{ArrayView1, ArrayView4};
use numpy::{PyArrayMethods, PyUntypedArrayMethods};
use pyo3::Python;

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
fn test_dia_data_creation() {
    let dia_data = DIAData::new();
    assert_eq!(dia_data.num_observations(), 0);
    assert!(MZIndex::global().len() > 0); // Should have some default mz index
}

#[test]
fn test_default_implementation() {
    let dia_data = DIAData::default();
    assert_eq!(dia_data.num_observations(), 0);
    assert!(MZIndex::global().len() > 0);
    assert!(dia_data.rt_index.rt.len() == 0); // RT index should be empty for new instance
}

#[test]
fn test_from_alpha_raw_view() {
    // Test data with 2 observations
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

    let cycle_data = [1.0f32];
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

    // Test that we can build DIAData from AlphaRawView directly
    use crate::dia_data_builder::DIADataBuilder;
    let dia_data = DIADataBuilder::from_alpha_raw(&alpha_raw_view);

    assert_eq!(dia_data.num_observations(), 2);
    assert!(MZIndex::global().len() > 0);
    assert!(dia_data.rt_index.rt.len() > 0);
}

#[test]
fn test_get_valid_observations() {
    // Test data with 3 observations with different isolation windows
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

    let cycle_data = [2.0f32];
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

    use crate::dia_data_builder::DIADataBuilder;
    let dia_data = DIADataBuilder::from_alpha_raw(&alpha_raw_view);

    // Test valid observations for different m/z values
    let valid_for_112 = dia_data.get_valid_observations(112.0);
    assert_eq!(valid_for_112, vec![0]); // Should match observation 0 (100-125)

    let valid_for_212 = dia_data.get_valid_observations(212.0);
    assert_eq!(valid_for_212, vec![1]); // Should match observation 1 (200-225)

    let valid_for_312 = dia_data.get_valid_observations(312.0);
    assert_eq!(valid_for_312, vec![2]); // Should match observation 2 (300-325)

    // Test edge cases
    let valid_for_100 = dia_data.get_valid_observations(100.0);
    assert_eq!(valid_for_100, vec![0]); // Lower boundary inclusive

    let valid_for_125 = dia_data.get_valid_observations(125.0);
    assert_eq!(valid_for_125, vec![0]); // Upper boundary inclusive

    // Test no matches
    let valid_for_50 = dia_data.get_valid_observations(50.0);
    assert_eq!(valid_for_50, Vec::<usize>::new()); // Should match no observations

    let valid_for_175 = dia_data.get_valid_observations(175.0);
    assert_eq!(valid_for_175, Vec::<usize>::new()); // Between windows
}

#[test]
fn test_observation_consistency() {
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

    let cycle_data = [3.0f32];
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

    use crate::dia_data_builder::DIADataBuilder;
    let dia_data = DIADataBuilder::from_alpha_raw(&alpha_raw_view);

    // Verify the number of observations matches expected
    assert_eq!(dia_data.num_observations(), 3);

    // Verify isolation windows are correctly set
    assert_eq!(
        dia_data.quadrupole_observations[0].isolation_window[0],
        100.0
    );
    assert_eq!(
        dia_data.quadrupole_observations[0].isolation_window[1],
        125.0
    );

    assert_eq!(
        dia_data.quadrupole_observations[1].isolation_window[0],
        200.0
    );
    assert_eq!(
        dia_data.quadrupole_observations[1].isolation_window[1],
        225.0
    );

    assert_eq!(
        dia_data.quadrupole_observations[2].isolation_window[0],
        300.0
    );
    assert_eq!(
        dia_data.quadrupole_observations[2].isolation_window[1],
        325.0
    );
}

#[test]
fn test_empty_data_handling() {
    let dia_data = DIAData::new();

    // Test empty data behavior
    assert_eq!(dia_data.num_observations(), 0);
    assert_eq!(dia_data.get_valid_observations(100.0), Vec::<usize>::new());
}

#[test]
fn test_has_mobility() {
    let dia_data = DIAData::new();
    assert_eq!(dia_data.has_mobility(), false);
}

#[test]
fn test_has_ms1() {
    let dia_data = DIAData::new();
    assert_eq!(dia_data.has_ms1(), false);
}

#[test]
fn test_mobility_values() {
    let dia_data = DIAData::new();
    let expected_mobility_values = vec![1e-6, 0.0];
    assert_eq!(dia_data.mobility_values(), expected_mobility_values);
}

#[test]
fn test_rt_values() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let dia_data = DIAData::new();
        let rt_values = dia_data.rt_values(py);
        // New DIAData should have empty RT values
        assert_eq!(rt_values.len(), 0);
    });
}

#[test]
fn test_rt_values_with_data() {
    // Create DIAData with some data to test RT values extraction
    let spectrum_delta_scan_idx = [0i64, 0];
    let isolation_lower_mz = [100.0f32, 100.0];
    let isolation_upper_mz = [125.0f32, 125.0];
    let spectrum_peak_start_idx = [0i64, 2];
    let spectrum_peak_stop_idx = [2i64, 4];
    let spectrum_cycle_idx = [0i64, 1];
    let spectrum_rt = [1.0f32, 1.1];
    let peak_mz = [110.0f32, 115.0, 111.0, 116.0];
    let peak_intensity = [1000.0f32, 1100.0, 1200.0, 1300.0];

    let cycle_data = [4.0f32];
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

    use crate::dia_data_builder::DIADataBuilder;
    let dia_data = DIADataBuilder::from_alpha_raw(&alpha_raw_view);

    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let rt_values = dia_data.rt_values(py);
        let rt_vec: Vec<f32> = rt_values.to_vec().unwrap();
        assert_eq!(rt_vec, vec![1.0f32, 1.1]);
    });
}

#[test]
fn test_cycle() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let dia_data = DIAData::new();
        let cycle_array = dia_data.cycle(py);
        // Should return a PyArray4 with shape (0, 0, 0, 0)
        assert_eq!(cycle_array.shape(), [0, 0, 0, 0]);
    });
}
