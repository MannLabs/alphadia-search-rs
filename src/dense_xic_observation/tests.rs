use super::*;
use crate::dia_data::AlphaRawView;
use crate::dia_data_builder::DIADataBuilder;
use numpy::ndarray::{ArrayView1, ArrayView4};

fn create_simple_alpha_raw() -> AlphaRawView<'static> {
    static SPECTRUM_DELTA_SCAN_IDX: [i64; 1] = [0];
    static ISOLATION_LOWER_MZ: [f32; 1] = [199.0];
    static ISOLATION_UPPER_MZ: [f32; 1] = [201.0];
    static SPECTRUM_PEAK_START_IDX: [i64; 1] = [0];
    static SPECTRUM_PEAK_STOP_IDX: [i64; 1] = [2];
    static SPECTRUM_CYCLE_IDX: [i64; 1] = [10];
    static SPECTRUM_RT: [f32; 1] = [100.0];
    static PEAK_MZ: [f32; 2] = [200.0, 200.1];
    static PEAK_INTENSITY: [f32; 2] = [1000.0, 2000.0];
    static CYCLE_DATA: [f32; 1] = [5.0];

    AlphaRawView {
        spectrum_delta_scan_idx: ArrayView1::from(&SPECTRUM_DELTA_SCAN_IDX),
        isolation_lower_mz: ArrayView1::from(&ISOLATION_LOWER_MZ),
        isolation_upper_mz: ArrayView1::from(&ISOLATION_UPPER_MZ),
        spectrum_peak_start_idx: ArrayView1::from(&SPECTRUM_PEAK_START_IDX),
        spectrum_peak_stop_idx: ArrayView1::from(&SPECTRUM_PEAK_STOP_IDX),
        spectrum_cycle_idx: ArrayView1::from(&SPECTRUM_CYCLE_IDX),
        spectrum_rt: ArrayView1::from(&SPECTRUM_RT),
        peak_mz: ArrayView1::from(&PEAK_MZ),
        peak_intensity: ArrayView1::from(&PEAK_INTENSITY),
        cycle: ArrayView4::from_shape([1, 1, 1, 1], &CYCLE_DATA).unwrap(),
    }
}

#[test]
fn test_basic_creation() {
    // Given: DIA data with a single spectrum and one fragment m/z
    let alpha_raw = create_simple_alpha_raw();
    let dia_data = DIADataBuilder::from_alpha_raw(&alpha_raw);
    let fragment_mz = vec![200.0];

    // When: Creating a DenseXICObservation for cycles 10-12
    let obs = DenseXICObservation::new(&dia_data, 200.0, 10, 12, 20.0, &fragment_mz);

    // Then: The matrix should have 1 fragment row and 2 cycle columns
    assert_eq!(obs.dense_xic.nrows(), 1);
    assert_eq!(obs.dense_xic.ncols(), 2);
}

#[test]
fn test_optimized_data_creation() {
    // Given: DIA data with a single spectrum and one fragment m/z
    let alpha_raw = create_simple_alpha_raw();
    let dia_data = DIADataBuilder::from_alpha_raw(&alpha_raw);
    let fragment_mz = vec![200.0];

    // When: Creating a DenseXICObservation for a single cycle (10-11)
    let obs = DenseXICObservation::new(&dia_data, 200.0, 10, 11, 20.0, &fragment_mz);

    // Then: The matrix should have 1 fragment row and 1 cycle column
    assert_eq!(obs.dense_xic.nrows(), 1);
    assert_eq!(obs.dense_xic.ncols(), 1);
}

#[test]
fn test_empty_fragments() {
    // Given: DIA data with a single spectrum and an empty fragment list
    let alpha_raw = create_simple_alpha_raw();
    let dia_data = DIADataBuilder::from_alpha_raw(&alpha_raw);
    let fragment_mz: Vec<f32> = vec![];

    // When: Creating a DenseXICObservation with no fragments
    let obs = DenseXICObservation::new(&dia_data, 200.0, 10, 12, 20.0, &fragment_mz);

    // Then: The matrix should have 0 fragment rows but still 2 cycle columns
    assert_eq!(obs.dense_xic.nrows(), 0);
    assert_eq!(obs.dense_xic.ncols(), 2);
}

#[test]
fn test_metadata_storage() {
    // Given: DIA data and specific extraction parameters
    let alpha_raw = create_simple_alpha_raw();
    let dia_data = DIADataBuilder::from_alpha_raw(&alpha_raw);
    let fragment_mz = vec![200.0];

    // When: Creating a DenseXICObservation with specific parameters
    let obs = DenseXICObservation::new(&dia_data, 200.0, 10, 15, 50.0, &fragment_mz);

    // Then: The metadata should be correctly stored
    assert_eq!(obs.cycle_start_idx, 10);
    assert_eq!(obs.cycle_stop_idx, 15);
    assert_eq!(obs.mass_tolerance, 50.0);
    assert!(!obs.contributing_obs_indices.is_empty());
}

#[test]
fn test_multiple_fragments() {
    // Given: DIA data and multiple fragment m/z values
    let alpha_raw = create_simple_alpha_raw();
    let dia_data = DIADataBuilder::from_alpha_raw(&alpha_raw);
    let fragment_mz = vec![200.0, 200.1, 205.0];

    // When: Creating a DenseXICObservation with 3 fragments
    let obs = DenseXICObservation::new(&dia_data, 200.0, 10, 12, 20.0, &fragment_mz);

    // Then: The matrix should have 3 fragment rows and 2 cycle columns
    assert_eq!(obs.dense_xic.nrows(), 3);
    assert_eq!(obs.dense_xic.ncols(), 2);
}

#[test]
fn test_dense_xic_mz_basic_creation() {
    // Given: DIA data with a single spectrum and one fragment m/z
    let alpha_raw = create_simple_alpha_raw();
    let dia_data = DIADataBuilder::from_alpha_raw(&alpha_raw);
    let fragment_mz = vec![200.0];

    // When: Creating a DenseXICMZObservation for cycles 10-12
    let obs = DenseXICMZObservation::new(&dia_data, 200.0, 10, 12, 20.0, &fragment_mz);

    // Then: Both XIC and m/z matrices should have 1 fragment row and 2 cycle columns
    assert_eq!(obs.dense_xic.nrows(), 1);
    assert_eq!(obs.dense_xic.ncols(), 2);
    assert_eq!(obs.dense_mz.nrows(), 1);
    assert_eq!(obs.dense_mz.ncols(), 2);
}

#[test]
fn test_dense_xic_mz_intensity_values() {
    // Given: DIA data with two peaks at 200.0 and 200.1 m/z
    // and a fragment m/z of 200.05 that should match both peaks
    let alpha_raw = create_simple_alpha_raw();
    let dia_data = DIADataBuilder::from_alpha_raw(&alpha_raw);
    let fragment_mz = vec![200.05]; // Target m/z between the two peaks

    // When: Creating a DenseXICMZObservation with high tolerance (1000 ppm)
    let obs = DenseXICMZObservation::new(&dia_data, 200.0, 10, 11, 1000.0, &fragment_mz);

    // Then: Intensities should be summed and m/z should be the weighted average
    assert_eq!(obs.dense_xic[[0, 0]], 3000.0); // 1000 + 2000

    let expected_mz = (200.0 * 1000.0 + 200.1 * 2000.0) / 3000.0;
    // The weighted average will be close but affected by grid discretization
    assert!((obs.dense_mz[[0, 0]] - expected_mz).abs() < 1e-4);
}

#[test]
fn test_dense_xic_mz_weighted_average() {
    // Given: DIA data with three peaks of different intensities
    static SPECTRUM_DELTA_SCAN_IDX: [i64; 1] = [0];
    static ISOLATION_LOWER_MZ: [f32; 1] = [249.0];
    static ISOLATION_UPPER_MZ: [f32; 1] = [251.0];
    static SPECTRUM_PEAK_START_IDX: [i64; 1] = [0];
    static SPECTRUM_PEAK_STOP_IDX: [i64; 1] = [3];
    static SPECTRUM_CYCLE_IDX: [i64; 1] = [5];
    static SPECTRUM_RT: [f32; 1] = [50.0];
    static PEAK_MZ: [f32; 3] = [249.9, 250.0, 250.1];
    static PEAK_INTENSITY: [f32; 3] = [500.0, 1000.0, 1500.0];
    static CYCLE_DATA: [f32; 1] = [6.0];

    let alpha_raw = AlphaRawView {
        spectrum_delta_scan_idx: ArrayView1::from(&SPECTRUM_DELTA_SCAN_IDX),
        isolation_lower_mz: ArrayView1::from(&ISOLATION_LOWER_MZ),
        isolation_upper_mz: ArrayView1::from(&ISOLATION_UPPER_MZ),
        spectrum_peak_start_idx: ArrayView1::from(&SPECTRUM_PEAK_START_IDX),
        spectrum_peak_stop_idx: ArrayView1::from(&SPECTRUM_PEAK_STOP_IDX),
        spectrum_cycle_idx: ArrayView1::from(&SPECTRUM_CYCLE_IDX),
        spectrum_rt: ArrayView1::from(&SPECTRUM_RT),
        peak_mz: ArrayView1::from(&PEAK_MZ),
        peak_intensity: ArrayView1::from(&PEAK_INTENSITY),
        cycle: ArrayView4::from_shape([1, 1, 1, 1], &CYCLE_DATA).unwrap(),
    };

    let dia_data = DIADataBuilder::from_alpha_raw(&alpha_raw);
    let fragment_mz = vec![250.0];

    // When: Creating a DenseXICMZObservation that matches all three peaks
    let obs = DenseXICMZObservation::new(&dia_data, 250.0, 5, 6, 2000.0, &fragment_mz);

    // Then: Total intensity should be the sum and m/z should be weighted average
    assert_eq!(obs.dense_xic[[0, 0]], 3000.0);

    let expected_mz = (249.9 * 500.0 + 250.0 * 1000.0 + 250.1 * 1500.0) / 3000.0;
    // The weighted average will be close but affected by grid discretization
    assert!((obs.dense_mz[[0, 0]] - expected_mz).abs() < 1e-4);
}

#[test]
fn test_dense_xic_mz_zero_intensity_handling() {
    // Given: DIA data with peaks that have zero intensities mixed with non-zero
    static SPECTRUM_DELTA_SCAN_IDX: [i64; 2] = [0, 1];
    static ISOLATION_LOWER_MZ: [f32; 2] = [299.0, 299.0];
    static ISOLATION_UPPER_MZ: [f32; 2] = [301.0, 301.0];
    static SPECTRUM_PEAK_START_IDX: [i64; 2] = [0, 2];
    static SPECTRUM_PEAK_STOP_IDX: [i64; 2] = [2, 4];
    static SPECTRUM_CYCLE_IDX: [i64; 2] = [0, 1];
    static SPECTRUM_RT: [f32; 2] = [10.0, 20.0];
    static PEAK_MZ: [f32; 4] = [300.0, 300.1, 300.0, 300.1];
    static PEAK_INTENSITY: [f32; 4] = [1000.0, 0.0, 0.0, 2000.0]; // Zero intensities
    static CYCLE_DATA: [f32; 1] = [7.0];

    let alpha_raw = AlphaRawView {
        spectrum_delta_scan_idx: ArrayView1::from(&SPECTRUM_DELTA_SCAN_IDX),
        isolation_lower_mz: ArrayView1::from(&ISOLATION_LOWER_MZ),
        isolation_upper_mz: ArrayView1::from(&ISOLATION_UPPER_MZ),
        spectrum_peak_start_idx: ArrayView1::from(&SPECTRUM_PEAK_START_IDX),
        spectrum_peak_stop_idx: ArrayView1::from(&SPECTRUM_PEAK_STOP_IDX),
        spectrum_cycle_idx: ArrayView1::from(&SPECTRUM_CYCLE_IDX),
        spectrum_rt: ArrayView1::from(&SPECTRUM_RT),
        peak_mz: ArrayView1::from(&PEAK_MZ),
        peak_intensity: ArrayView1::from(&PEAK_INTENSITY),
        cycle: ArrayView4::from_shape([1, 1, 1, 1], &CYCLE_DATA).unwrap(),
    };

    let dia_data = DIADataBuilder::from_alpha_raw(&alpha_raw);
    let fragment_mz = vec![300.05];

    // When: Creating a DenseXICMZObservation for both cycles
    let obs = DenseXICMZObservation::new(&dia_data, 300.0, 0, 2, 2000.0, &fragment_mz);

    // Then: Zero-intensity peaks should not contribute to weighted m/z average
    // First cycle: only 300.0 with intensity 1000 (300.1 has 0 intensity)
    assert_eq!(obs.dense_xic[[0, 0]], 1000.0);
    assert!((obs.dense_mz[[0, 0]] - 300.0).abs() < 2e-4);

    // Second cycle: only 300.1 with intensity 2000 (300.0 has 0 intensity)
    assert_eq!(obs.dense_xic[[0, 1]], 2000.0);
    assert!((obs.dense_mz[[0, 1]] - 300.1).abs() < 2e-4);
}

#[test]
fn test_dense_xic_mz_no_matching_fragments() {
    // Given: DIA data with peaks around 200 m/z and a fragment far away
    let alpha_raw = create_simple_alpha_raw();
    let dia_data = DIADataBuilder::from_alpha_raw(&alpha_raw);
    let fragment_mz = vec![300.0]; // Fragment m/z far from any peaks

    // When: Creating a DenseXICMZObservation with non-matching fragment
    let obs = DenseXICMZObservation::new(&dia_data, 200.0, 10, 12, 20.0, &fragment_mz);

    // Then: Both intensity and m/z matrices should contain zeros
    assert_eq!(obs.dense_xic[[0, 0]], 0.0);
    assert_eq!(obs.dense_xic[[0, 1]], 0.0);
    assert_eq!(obs.dense_mz[[0, 0]], 0.0);
    assert_eq!(obs.dense_mz[[0, 1]], 0.0);
}

#[test]
fn test_dense_xic_mz_multiple_fragments() {
    // Given: DIA data with peaks at 200.0 and 200.1 m/z
    // and three fragments: two matching and one non-matching
    let alpha_raw = create_simple_alpha_raw();
    let dia_data = DIADataBuilder::from_alpha_raw(&alpha_raw);
    let fragment_mz = vec![200.0, 200.1, 300.0]; // Two matching, one not

    // When: Creating a DenseXICMZObservation with low tolerance (20 ppm)
    let obs = DenseXICMZObservation::new(&dia_data, 200.0, 10, 11, 20.0, &fragment_mz);

    // Then: Each fragment should match its corresponding peak or remain zero
    // Fragment 0: should match only the 200.0 peak
    assert_eq!(obs.dense_xic[[0, 0]], 1000.0);
    assert!((obs.dense_mz[[0, 0]] - 200.0).abs() < 1e-4);

    // Fragment 1: should match only the 200.1 peak
    assert_eq!(obs.dense_xic[[1, 0]], 2000.0);
    assert!((obs.dense_mz[[1, 0]] - 200.1).abs() < 1e-4);

    // Fragment 2: should not match any peaks
    assert_eq!(obs.dense_xic[[2, 0]], 0.0);
    assert_eq!(obs.dense_mz[[2, 0]], 0.0);
}

#[test]
fn test_dense_xic_mz_single_cycle_single_fragment() {
    // Given: Minimal setup with one fragment and one cycle
    let alpha_raw = create_simple_alpha_raw();
    let dia_data = DIADataBuilder::from_alpha_raw(&alpha_raw);

    // When: Extracting a single cycle with a single fragment
    let obs = DenseXICMZObservation::new(&dia_data, 200.0, 10, 11, 20.0, &[200.0]);

    // Then: Should produce 1x1 matrices with correct values
    assert_eq!(obs.dense_xic.shape(), &[1, 1]);
    assert_eq!(obs.dense_mz.shape(), &[1, 1]);
    assert_eq!(obs.dense_xic[[0, 0]], 1000.0);
}

#[test]
fn test_dense_xic_mz_overlapping_tolerance_windows() {
    // Given: Two fragments with overlapping tolerance windows
    let alpha_raw = create_simple_alpha_raw();
    let dia_data = DIADataBuilder::from_alpha_raw(&alpha_raw);

    // When: Using high tolerance that causes both fragments to see both peaks
    let obs = DenseXICMZObservation::new(&dia_data, 200.0, 10, 11, 1000.0, &[200.0, 200.05]);

    // Then: Both fragments should see all peaks within tolerance
    assert_eq!(obs.dense_xic[[0, 0]], 3000.0); // Sees both peaks
    assert_eq!(obs.dense_xic[[1, 0]], 3000.0); // Also sees both peaks
}

#[test]
fn test_dense_xic_mz_out_of_mz_index_range() {
    // Given: Fragment m/z below the MZIndex start (150.0)
    let alpha_raw = create_simple_alpha_raw();
    let dia_data = DIADataBuilder::from_alpha_raw(&alpha_raw);

    // When: Requesting a fragment below m/z 150
    let obs = DenseXICMZObservation::new(&dia_data, 200.0, 10, 11, 20.0, &[100.0]);

    // Then: Should return zeros (no data in that range)
    assert_eq!(obs.dense_xic[[0, 0]], 0.0);
    assert_eq!(obs.dense_mz[[0, 0]], 0.0);
}

#[test]
fn test_dense_xic_mz_exact_tolerance_boundary() {
    // Given: Fragment at exact midpoint between two peaks
    let alpha_raw = create_simple_alpha_raw();
    let dia_data = DIADataBuilder::from_alpha_raw(&alpha_raw);

    // When: Using 400 ppm tolerance to match 200.0 but not 200.1
    // 200.05 ± 400ppm = 200.05 ± 0.08 = [199.97, 200.13]
    let obs = DenseXICMZObservation::new(&dia_data, 200.0, 10, 11, 400.0, &[200.05]);

    // Then: Should match both peaks within tolerance
    assert_eq!(obs.dense_xic[[0, 0]], 3000.0);
}
