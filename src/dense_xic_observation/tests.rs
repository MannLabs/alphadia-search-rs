use super::*;
use crate::dia_data::AlphaRawView;
use crate::dia_data_builder::DIADataBuilder;
use numpy::ndarray::ArrayView1;

fn create_simple_alpha_raw() -> AlphaRawView<'static> {
    static SPECTRUM_DELTA_SCAN_IDX: [i64; 1] = [0];
    static ISOLATION_LOWER_MZ: [f32; 1] = [124.0];
    static ISOLATION_UPPER_MZ: [f32; 1] = [126.0];
    static SPECTRUM_PEAK_START_IDX: [i64; 1] = [0];
    static SPECTRUM_PEAK_STOP_IDX: [i64; 1] = [2];
    static SPECTRUM_CYCLE_IDX: [i64; 1] = [10];
    static SPECTRUM_RT: [f32; 1] = [100.0];
    static PEAK_MZ: [f32; 2] = [125.0, 125.1];
    static PEAK_INTENSITY: [f32; 2] = [1000.0, 2000.0];

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
    }
}

#[test]
fn test_basic_creation() {
    let alpha_raw = create_simple_alpha_raw();
    let dia_data = DIADataBuilder::from_alpha_raw(&alpha_raw);
    let fragment_mz = vec![125.0];

    let obs = DenseXICObservation::new(&dia_data, 125.0, 10, 12, 20.0, &fragment_mz);

    assert_eq!(obs.dense_xic.nrows(), 1);
    assert_eq!(obs.dense_xic.ncols(), 2);
}

#[test]
fn test_optimized_data_creation() {
    let alpha_raw = create_simple_alpha_raw();
    let dia_data = DIADataBuilder::from_alpha_raw(&alpha_raw);
    let fragment_mz = vec![125.0];

    let obs = DenseXICObservation::new(&dia_data, 125.0, 10, 11, 20.0, &fragment_mz);

    assert_eq!(obs.dense_xic.nrows(), 1);
    assert_eq!(obs.dense_xic.ncols(), 1);
}

#[test]
fn test_empty_fragments() {
    let alpha_raw = create_simple_alpha_raw();
    let dia_data = DIADataBuilder::from_alpha_raw(&alpha_raw);
    let fragment_mz: Vec<f32> = vec![];

    let obs = DenseXICObservation::new(&dia_data, 125.0, 10, 12, 20.0, &fragment_mz);

    assert_eq!(obs.dense_xic.nrows(), 0);
    assert_eq!(obs.dense_xic.ncols(), 2);
}

#[test]
fn test_metadata_storage() {
    let alpha_raw = create_simple_alpha_raw();
    let dia_data = DIADataBuilder::from_alpha_raw(&alpha_raw);
    let fragment_mz = vec![125.0];

    let obs = DenseXICObservation::new(&dia_data, 125.0, 10, 15, 50.0, &fragment_mz);

    assert_eq!(obs.cycle_start_idx, 10);
    assert_eq!(obs.cycle_stop_idx, 15);
    assert_eq!(obs.mass_tolerance, 50.0);
    assert!(!obs.contributing_obs_indices.is_empty());
}

#[test]
fn test_multiple_fragments() {
    let alpha_raw = create_simple_alpha_raw();
    let dia_data = DIADataBuilder::from_alpha_raw(&alpha_raw);
    let fragment_mz = vec![125.0, 125.1, 130.0];

    let obs = DenseXICObservation::new(&dia_data, 125.0, 10, 12, 20.0, &fragment_mz);

    assert_eq!(obs.dense_xic.nrows(), 3);
    assert_eq!(obs.dense_xic.ncols(), 2);
}
