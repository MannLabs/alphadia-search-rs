use super::parameters::SelectionParameters;
use super::*;
use numpy::ndarray::arr1;
use pyo3::types::{PyDict, PyDictMethods};
use pyo3::Python;

#[test]
fn test_find_local_maxima_multiple_peaks() {
    // Local maxima at indices 2, 4 and 8
    let array = arr1(&[1.0, 2.0, 3.0, 2.0, 5.0, 3.0, 2.0, 4.0, 7.0, 5.0, 3.0, 2.0]);
    let offset = 10;
    let (indices, values) = find_local_maxima(&array, offset);

    assert_eq!(indices, vec![18, 14, 12]);
    assert_eq!(values, vec![7.0, 5.0, 3.0]);
}

#[test]
fn test_find_local_maxima_narrow_peak_with_noisy_shoulders() {
    // A one-cycle peak whose points two cycles away are higher than the direct neighbours.
    let array = arr1(&[0.0, 1.0, 0.5, 4.0, 0.6, 1.0, 0.0]);
    let (indices, values) = find_local_maxima(&array, 0);

    assert_eq!(indices, vec![3, 1, 5]);
    assert_eq!(values, vec![4.0, 1.0, 1.0]);
}

#[test]
fn test_find_local_maxima_no_peaks() {
    let array = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    let (indices, values) = find_local_maxima(&array, 0);

    assert!(indices.is_empty());
    assert!(values.is_empty());
}

#[test]
fn test_find_local_maxima_too_few_points() {
    let array = arr1(&[1.0, 2.0]);
    let (indices, values) = find_local_maxima(&array, 0);

    assert!(indices.is_empty());
    assert!(values.is_empty());
}

#[test]
fn test_find_local_maxima_edges_are_not_maxima() {
    let array = arr1(&[5.0, 1.0, 2.0, 1.0, 5.0]);
    let (indices, values) = find_local_maxima(&array, 0);

    assert_eq!(indices, vec![2]);
    assert_eq!(values, vec![2.0]);
}

#[test]
fn test_find_local_maxima_flat_regions() {
    let array = arr1(&[1.0, 2.0, 5.0, 5.0, 5.0, 2.0, 1.0]);
    let (indices, values) = find_local_maxima(&array, 0);

    assert!(indices.is_empty());
    assert!(values.is_empty());
}

#[test]
fn test_parameter_defaults() {
    let params = SelectionParameters::new();

    // Verify all default values
    assert_eq!(params.fwhm_rt, 3.0);
    assert_eq!(params.kernel_size, 20);
    assert_eq!(params.peak_length, 5);
    assert_eq!(params.mass_tolerance, 7.0);
    assert_eq!(params.rt_tolerance, 200.0);
    assert_eq!(params.candidate_count, 3);
}

#[test]
fn test_parameter_internal_modification() {
    let mut params = SelectionParameters::new();

    // Test that we can still modify parameters internally in Rust
    // (This is for internal Rust usage, not Python)
    params.fwhm_rt = 5.0;
    params.kernel_size = 30;
    params.peak_length = 8;
    params.mass_tolerance = 12.0;
    params.rt_tolerance = 300.0;
    params.candidate_count = 4;

    assert_eq!(params.fwhm_rt, 5.0);
    assert_eq!(params.kernel_size, 30);
    assert_eq!(params.peak_length, 8);
    assert_eq!(params.mass_tolerance, 12.0);
    assert_eq!(params.rt_tolerance, 300.0);
    assert_eq!(params.candidate_count, 4);
}

#[test]
fn test_update_method_partial() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let mut params = SelectionParameters::new();

        // Update only one parameter
        let dict = PyDict::new(py);
        dict.set_item("rt_tolerance", 150.0).unwrap();

        params.update(&dict).unwrap();

        // Verify only the updated parameter changed
        assert_eq!(params.rt_tolerance, 150.0);
    });
}

#[test]
fn test_candidate_cycle_limits_apex_near_run_start() {
    assert_eq!(candidate_cycle_limits(2, 5, 100), (0, 8));
}

#[test]
fn test_candidate_cycle_limits_apex_near_run_end() {
    assert_eq!(candidate_cycle_limits(97, 5, 100), (92, 100));
}

#[test]
fn test_candidate_cycle_limits_interior() {
    assert_eq!(candidate_cycle_limits(50, 5, 100), (45, 56));
}

#[test]
fn test_rt_prior_defaults_to_off() {
    let params = SelectionParameters::new();
    assert_eq!(params.rt_prior_scale, 0.0);
    assert_eq!(params.rt_prior_df, 0.0);
}

#[test]
fn test_rt_prior_parameters_update() {
    Python::initialize();
    Python::attach(|py| {
        let mut params = SelectionParameters::new();
        let dict = PyDict::new(py);
        dict.set_item("rt_prior_scale", 20.0).unwrap();
        dict.set_item("rt_prior_df", 5.0).unwrap();
        params.update(&dict).unwrap();
        assert_eq!(params.rt_prior_scale, 20.0);
        assert_eq!(params.rt_prior_df, 5.0);
    });
}

#[test]
fn test_rt_log_prior_is_zero_at_the_predicted_rt_and_falls_symmetrically() {
    // Cycles every 10 s, predicted RT 50 s, Student-t prior with scale 10 s and 5 degrees of freedom.
    let rt = arr1(&[0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
    let mut score = arr1(&[1.0, 1.0, 1.0, 1.0, 1.0]);
    add_rt_log_prior(&mut score, &rt, 3, 50.0, 10.0, 5.0);

    // Cycles 3..8 are 30, 40, 50, 60, 70 s.
    let at_one_scale = -3.0 * (1.0f32 + 1.0 / 5.0).ln();
    let at_two_scales = -3.0 * (1.0f32 + 4.0 / 5.0).ln();
    let expected = [
        1.0 + at_two_scales,
        1.0 + at_one_scale,
        1.0,
        1.0 + at_one_scale,
        1.0 + at_two_scales,
    ];
    for (value, expected) in score.iter().zip(expected) {
        assert!((value - expected).abs() < 1e-5, "{value} != {expected}");
    }
}

#[test]
fn test_rt_log_prior_moves_the_best_peak_towards_the_predicted_rt() {
    // Two equal peaks at cycles 1 and 5, the predicted RT at cycle 5.
    let rt = arr1(&[0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    let mut score = arr1(&[0.0, 5.0, 0.0, 0.0, 0.0, 5.0, 0.0]);
    add_rt_log_prior(&mut score, &rt, 0, 50.0, 10.0, 5.0);

    let (indices, _) = find_local_maxima(&score, 0);
    assert_eq!(indices[0], 5);
}

#[test]
fn test_min_fragments_defaults_to_off() {
    assert_eq!(SelectionParameters::new().min_fragments, 0);
}

#[test]
fn test_fragments_with_signal_counts_fragments_around_the_apex() {
    // 4 fragments over 7 cycles; apex at cycle 3, so cycles 2 to 4 count
    let dense_xic = numpy::ndarray::arr2(&[
        [0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0], // signal at the apex
        [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0], // signal one cycle before
        [9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0], // signal only away from the apex
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // no signal
    ]);
    assert_eq!(fragments_with_signal(&dense_xic, 3), 2);
    // An apex at the first cycle only looks at cycles 0 and 1
    assert_eq!(fragments_with_signal(&dense_xic, 0), 1);
}

#[test]
fn test_min_score_defaults_to_off() {
    assert_eq!(SelectionParameters::new().min_score, f32::NEG_INFINITY);
}

#[test]
fn test_min_score_parameter_update() {
    Python::initialize();
    Python::attach(|py| {
        let mut params = SelectionParameters::new();
        let dict = PyDict::new(py);
        dict.set_item("min_score", 9.5).unwrap();
        params.update(&dict).unwrap();
        assert_eq!(params.min_score, 9.5);
    });
}
