#[allow(unused_imports)]
use super::{background_corrected_intensities, PeakGroupQuantification, QuantificationParameters};
#[allow(unused_imports)]
use numpy::ndarray::array;
#[allow(unused_imports)]
use pyo3::{
    types::{PyDict, PyDictMethods},
    Python,
};

#[test]
fn test_peak_group_quantification_creation() {
    let params = QuantificationParameters::new();
    let _quantifier = PeakGroupQuantification::new(params);
    // Test passes if creation succeeds without panicking
}

#[test]
fn test_parameter_defaults() {
    let params = QuantificationParameters::new();

    // Verify all default values
    assert_eq!(params.tolerance_ppm, 7.0);
    assert_eq!(params.top_k_fragments, 10000);
    assert_eq!(params.background_subtraction, "none");
}

#[test]
fn test_parameter_internal_modification() {
    let mut params = QuantificationParameters::new();

    // Test that we can still modify parameters internally in Rust
    // (This is for internal Rust usage, not Python)
    params.tolerance_ppm = 12.0;
    params.top_k_fragments = 50;

    assert_eq!(params.tolerance_ppm, 12.0);
    assert_eq!(params.top_k_fragments, 50);
}

#[test]
fn test_update_method_partial() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let mut params = QuantificationParameters::new();

        // Update only one parameter
        let dict = PyDict::new(py);
        dict.set_item("tolerance_ppm", 15.0).unwrap();

        params.update(&dict).unwrap();

        // Verify only the updated parameter changed
        assert_eq!(params.tolerance_ppm, 15.0);
        assert_eq!(params.top_k_fragments, 10000); // Should remain unchanged
    });
}

#[test]
fn test_update_method_all_parameters() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let mut params = QuantificationParameters::new();

        // Update all parameters
        let dict = PyDict::new(py);
        dict.set_item("tolerance_ppm", 20.0).unwrap();
        dict.set_item("top_k_fragments", 200).unwrap();

        params.update(&dict).unwrap();

        // Verify all parameters changed
        assert_eq!(params.tolerance_ppm, 20.0);
        assert_eq!(params.top_k_fragments, 200);
    });
}

#[test]
fn test_update_method_empty_dict() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let mut params = QuantificationParameters::new();
        let original_tolerance = params.tolerance_ppm;
        let original_fragments = params.top_k_fragments;

        // Update with empty dictionary
        let dict = PyDict::new(py);
        params.update(&dict).unwrap();

        // Verify no parameters changed
        assert_eq!(params.tolerance_ppm, original_tolerance);
        assert_eq!(params.top_k_fragments, original_fragments);
    });
}

#[test]
fn test_update_background_subtraction() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let mut params = QuantificationParameters::new();
        let dict = PyDict::new(py);
        dict.set_item("background_subtraction", "profile").unwrap();
        params.update(&dict).unwrap();
        assert_eq!(params.background_subtraction, "profile");

        let bad = PyDict::new(py);
        bad.set_item("background_subtraction", "median").unwrap();
        assert!(params.update(&bad).is_err());
        assert_eq!(params.background_subtraction, "profile");
    });
}

#[test]
fn test_background_none_is_plain_sum() {
    let xic = array![[1.0f32, 3.0, 9.0, 3.0, 1.0], [2.0, 2.0, 2.0, 2.0, 2.0]];
    let profile = [0.1f32, 0.3, 1.0, 0.3, 0.1];
    assert_eq!(
        background_corrected_intensities(&xic, &profile, "none"),
        vec![17.0, 10.0]
    );
}

#[test]
fn test_background_removes_flat_offset() {
    // signal 10 * profile on top of a flat background of 5 in every cycle
    let profile = [0.0f32, 0.25, 1.0, 0.25, 0.0];
    let signal: Vec<f32> = profile.iter().map(|p| 10.0 * p).collect();
    let xic = numpy::ndarray::Array2::from_shape_fn((1, 5), |(_, c)| signal[c] + 5.0);
    let expected = 10.0 * profile.iter().sum::<f32>();
    for mode in ["edges", "profile"] {
        let got = background_corrected_intensities(&xic, &profile, mode)[0];
        assert!((got - expected).abs() < 1e-4, "{mode}: {got} vs {expected}");
    }
}

#[test]
fn test_background_pure_noise_is_clipped_at_zero() {
    let profile = [0.0f32, 0.25, 1.0, 0.25, 0.0];
    // fragment 0 carries signal on a background, fragment 1 is anti-correlated with the profile: no signal
    let xic = array![[5.0f32, 7.5, 15.0, 7.5, 5.0], [6.0, 5.0, 2.0, 5.0, 6.0]];
    for mode in ["edges", "profile"] {
        let got = background_corrected_intensities(&xic, &profile, mode);
        assert!(got[0] > 0.0, "{mode}");
        assert_eq!(got[1], 0.0, "{mode}");
    }
}

#[test]
fn test_background_all_clipped_falls_back_to_sum() {
    let profile = [0.0f32, 0.25, 1.0, 0.25, 0.0];
    let xic = array![[6.0f32, 5.0, 2.0, 5.0, 6.0]];
    for mode in ["edges", "profile"] {
        assert_eq!(
            background_corrected_intensities(&xic, &profile, mode),
            vec![24.0]
        );
    }
}

#[test]
fn test_background_flat_profile_falls_back_to_sum() {
    let profile = [1.0f32; 5];
    let xic = array![[1.0f32, 2.0, 3.0, 2.0, 1.0]];
    assert_eq!(
        background_corrected_intensities(&xic, &profile, "profile"),
        vec![9.0]
    );
}

#[test]
fn test_weighted_recovers_a_profile_shaped_signal() {
    let profile = [0.1f32, 0.3, 1.0, 0.3, 0.1];
    let xic = numpy::ndarray::Array2::from_shape_fn((1, 5), |(_, c)| 10.0 * profile[c]);
    let got = background_corrected_intensities(&xic, &profile, "weighted")[0];
    let expected = 10.0 * profile.iter().sum::<f32>();
    assert!((got - expected).abs() < 1e-4, "{got} vs {expected}");
}

#[test]
fn test_weighted_keeps_less_flat_background_than_the_sum() {
    // a flat background of 5 alone: the plain sum keeps 5 per cycle, the weighted fit (sum p)^2 / sum p^2 cycles
    let profile = [0.0f32, 0.25, 1.0, 0.25, 0.0];
    let xic = array![[5.0f32, 5.0, 5.0, 5.0, 5.0]];
    let got = background_corrected_intensities(&xic, &profile, "weighted")[0];
    let sum: f32 = profile.iter().sum();
    let square_sum: f32 = profile.iter().map(|p| p * p).sum();
    assert!((got - 5.0 * sum * sum / square_sum).abs() < 1e-4);
    assert!(got < 25.0);
}

#[test]
fn test_weighted_empty_profile_falls_back_to_sum() {
    let profile = [0.0f32; 5];
    let xic = array![[1.0f32, 2.0, 3.0, 2.0, 1.0]];
    assert_eq!(
        background_corrected_intensities(&xic, &profile, "weighted"),
        vec![9.0]
    );
}
