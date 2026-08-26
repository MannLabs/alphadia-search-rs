#[allow(unused_imports)]
use super::{PeakGroupQuantification, QuantificationMethod, QuantificationParameters};
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

/// The quantification method this build ships as its default.
///
/// Benchmarking branches that flip the default change this line alongside
/// `QuantificationParameters::new`, so the two cannot drift apart unnoticed.
#[cfg(test)]
const EXPECTED_DEFAULT_METHOD: QuantificationMethod = QuantificationMethod::Sum;

#[test]
fn test_quantification_method_default_is_the_expected_one() {
    let params = QuantificationParameters::new();
    assert_eq!(params.method, EXPECTED_DEFAULT_METHOD);
    assert_eq!(params.get_method(), EXPECTED_DEFAULT_METHOD.as_str());
}

#[test]
fn test_integration_parameter_defaults() {
    let params = QuantificationParameters::new();

    assert_eq!(params.template_smoothing_lambda, 5e-7);
    assert_eq!(params.template_min_correlation, 0.5);
    assert_eq!(params.boundary_valley_tolerance, 1.15);
    assert_eq!(params.min_area_surviving_ratio, 0.0);
    assert!(params.subtract_baseline);
    assert_eq!(params.robust_iterations, 3);
    assert!(params.projection_fit_baseline);
    assert_eq!(params.huber_k, 1.5);
    assert_eq!(params.emg_extrapolation_factor, 1.0);
    assert_eq!(params.emg_upsample_factor, 8);
}

#[test]
fn test_quantification_method_names_round_trip() {
    for method in [
        QuantificationMethod::Sum,
        QuantificationMethod::Trapezoid,
        QuantificationMethod::BoundaryTrapezoid,
        QuantificationMethod::ProfileProjection,
        QuantificationMethod::EmgFit,
    ] {
        assert_eq!(
            QuantificationMethod::from_name(method.as_str()),
            Ok(method),
            "{method:?}"
        );
    }
}

#[test]
fn test_quantification_method_aliases_and_casing() {
    assert_eq!(
        QuantificationMethod::from_name("trapezoidal"),
        Ok(QuantificationMethod::Trapezoid)
    );
    assert_eq!(
        QuantificationMethod::from_name("Boundary"),
        Ok(QuantificationMethod::BoundaryTrapezoid)
    );
    assert_eq!(
        QuantificationMethod::from_name("MATCHED_FILTER"),
        Ok(QuantificationMethod::ProfileProjection)
    );
    assert_eq!(
        QuantificationMethod::from_name("emg"),
        Ok(QuantificationMethod::EmgFit)
    );
}

#[test]
fn test_quantification_method_rejects_unknown_names() {
    let error = QuantificationMethod::from_name("gaussian").unwrap_err();
    assert!(error.contains("gaussian"), "{error}");
    assert!(error.contains("profile_projection"), "{error}");
}

#[test]
fn test_update_sets_method_and_integration_parameters() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let mut params = QuantificationParameters::new();

        let dict = PyDict::new(py);
        dict.set_item("method", "emg_fit").unwrap();
        dict.set_item("template_smoothing_lambda", 0.0).unwrap();
        dict.set_item("template_min_correlation", 0.7).unwrap();
        dict.set_item("boundary_valley_tolerance", 1.3).unwrap();
        dict.set_item("min_area_surviving_ratio", 5.0).unwrap();
        dict.set_item("subtract_baseline", false).unwrap();
        dict.set_item("robust_iterations", 5).unwrap();
        dict.set_item("projection_fit_baseline", false).unwrap();
        dict.set_item("huber_k", 2.0).unwrap();
        dict.set_item("emg_extrapolation_factor", 1.5).unwrap();
        dict.set_item("emg_upsample_factor", 16).unwrap();

        params.update(&dict).unwrap();

        assert_eq!(params.method, QuantificationMethod::EmgFit);
        assert_eq!(params.template_smoothing_lambda, 0.0);
        assert_eq!(params.template_min_correlation, 0.7);
        assert_eq!(params.boundary_valley_tolerance, 1.3);
        assert_eq!(params.min_area_surviving_ratio, 5.0);
        assert!(!params.subtract_baseline);
        assert_eq!(params.robust_iterations, 5);
        assert!(!params.projection_fit_baseline);
        assert_eq!(params.huber_k, 2.0);
        assert_eq!(params.emg_extrapolation_factor, 1.5);
        assert_eq!(params.emg_upsample_factor, 16);
    });
}

#[test]
fn test_update_rejects_an_unknown_method() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let mut params = QuantificationParameters::new();

        let original = params.method;

        let dict = PyDict::new(py);
        dict.set_item("method", "not_a_method").unwrap();

        assert!(params.update(&dict).is_err());
        // A rejected update must leave the parameters untouched.
        assert_eq!(params.method, original);
    });
}
