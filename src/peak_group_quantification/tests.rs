#[allow(unused_imports)]
use super::{PeakGroupQuantification, QuantificationParameters};
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
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
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
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
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
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
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
