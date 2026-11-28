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

#[cfg(test)]
mod fragment_isotopes {
    use super::{PeakGroupQuantification, QuantificationParameters};
    use crate::candidate::Candidate;
    use crate::peak_group_scoring::tests::feature_sets::{
        dia_data, dia_data_with_isotopes, library, APEX_CYCLE, FRAGMENTS,
    };
    use crate::precursor_quantified::PrecursorQuantified;

    /// Quantifies the library precursor at its apex in `run`, with or without the M+1 XICs.
    fn quantify(run: &crate::dia_data::DIAData, fragment_isotopes: bool) -> PrecursorQuantified {
        let mut params = QuantificationParameters::new();
        params.fragment_isotopes = fragment_isotopes;
        let precursor = library()
            .get_precursor_by_idx_filtered(0, true, true, false, params.top_k_fragments)
            .unwrap();
        let apex = Candidate::new(0, 0, 1.0, APEX_CYCLE - 3, APEX_CYCLE, APEX_CYCLE + 3);
        PeakGroupQuantification::new(params)
            .quantify_precursor(run, &precursor, &apex, &[])
            .unwrap()
    }

    #[test]
    fn test_fragment_isotopes_default_to_off() {
        // Given / When: default quantification parameters
        let params = QuantificationParameters::new();

        // Then: the M+1 XICs are not added
        assert!(!params.fragment_isotopes);
    }

    #[test]
    fn test_fragment_isotopes_add_the_mplus1_intensity_and_keep_the_mass_errors() {
        // Given: a run in which every fragment has its M+1 peak at the averagine ratio
        let run = dia_data_with_isotopes();

        // When: the precursor is quantified with and without the M+1 XICs
        let monoisotopic = quantify(&run, false);
        let summed = quantify(&run, true);

        // Then: each fragment intensity grows by its M+1 share, and the observed m/z values stay the same
        assert_eq!(summed.fragment_mz, monoisotopic.fragment_mz);
        for i in 0..monoisotopic.fragment_mz.len() {
            let expected_ratio = 1.0 + 0.000_54 * monoisotopic.fragment_mz[i];
            let ratio = summed.fragment_intensity[i] / monoisotopic.fragment_intensity[i];
            assert!(
                (ratio - expected_ratio).abs() < 1e-3,
                "{ratio} vs {expected_ratio}"
            );
        }
        assert_eq!(
            summed.fragment_mz_observed,
            monoisotopic.fragment_mz_observed
        );
        assert_eq!(
            summed.fragment_mass_error_observed,
            monoisotopic.fragment_mass_error_observed
        );
    }

    #[test]
    fn test_fragment_isotopes_change_nothing_in_a_run_without_isotope_peaks() {
        // Given: a run without isotope peaks
        let run = dia_data();

        // When: the precursor is quantified with and without the M+1 XICs
        let monoisotopic = quantify(&run, false);
        let summed = quantify(&run, true);

        // Then: the intensities are the same
        assert_eq!(summed.fragment_intensity, monoisotopic.fragment_intensity);
        assert_eq!(summed.fragment_mz.len(), FRAGMENTS.len());
    }
}
