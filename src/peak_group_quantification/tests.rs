#[allow(unused_imports)]
use super::{PeakGroupQuantification, QuantificationParameters};

#[test]
fn test_quantification_parameters_default() {
    let params = QuantificationParameters::new();
    assert_eq!(params.tolerance_ppm, 20.0);
    assert_eq!(params.top_k_fragments, 50);
}

#[test]
fn test_peak_group_quantification_creation() {
    let params = QuantificationParameters::new();
    let quantifier = PeakGroupQuantification::new(params);

    assert_eq!(quantifier.params.tolerance_ppm, 20.0);
    assert_eq!(quantifier.params.top_k_fragments, 50);
}
