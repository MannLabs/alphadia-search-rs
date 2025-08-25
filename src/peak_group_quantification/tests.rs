#[allow(unused_imports)]
use super::{PeakGroupQuantification, QuantificationParameters};

#[test]
fn test_peak_group_quantification_creation() {
    let params = QuantificationParameters::new();
    let _quantifier = PeakGroupQuantification::new(params);
    // Test passes if creation succeeds without panicking
}
