use super::*;

#[test]
fn test_precursor_quantified_creation() {
    let precursor = PrecursorQuantified {
        idx: 1,
        mz: 500.5,
        rt: 120.0,
        naa: 12,
        fragment_mz: vec![200.0, 300.0],
        fragment_intensity: vec![100.0, 150.0],
        fragment_cardinality: vec![1, 1],
        fragment_charge: vec![1, 1],
        fragment_loss_type: vec![0, 0],
        fragment_number: vec![1, 2],
        fragment_position: vec![1, 2],
        fragment_type: vec![1, 1],
        fragment_mz_observed: vec![200.1, 299.9],
        fragment_correlation_observed: vec![0.95, 0.88],
        fragment_mass_error_observed: vec![0.1, -0.1],
    };

    assert_eq!(precursor.idx, 1);
    assert_eq!(precursor.mz, 500.5);
    assert_eq!(precursor.rt, 120.0);
    assert_eq!(precursor.naa, 12);
    assert_eq!(precursor.fragment_mz.len(), 2);
    assert_eq!(precursor.fragment_mz_observed.len(), 2);
    assert_eq!(precursor.fragment_correlation_observed.len(), 2);
    assert_eq!(precursor.fragment_mass_error_observed.len(), 2);
}

#[test]
fn test_precursor_quantified_data_consistency() {
    let precursor = PrecursorQuantified {
        idx: 0,
        mz: 400.0,
        rt: 80.0,
        naa: 8,
        fragment_mz: vec![150.0, 250.0, 350.0],
        fragment_intensity: vec![50.0, 75.0, 100.0],
        fragment_cardinality: vec![1, 1, 1],
        fragment_charge: vec![1, 1, 1],
        fragment_loss_type: vec![0, 0, 0],
        fragment_number: vec![1, 2, 3],
        fragment_position: vec![1, 2, 3],
        fragment_type: vec![1, 1, 1],
        fragment_mz_observed: vec![150.05, 249.95, 350.1],
        fragment_correlation_observed: vec![0.92, 0.87, 0.94],
        fragment_mass_error_observed: vec![0.05, -0.05, 0.1],
    };

    // Verify all fragment vectors have the same length
    let fragment_count = precursor.fragment_mz.len();
    assert_eq!(precursor.fragment_intensity.len(), fragment_count);
    assert_eq!(precursor.fragment_cardinality.len(), fragment_count);
    assert_eq!(precursor.fragment_charge.len(), fragment_count);
    assert_eq!(precursor.fragment_loss_type.len(), fragment_count);
    assert_eq!(precursor.fragment_number.len(), fragment_count);
    assert_eq!(precursor.fragment_position.len(), fragment_count);
    assert_eq!(precursor.fragment_type.len(), fragment_count);
    assert_eq!(precursor.fragment_mz_observed.len(), fragment_count);
    assert_eq!(
        precursor.fragment_correlation_observed.len(),
        fragment_count
    );
    assert_eq!(precursor.fragment_mass_error_observed.len(), fragment_count);
}
