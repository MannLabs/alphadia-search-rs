use super::*;

#[test]
fn test_precursor_quantified_creation() {
    let precursor = PrecursorQuantified {
        precursor_idx: 1,
        mz_library: 500.3,
        mz: 500.5,
        rt_library: 119.8,
        rt: 120.0,
        naa: 12,
        rank: 1,
        rt_observed: 121.5,
        fragment_mz_library: vec![199.8, 299.9],
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

    assert_eq!(precursor.precursor_idx, 1);
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
        precursor_idx: 0,
        mz_library: 399.8,
        mz: 400.0,
        rt_library: 79.5,
        rt: 80.0,
        naa: 8,
        rank: 2,
        rt_observed: 82.3,
        fragment_mz_library: vec![149.9, 249.8, 349.9],
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

#[test]
fn test_filter_fragments_by_intensity_keeps_valid_fragments() {
    let precursor = PrecursorQuantified {
        precursor_idx: 1,
        mz_library: 500.3,
        mz: 500.5,
        rt_library: 119.8,
        rt: 120.0,
        naa: 12,
        rank: 1,
        rt_observed: 121.5,
        fragment_mz_library: vec![199.8, 299.9, 399.7],
        fragment_mz: vec![200.0, 300.0, 400.0],
        fragment_intensity: vec![100.0, 0.0, 150.0],
        fragment_cardinality: vec![1, 1, 1],
        fragment_charge: vec![1, 1, 1],
        fragment_loss_type: vec![0, 0, 0],
        fragment_number: vec![1, 2, 3],
        fragment_position: vec![1, 2, 3],
        fragment_type: vec![1, 1, 1],
        fragment_mz_observed: vec![200.1, 299.9, 399.8],
        fragment_correlation_observed: vec![0.95, 0.88, 0.92],
        fragment_mass_error_observed: vec![0.1, -0.1, 0.05],
    };

    let filtered = precursor.filter_fragments_by_intensity(0.0).unwrap();

    assert_eq!(filtered.fragment_intensity.len(), 2);
    assert_eq!(filtered.fragment_intensity, vec![100.0, 150.0]);
    assert_eq!(filtered.fragment_mz, vec![200.0, 400.0]);
    assert_eq!(filtered.fragment_mz_library, vec![199.8, 399.7]);
    assert_eq!(filtered.fragment_number, vec![1, 3]);
}

#[test]
fn test_filter_fragments_by_intensity_returns_none_when_empty() {
    let precursor = PrecursorQuantified {
        precursor_idx: 1,
        mz_library: 500.3,
        mz: 500.5,
        rt_library: 119.8,
        rt: 120.0,
        naa: 12,
        rank: 1,
        rt_observed: 121.5,
        fragment_mz_library: vec![199.8, 299.9],
        fragment_mz: vec![200.0, 300.0],
        fragment_intensity: vec![0.0, -5.0],
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

    let filtered = precursor.filter_fragments_by_intensity(0.0);
    assert!(filtered.is_none());
}

#[test]
fn test_filter_fragments_by_intensity_with_higher_threshold() {
    let precursor = PrecursorQuantified {
        precursor_idx: 1,
        mz_library: 500.3,
        mz: 500.5,
        rt_library: 119.8,
        rt: 120.0,
        naa: 12,
        rank: 1,
        rt_observed: 121.5,
        fragment_mz_library: vec![199.8, 299.9, 399.7],
        fragment_mz: vec![200.0, 300.0, 400.0],
        fragment_intensity: vec![50.0, 100.0, 150.0],
        fragment_cardinality: vec![1, 1, 1],
        fragment_charge: vec![1, 1, 1],
        fragment_loss_type: vec![0, 0, 0],
        fragment_number: vec![1, 2, 3],
        fragment_position: vec![1, 2, 3],
        fragment_type: vec![1, 1, 1],
        fragment_mz_observed: vec![200.1, 299.9, 399.8],
        fragment_correlation_observed: vec![0.95, 0.88, 0.92],
        fragment_mass_error_observed: vec![0.1, -0.1, 0.05],
    };

    let filtered = precursor.filter_fragments_by_intensity(75.0).unwrap();

    assert_eq!(filtered.fragment_intensity.len(), 2);
    assert_eq!(filtered.fragment_intensity, vec![100.0, 150.0]);
    assert_eq!(filtered.fragment_number, vec![2, 3]);
}
