use super::*;
use crate::precursor_quantified::PrecursorQuantified;

fn test_precursor_quantified() -> PrecursorQuantified {
    PrecursorQuantified {
        precursor_idx: 0,
        mz_library: 499.8,
        mz: 500.0,
        rt_library: 99.5,
        rt: 100.0,
        naa: 10,
        rank: 1,
        rt_observed: 102.5,
        fragment_mz_library: vec![199.9, 299.8, 399.9],
        fragment_mz: vec![200.0, 300.0, 400.0],
        fragment_intensity: vec![10.0, 15.0, 20.0],
        fragment_cardinality: vec![1, 1, 1],
        fragment_charge: vec![1, 1, 1],
        fragment_loss_type: vec![0, 0, 0],
        fragment_number: vec![1, 2, 3],
        fragment_position: vec![1, 2, 3],
        fragment_type: vec![1, 1, 1],
        fragment_mz_observed: vec![200.1, 299.9, 400.2],
        fragment_correlation_observed: vec![0.95, 0.87, 0.92],
        fragment_mass_error_observed: vec![0.1, -0.1, 0.2],
    }
}

#[test]
fn test_new() {
    let spec_lib = SpecLibFlatQuantified::new();
    assert_eq!(spec_lib.num_precursors(), 0);
    assert_eq!(spec_lib.num_fragments(), 0);
}

#[test]
fn test_from_precursor_quantified_vec_empty() {
    let spec_lib = SpecLibFlatQuantified::from_precursor_quantified_vec(vec![]);
    assert_eq!(spec_lib.num_precursors(), 0);
    assert_eq!(spec_lib.num_fragments(), 0);
}

#[test]
fn test_from_precursor_quantified_vec_single() {
    let precursor = test_precursor_quantified();
    let spec_lib = SpecLibFlatQuantified::from_precursor_quantified_vec(vec![precursor]);

    // Test basic functionality since we removed the getter methods
    assert_eq!(spec_lib.num_precursors(), 1);
    assert_eq!(spec_lib.num_fragments(), 3);
}

#[test]
fn test_from_precursor_quantified_vec_multiple() {
    let mut precursor1 = test_precursor_quantified();
    precursor1.precursor_idx = 5;

    let mut precursor2 = test_precursor_quantified();
    precursor2.precursor_idx = 2;
    precursor2.mz_library = 599.7;
    precursor2.mz = 600.0;
    precursor2.fragment_mz_library = vec![249.8, 349.9];
    precursor2.fragment_mz = vec![250.0, 350.0];
    precursor2.fragment_intensity = vec![5.0, 8.0];
    precursor2.fragment_mz_observed = vec![250.2, 349.8];
    precursor2.fragment_correlation_observed = vec![0.88, 0.93];
    precursor2.fragment_mass_error_observed = vec![0.2, -0.2];
    // Update other fragment vectors to match length
    precursor2.fragment_cardinality = vec![1, 1];
    precursor2.fragment_charge = vec![1, 1];
    precursor2.fragment_loss_type = vec![0, 0];
    precursor2.fragment_number = vec![1, 2];
    precursor2.fragment_position = vec![1, 2];
    precursor2.fragment_type = vec![1, 1];

    let spec_lib =
        SpecLibFlatQuantified::from_precursor_quantified_vec(vec![precursor1, precursor2]);

    // Test basic functionality
    assert_eq!(spec_lib.num_precursors(), 2);
    assert_eq!(spec_lib.num_fragments(), 5); // 3 + 2
}

#[test]
fn test_basic_functionality() {
    let mut precursor1 = test_precursor_quantified();
    precursor1.precursor_idx = 10;

    let mut precursor2 = test_precursor_quantified();
    precursor2.precursor_idx = 5;
    precursor2.mz = 600.0;

    let spec_lib =
        SpecLibFlatQuantified::from_precursor_quantified_vec(vec![precursor1, precursor2]);

    // Test basic functionality
    assert_eq!(spec_lib.num_precursors(), 2);
    assert_eq!(spec_lib.num_fragments(), 6); // 3 + 3
}

#[test]
fn test_quantified_data_structure() {
    let precursor = test_precursor_quantified();
    let spec_lib = SpecLibFlatQuantified::from_precursor_quantified_vec(vec![precursor]);

    // Test that the structure is created correctly
    assert_eq!(spec_lib.num_precursors(), 1);
    assert_eq!(spec_lib.num_fragments(), 3);
}

#[test]
fn test_fragment_precursor_columns() {
    let mut precursor1 = test_precursor_quantified();
    precursor1.precursor_idx = 5;
    precursor1.rank = 2;

    let mut precursor2 = test_precursor_quantified();
    precursor2.precursor_idx = 10;
    precursor2.rank = 1;
    precursor2.fragment_mz_library = vec![249.8, 349.9];
    precursor2.fragment_mz = vec![250.0, 350.0];
    precursor2.fragment_intensity = vec![5.0, 8.0];
    precursor2.fragment_mz_observed = vec![250.2, 349.8];
    precursor2.fragment_correlation_observed = vec![0.88, 0.93];
    precursor2.fragment_mass_error_observed = vec![0.2, -0.2];
    precursor2.fragment_cardinality = vec![1, 1];
    precursor2.fragment_charge = vec![1, 1];
    precursor2.fragment_loss_type = vec![0, 0];
    precursor2.fragment_number = vec![1, 2];
    precursor2.fragment_position = vec![1, 2];
    precursor2.fragment_type = vec![1, 1];

    let spec_lib =
        SpecLibFlatQuantified::from_precursor_quantified_vec(vec![precursor1, precursor2]);

    assert_eq!(spec_lib.num_precursors(), 2);
    assert_eq!(spec_lib.num_fragments(), 5); // 3 + 2

    // Check that fragment_precursor_idx and fragment_precursor_rank are correctly expanded
    assert_eq!(spec_lib.fragment_precursor_idx.len(), 5);
    assert_eq!(spec_lib.fragment_precursor_rank.len(), 5);

    // Since precursors are sorted by idx (5, 10), fragments should be arranged accordingly
    // First 3 fragments belong to precursor with idx=5, rank=2
    assert_eq!(spec_lib.fragment_precursor_idx[0], 5);
    assert_eq!(spec_lib.fragment_precursor_idx[1], 5);
    assert_eq!(spec_lib.fragment_precursor_idx[2], 5);
    assert_eq!(spec_lib.fragment_precursor_rank[0], 2);
    assert_eq!(spec_lib.fragment_precursor_rank[1], 2);
    assert_eq!(spec_lib.fragment_precursor_rank[2], 2);

    // Last 2 fragments belong to precursor with idx=10, rank=1
    assert_eq!(spec_lib.fragment_precursor_idx[3], 10);
    assert_eq!(spec_lib.fragment_precursor_idx[4], 10);
    assert_eq!(spec_lib.fragment_precursor_rank[3], 1);
    assert_eq!(spec_lib.fragment_precursor_rank[4], 1);
}
