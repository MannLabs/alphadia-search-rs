use super::*;
use crate::speclib_flat::filter_sort_fragments;

fn test_precursor() -> Precursor {
    Precursor {
        precursor_idx: 0,
        mz: 500.0,
        mz_library: 500.5,
        rt: 100.0,
        rt_library: 100.5,
        naa: 10,
        fragment_mz: vec![200.0, 300.0, 400.0, 500.0, 600.0],
        fragment_mz_library: vec![200.5, 300.5, 400.5, 500.5, 600.5],
        fragment_intensity: vec![0.0, 10.0, 5.0, 20.0, 0.0],
        fragment_cardinality: vec![1, 1, 1, 1, 1],
        fragment_charge: vec![1, 1, 1, 1, 1],
        fragment_loss_type: vec![0, 0, 0, 0, 0],
        fragment_number: vec![1, 2, 3, 4, 5],
        fragment_position: vec![1, 2, 3, 4, 5],
        fragment_type: vec![1, 1, 1, 1, 1],
    }
}

#[test]
fn test_no_filtering() {
    let precursor = test_precursor();
    let (mz, _mz_library, intensity, _, _, _, _, _, _) = filter_sort_fragments(
        &precursor.fragment_mz,
        &precursor.fragment_mz_library,
        &precursor.fragment_intensity,
        &precursor.fragment_cardinality,
        &precursor.fragment_charge,
        &precursor.fragment_loss_type,
        &precursor.fragment_number,
        &precursor.fragment_position,
        &precursor.fragment_type,
        false,
        false, // Don't filter Y1 ions in basic tests
        usize::MAX,
    );

    assert_eq!(mz, vec![200.0, 300.0, 400.0, 500.0, 600.0]);
    assert_eq!(intensity, vec![0.0, 10.0, 5.0, 20.0, 0.0]);
}

#[test]
fn test_non_zero_filtering() {
    let precursor = test_precursor();
    let (mz, _mz_library, intensity, _, _, _, _, _, _) = filter_sort_fragments(
        &precursor.fragment_mz,
        &precursor.fragment_mz_library,
        &precursor.fragment_intensity,
        &precursor.fragment_cardinality,
        &precursor.fragment_charge,
        &precursor.fragment_loss_type,
        &precursor.fragment_number,
        &precursor.fragment_position,
        &precursor.fragment_type,
        true,
        false, // Don't filter Y1 ions in basic tests
        usize::MAX,
    );

    assert_eq!(mz, vec![300.0, 400.0, 500.0]);
    assert_eq!(intensity, vec![10.0, 5.0, 20.0]);
    assert!(intensity.iter().all(|&i| i > 0.0));
}

#[test]
fn test_top_k_selection() {
    let precursor = test_precursor();
    let (mz, _mz_library, intensity, _, _, _, _, _, _) = filter_sort_fragments(
        &precursor.fragment_mz,
        &precursor.fragment_mz_library,
        &precursor.fragment_intensity,
        &precursor.fragment_cardinality,
        &precursor.fragment_charge,
        &precursor.fragment_loss_type,
        &precursor.fragment_number,
        &precursor.fragment_position,
        &precursor.fragment_type,
        false,
        false, // Don't filter Y1 ions in basic tests
        2,
    );

    // Top 2: intensity 20.0 (mz 500.0) and 10.0 (mz 300.0), sorted by fragment_mz ascending
    assert_eq!(mz, vec![300.0, 500.0]);
    assert_eq!(intensity, vec![10.0, 20.0]);
    assert!(mz.windows(2).all(|w| w[0] <= w[1])); // Sorted by fragment_mz ascending
}

#[test]
fn test_combined_filtering() {
    let precursor = test_precursor();
    let (mz, _mz_library, intensity, _, _, _, _, _, _) = filter_sort_fragments(
        &precursor.fragment_mz,
        &precursor.fragment_mz_library,
        &precursor.fragment_intensity,
        &precursor.fragment_cardinality,
        &precursor.fragment_charge,
        &precursor.fragment_loss_type,
        &precursor.fragment_number,
        &precursor.fragment_position,
        &precursor.fragment_type,
        true,
        false,
        2,
    );

    // Non-zero: [300.0->10.0, 400.0->5.0, 500.0->20.0], top 2: [300.0->10.0, 500.0->20.0]
    assert_eq!(mz, vec![300.0, 500.0]);
    assert_eq!(intensity, vec![10.0, 20.0]);
    assert!(intensity.iter().all(|&i| i > 0.0));
    assert!(mz.windows(2).all(|w| w[0] <= w[1])); // Sorted by fragment_mz ascending
}

#[test]
fn test_ordering_preservation() {
    let precursor = Precursor {
        precursor_idx: 0,
        mz: 500.0,
        mz_library: 500.5,
        rt: 100.0,
        rt_library: 100.5,
        naa: 12,
        fragment_mz: vec![600.0, 200.0, 800.0, 100.0, 400.0],
        fragment_mz_library: vec![600.5, 200.5, 800.5, 100.5, 400.5],
        fragment_intensity: vec![15.0, 25.0, 5.0, 30.0, 20.0],
        fragment_cardinality: vec![1, 1, 1, 1, 1],
        fragment_charge: vec![1, 1, 1, 1, 1],
        fragment_loss_type: vec![0, 0, 0, 0, 0],
        fragment_number: vec![1, 2, 3, 4, 5],
        fragment_position: vec![1, 2, 3, 4, 5],
        fragment_type: vec![1, 1, 1, 1, 1],
    };

    let (mz, _mz_library, intensity, _, _, _, _, _, _) = filter_sort_fragments(
        &precursor.fragment_mz,
        &precursor.fragment_mz_library,
        &precursor.fragment_intensity,
        &precursor.fragment_cardinality,
        &precursor.fragment_charge,
        &precursor.fragment_loss_type,
        &precursor.fragment_number,
        &precursor.fragment_position,
        &precursor.fragment_type,
        false,
        false,
        3,
    );

    // Top 3: 100.0->30.0, 200.0->25.0, 400.0->20.0 sorted by fragment_mz ascending
    assert_eq!(mz, vec![100.0, 200.0, 400.0]);
    assert_eq!(intensity, vec![30.0, 25.0, 20.0]);

    // Verify top-k correctness
    let mut sorted_intensity = intensity.clone();
    sorted_intensity.sort_by(|a, b| b.partial_cmp(a).unwrap());
    assert_eq!(sorted_intensity, vec![30.0, 25.0, 20.0]);
}

#[test]
fn test_top_k_larger_than_available() {
    let small_precursor = Precursor {
        precursor_idx: 0,
        mz: 500.0,
        mz_library: 500.5,
        rt: 100.0,
        rt_library: 100.5,
        naa: 8,
        fragment_mz: vec![300.0, 400.0],
        fragment_mz_library: vec![300.5, 400.5],
        fragment_intensity: vec![10.0, 5.0],
        fragment_cardinality: vec![1, 1],
        fragment_charge: vec![1, 1],
        fragment_loss_type: vec![0, 0],
        fragment_number: vec![1, 2],
        fragment_position: vec![1, 2],
        fragment_type: vec![1, 1],
    };

    let (mz, _mz_library, intensity, _, _, _, _, _, _) = filter_sort_fragments(
        &small_precursor.fragment_mz,
        &small_precursor.fragment_mz_library,
        &small_precursor.fragment_intensity,
        &small_precursor.fragment_cardinality,
        &small_precursor.fragment_charge,
        &small_precursor.fragment_loss_type,
        &small_precursor.fragment_number,
        &small_precursor.fragment_position,
        &small_precursor.fragment_type,
        false,
        false,
        5,
    );
    assert_eq!(mz, vec![300.0, 400.0]);
    assert_eq!(intensity, vec![10.0, 5.0]);
}

#[test]
fn test_all_zero_intensities_filtered() {
    let zero_precursor = Precursor {
        precursor_idx: 0,
        mz: 500.0,
        mz_library: 500.5,
        rt: 100.0,
        rt_library: 100.5,
        naa: 6,
        fragment_mz: vec![300.0, 400.0],
        fragment_mz_library: vec![300.5, 400.5],
        fragment_intensity: vec![0.0, 0.0],
        fragment_cardinality: vec![1, 1],
        fragment_charge: vec![1, 1],
        fragment_loss_type: vec![0, 0],
        fragment_number: vec![1, 2],
        fragment_position: vec![1, 2],
        fragment_type: vec![1, 1],
    };

    let (mz, _mz_library, intensity, _, _, _, _, _, _) = filter_sort_fragments(
        &zero_precursor.fragment_mz,
        &zero_precursor.fragment_mz_library,
        &zero_precursor.fragment_intensity,
        &zero_precursor.fragment_cardinality,
        &zero_precursor.fragment_charge,
        &zero_precursor.fragment_loss_type,
        &zero_precursor.fragment_number,
        &zero_precursor.fragment_position,
        &zero_precursor.fragment_type,
        true,
        false,
        usize::MAX,
    );
    assert_eq!(mz, Vec::<f32>::new());
    assert_eq!(intensity, Vec::<f32>::new());
}
