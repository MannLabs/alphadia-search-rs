use super::*;
use crate::speclib_flat::filter_fragments;

fn test_precursor() -> Precursor {
    Precursor {
        idx: 0,
        mz: 500.0,
        rt: 100.0,
        fragment_mz: vec![200.0, 300.0, 400.0, 500.0, 600.0],
        fragment_intensity: vec![0.0, 10.0, 5.0, 20.0, 0.0],
    }
}

#[test]
fn test_no_filtering() {
    let precursor = test_precursor();
    let (mz, intensity) = filter_fragments(
        &precursor.fragment_mz,
        &precursor.fragment_intensity,
        false,
        usize::MAX,
    );

    assert_eq!(mz, vec![200.0, 300.0, 400.0, 500.0, 600.0]);
    assert_eq!(intensity, vec![0.0, 10.0, 5.0, 20.0, 0.0]);
}

#[test]
fn test_non_zero_filtering() {
    let precursor = test_precursor();
    let (mz, intensity) = filter_fragments(
        &precursor.fragment_mz,
        &precursor.fragment_intensity,
        true,
        usize::MAX,
    );

    assert_eq!(mz, vec![300.0, 400.0, 500.0]);
    assert_eq!(intensity, vec![10.0, 5.0, 20.0]);
    assert!(intensity.iter().all(|&i| i > 0.0));
}

#[test]
fn test_top_k_selection() {
    let precursor = test_precursor();
    let (mz, intensity) = filter_fragments(
        &precursor.fragment_mz,
        &precursor.fragment_intensity,
        false,
        2,
    );

    // Top 2: intensity 20.0 (mz 500.0) and 10.0 (mz 300.0), in original order
    assert_eq!(mz, vec![300.0, 500.0]);
    assert_eq!(intensity, vec![10.0, 20.0]);
    assert!(mz.windows(2).all(|w| w[0] <= w[1])); // Maintains order
}

#[test]
fn test_combined_filtering() {
    let precursor = test_precursor();
    let (mz, intensity) = filter_fragments(
        &precursor.fragment_mz,
        &precursor.fragment_intensity,
        true,
        2,
    );

    // Non-zero: [300.0->10.0, 400.0->5.0, 500.0->20.0], top 2: [300.0->10.0, 500.0->20.0]
    assert_eq!(mz, vec![300.0, 500.0]);
    assert_eq!(intensity, vec![10.0, 20.0]);
    assert!(intensity.iter().all(|&i| i > 0.0));
    assert!(mz.windows(2).all(|w| w[0] <= w[1]));
}

#[test]
fn test_ordering_preservation() {
    let precursor = Precursor {
        idx: 0,
        mz: 500.0,
        rt: 100.0,
        fragment_mz: vec![600.0, 200.0, 800.0, 100.0, 400.0],
        fragment_intensity: vec![15.0, 25.0, 5.0, 30.0, 20.0],
    };

    let (mz, intensity) = filter_fragments(
        &precursor.fragment_mz,
        &precursor.fragment_intensity,
        false,
        3,
    );

    // Top 3: 100.0->30.0, 200.0->25.0, 400.0->20.0 in original index order
    assert_eq!(mz, vec![200.0, 100.0, 400.0]);
    assert_eq!(intensity, vec![25.0, 30.0, 20.0]);

    // Verify top-k correctness
    let mut sorted_intensity = intensity.clone();
    sorted_intensity.sort_by(|a, b| b.partial_cmp(a).unwrap());
    assert_eq!(sorted_intensity, vec![30.0, 25.0, 20.0]);
}

#[test]
fn test_top_k_larger_than_available() {
    let small_precursor = Precursor {
        idx: 0,
        mz: 500.0,
        rt: 100.0,
        fragment_mz: vec![300.0, 400.0],
        fragment_intensity: vec![10.0, 5.0],
    };

    let (mz, intensity) = filter_fragments(
        &small_precursor.fragment_mz,
        &small_precursor.fragment_intensity,
        false,
        5,
    );
    assert_eq!(mz, vec![300.0, 400.0]);
    assert_eq!(intensity, vec![10.0, 5.0]);
}

#[test]
fn test_all_zero_intensities_filtered() {
    let zero_precursor = Precursor {
        idx: 0,
        mz: 500.0,
        rt: 100.0,
        fragment_mz: vec![300.0, 400.0],
        fragment_intensity: vec![0.0, 0.0],
    };

    let (mz, intensity) = filter_fragments(
        &zero_precursor.fragment_mz,
        &zero_precursor.fragment_intensity,
        true,
        usize::MAX,
    );
    assert_eq!(mz, Vec::<f32>::new());
    assert_eq!(intensity, Vec::<f32>::new());
}
