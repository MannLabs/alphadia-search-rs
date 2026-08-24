//! Unit tests for the fragment competition port. Cases mirror
//! `tests/unit_tests/fragcomp/test_fragcomp.py` in the `alphadia` repo.

use super::algorithm::compete_for_fragments;

#[test]
fn test_fragment_overlap_via_compete() {
    // Two candidates, same window, close RT, all ten fragments shared -> the
    // second is invalidated.
    let valid = compete_for_fragments(
        &[0, 0],
        &[10.0, 10.0],
        &[0, 10],
        &[10, 20],
        &(0..2)
            .flat_map(|_| (100..110).map(|v| v as f32))
            .collect::<Vec<_>>(),
        3.0,
        15.0,
    )
    .unwrap();
    assert_eq!(valid, vec![true, false]);

    // Only a single overlapping fragment -> below the threshold, both survive.
    let valid = compete_for_fragments(
        &[0, 0],
        &[10.0, 10.0],
        &[0, 10],
        &[10, 11],
        &(100..110)
            .map(|v| v as f32)
            .chain([100.0])
            .collect::<Vec<_>>(),
        3.0,
        15.0,
    )
    .unwrap();
    assert_eq!(valid, vec![true, true]);

    // No shared fragments at all -> both survive.
    let valid = compete_for_fragments(
        &[0, 0],
        &[10.0, 10.0],
        &[0, 10],
        &[10, 20],
        &(100..110)
            .map(|v| v as f32)
            .chain((200..210).map(|v| v as f32))
            .collect::<Vec<_>>(),
        3.0,
        15.0,
    )
    .unwrap();
    assert_eq!(valid, vec![true, true]);
}

#[test]
fn test_compete_for_fragments() {
    // Two DIA windows of three candidates each; every candidate shares all ten
    // fragments with every other candidate, so only the RT tolerance decides who
    // competes.
    let window_idx = [0, 0, 0, 1, 1, 1];
    let rt = [10.0, 20.0, 20.0, 10.0, 10.0, 20.0];
    let frag_start_idx = [0, 10, 20, 30, 40, 50];
    let frag_stop_idx = [10, 20, 30, 40, 50, 60];
    let fragment_mz: Vec<f32> = (0..6).flat_map(|_| (100..110).map(|v| v as f32)).collect();

    let valid = compete_for_fragments(
        &window_idx,
        &rt,
        &frag_start_idx,
        &frag_stop_idx,
        &fragment_mz,
        3.0,
        15.0,
    )
    .unwrap();

    assert_eq!(valid, vec![true, true, false, true, false, true]);
}

#[test]
fn test_compete_for_fragments_empty() {
    let valid = compete_for_fragments(&[], &[], &[], &[], &[], 3.0, 15.0).unwrap();
    assert!(valid.is_empty());
}

#[test]
fn test_compete_for_fragments_mismatched_lengths() {
    let result = compete_for_fragments(&[0, 0], &[10.0], &[0, 1], &[1, 2], &[100.0], 3.0, 15.0);
    assert!(result.is_err());
}
