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

#[test]
fn test_compete_for_fragments_out_of_bounds_fragment_range() {
    // frag_stop_idx=11 exceeds the fragment_mz array (length 10).
    let result = compete_for_fragments(
        &[0, 0],
        &[10.0, 10.0],
        &[0, 1],
        &[1, 11],
        &[100.0; 10],
        3.0,
        15.0,
    );
    assert!(result.is_err());

    // frag_start_idx > frag_stop_idx is also invalid.
    let result = compete_for_fragments(
        &[0, 0],
        &[10.0, 10.0],
        &[5, 0],
        &[3, 1],
        &[100.0; 10],
        3.0,
        15.0,
    );
    assert!(result.is_err());
}

#[test]
fn test_overlap_threshold_boundary() {
    // Sharing MIN_OVERLAPPING_FRAGMENTS - 1 = 2 fragments -> below the threshold,
    // both survive.
    let valid = compete_for_fragments(
        &[0, 0],
        &[10.0, 10.0],
        &[0, 3],
        &[3, 6],
        &[100.0, 101.0, 102.0, 100.0, 101.0, 999.0],
        3.0,
        15.0,
    )
    .unwrap();
    assert_eq!(valid, vec![true, true]);

    // Sharing exactly MIN_OVERLAPPING_FRAGMENTS = 3 fragments -> at the threshold,
    // the second candidate is invalidated.
    let valid = compete_for_fragments(
        &[0, 0],
        &[10.0, 10.0],
        &[0, 3],
        &[3, 6],
        &[100.0, 101.0, 102.0, 100.0, 101.0, 102.0],
        3.0,
        15.0,
    )
    .unwrap();
    assert_eq!(valid, vec![true, false]);
}

#[test]
fn test_rt_tolerance_boundary() {
    let fragment_mz: Vec<f32> = (0..2).flat_map(|_| (100..110).map(|v| v as f32)).collect();

    // Delta RT exactly at the tolerance -> excluded (strict `<`), both survive
    // even though fragments fully overlap.
    let valid = compete_for_fragments(
        &[0, 0],
        &[10.0, 13.0],
        &[0, 10],
        &[10, 20],
        &fragment_mz,
        3.0,
        15.0,
    )
    .unwrap();
    assert_eq!(valid, vec![true, true]);

    // Just inside the tolerance -> the second candidate is invalidated.
    let valid = compete_for_fragments(
        &[0, 0],
        &[10.0, 12.9],
        &[0, 10],
        &[10, 20],
        &fragment_mz,
        3.0,
        15.0,
    )
    .unwrap();
    assert_eq!(valid, vec![true, false]);
}

#[test]
fn test_mass_tolerance_boundary() {
    // Three fragment pairs each ~20 ppm apart (above the 15 ppm tolerance) -> no
    // overlap counted, both survive.
    let bases = [1_000_000.0_f32, 2_000_000.0, 3_000_000.0];
    let above: Vec<f32> = bases
        .iter()
        .copied()
        .chain(bases.iter().map(|&b| b + b * 20.0 / 1e6))
        .collect();
    let valid =
        compete_for_fragments(&[0, 0], &[10.0, 10.0], &[0, 3], &[3, 6], &above, 3.0, 15.0).unwrap();
    assert_eq!(valid, vec![true, true]);

    // Same three pairs, ~10 ppm apart (below the tolerance) -> all three count as
    // overlaps, so the second candidate is invalidated.
    let below: Vec<f32> = bases
        .iter()
        .copied()
        .chain(bases.iter().map(|&b| b + b * 10.0 / 1e6))
        .collect();
    let valid =
        compete_for_fragments(&[0, 0], &[10.0, 10.0], &[0, 3], &[3, 6], &below, 3.0, 15.0).unwrap();
    assert_eq!(valid, vec![true, false]);
}
