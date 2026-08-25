//! Unit tests for the fragment competition port. Cases mirror
//! `tests/unit_tests/fragcomp/test_fragcomp.py` in the `alphadia` repo.

use super::algorithm::{compete_for_fragments, WindowBounds};
use numpy::ndarray::{Array4, ArrayView4};

/// A cycle with a single window spanning `[0, 10_000)`, so every candidate lands in
/// window 0 and only RT / fragment overlap decide the outcome.
fn single_window() -> Array4<f32> {
    Array4::from_shape_vec((1, 1, 1, 2), vec![0.0, 10_000.0]).unwrap()
}

/// A cycle with two windows, `[90, 110)` and `[190, 210)`.
fn two_windows() -> Array4<f32> {
    Array4::from_shape_vec((1, 2, 1, 2), vec![90.0, 110.0, 190.0, 210.0]).unwrap()
}

#[allow(clippy::too_many_arguments)]
fn compete(
    precursor_mz: &[f32],
    precursor_idx: &[i64],
    proba: &[f64],
    rt: &[f32],
    frag_start_idx: &[i64],
    frag_stop_idx: &[i64],
    fragment_mz: &[f32],
    cycle: ArrayView4<'_, f32>,
) -> Result<Vec<bool>, String> {
    let bounds = WindowBounds::from_cycle(cycle)?;
    compete_for_fragments(
        precursor_mz,
        precursor_idx,
        proba,
        rt,
        frag_start_idx,
        frag_stop_idx,
        fragment_mz,
        &bounds,
        3.0,
        15.0,
    )
}

/// `n` candidates in one window, all sharing the same ten fragment ions.
fn shared_fragments(n: usize) -> Vec<f32> {
    (0..n).flat_map(|_| (100..110).map(|v| v as f32)).collect()
}

#[test]
fn test_fragment_overlap() {
    let cycle = single_window();

    // All ten fragments shared -> the lower-priority candidate is invalidated.
    let valid = compete(
        &[500.0, 500.0],
        &[0, 1],
        &[0.1, 0.2],
        &[10.0, 10.0],
        &[0, 10],
        &[10, 20],
        &shared_fragments(2),
        cycle.view(),
    )
    .unwrap();
    assert_eq!(valid, vec![true, false]);

    // Only a single overlapping fragment -> below the threshold, both survive.
    let valid = compete(
        &[500.0, 500.0],
        &[0, 1],
        &[0.1, 0.2],
        &[10.0, 10.0],
        &[0, 10],
        &[10, 11],
        &(100..110)
            .map(|v| v as f32)
            .chain([100.0])
            .collect::<Vec<_>>(),
        cycle.view(),
    )
    .unwrap();
    assert_eq!(valid, vec![true, true]);

    // No shared fragments at all -> both survive.
    let valid = compete(
        &[500.0, 500.0],
        &[0, 1],
        &[0.1, 0.2],
        &[10.0, 10.0],
        &[0, 10],
        &[10, 20],
        &(100..110)
            .chain(200..210)
            .map(|v| v as f32)
            .collect::<Vec<_>>(),
        cycle.view(),
    )
    .unwrap();
    assert_eq!(valid, vec![true, true]);
}

#[test]
fn test_compete_for_fragments() {
    // Two DIA windows of three candidates each; every candidate shares all ten
    // fragments with every other, so only the RT tolerance decides who competes.
    let valid = compete(
        &[100.0, 100.0, 100.0, 200.0, 200.0, 200.0],
        &[0, 1, 2, 3, 4, 5],
        &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        &[10.0, 20.0, 20.0, 10.0, 10.0, 20.0],
        &[0, 10, 20, 30, 40, 50],
        &[10, 20, 30, 40, 50, 60],
        &shared_fragments(6),
        two_windows().view(),
    )
    .unwrap();
    assert_eq!(valid, vec![true, true, false, true, false, true]);
}

#[test]
fn test_result_order_is_independent_of_input_order() {
    // Same six candidates as above, passed worst-priority first. The mask must come
    // back in input order with the same candidates surviving.
    let valid = compete(
        &[200.0, 200.0, 200.0, 100.0, 100.0, 100.0],
        &[5, 4, 3, 2, 1, 0],
        &[0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        &[20.0, 10.0, 10.0, 20.0, 20.0, 10.0],
        &[0, 10, 20, 30, 40, 50],
        &[10, 20, 30, 40, 50, 60],
        &shared_fragments(6),
        two_windows().view(),
    )
    .unwrap();
    // reversed candidates 5,4,3,2,1,0 -> reversed mask of [T,T,F,T,F,T]
    assert_eq!(valid, vec![true, false, true, false, true, true]);
}

#[test]
fn test_priority_decides_the_winner_not_position() {
    // The candidate passed second has the better (lower) proba, so it wins.
    let valid = compete(
        &[500.0, 500.0],
        &[0, 1],
        &[0.9, 0.1],
        &[10.0, 10.0],
        &[0, 10],
        &[10, 20],
        &shared_fragments(2),
        single_window().view(),
    )
    .unwrap();
    assert_eq!(valid, vec![false, true]);
}

#[test]
fn test_proba_ties_broken_by_precursor_idx() {
    // Equal proba -> the lower precursor_idx wins regardless of input position.
    let valid = compete(
        &[500.0, 500.0],
        &[7, 3],
        &[0.5, 0.5],
        &[10.0, 10.0],
        &[0, 10],
        &[10, 20],
        &shared_fragments(2),
        single_window().view(),
    )
    .unwrap();
    assert_eq!(valid, vec![false, true]);
}

#[test]
fn test_candidates_in_different_windows_do_not_compete() {
    // Identical fragments and RT, but the precursors fall in different windows.
    let valid = compete(
        &[100.0, 200.0],
        &[0, 1],
        &[0.1, 0.2],
        &[10.0, 10.0],
        &[0, 10],
        &[10, 20],
        &shared_fragments(2),
        two_windows().view(),
    )
    .unwrap();
    assert_eq!(valid, vec![true, true]);
}

#[test]
fn test_compete_for_fragments_empty() {
    let valid = compete(&[], &[], &[], &[], &[], &[], &[], single_window().view()).unwrap();
    assert!(valid.is_empty());
}

#[test]
fn test_compete_for_fragments_mismatched_lengths() {
    let result = compete(
        &[500.0, 500.0],
        &[0, 1],
        &[0.1],
        &[10.0, 10.0],
        &[0, 1],
        &[1, 2],
        &[100.0],
        single_window().view(),
    );
    assert!(result.is_err());
}

#[test]
fn test_compete_for_fragments_out_of_bounds_fragment_range() {
    // frag_stop_idx=11 exceeds the fragment_mz array (length 10).
    let result = compete(
        &[500.0, 500.0],
        &[0, 1],
        &[0.1, 0.2],
        &[10.0, 10.0],
        &[0, 1],
        &[1, 11],
        &[100.0; 10],
        single_window().view(),
    );
    assert!(result.is_err());

    // frag_start_idx > frag_stop_idx is also invalid.
    let result = compete(
        &[500.0, 500.0],
        &[0, 1],
        &[0.1, 0.2],
        &[10.0, 10.0],
        &[5, 0],
        &[3, 1],
        &[100.0; 10],
        single_window().view(),
    );
    assert!(result.is_err());
}

#[test]
fn test_overlap_threshold_boundary() {
    let cycle = single_window();

    // Sharing MIN_OVERLAPPING_FRAGMENTS - 1 = 2 fragments -> both survive.
    let valid = compete(
        &[500.0, 500.0],
        &[0, 1],
        &[0.1, 0.2],
        &[10.0, 10.0],
        &[0, 3],
        &[3, 6],
        &[100.0, 101.0, 102.0, 100.0, 101.0, 999.0],
        cycle.view(),
    )
    .unwrap();
    assert_eq!(valid, vec![true, true]);

    // Sharing exactly MIN_OVERLAPPING_FRAGMENTS = 3 -> the loser is invalidated.
    let valid = compete(
        &[500.0, 500.0],
        &[0, 1],
        &[0.1, 0.2],
        &[10.0, 10.0],
        &[0, 3],
        &[3, 6],
        &[100.0, 101.0, 102.0, 100.0, 101.0, 102.0],
        cycle.view(),
    )
    .unwrap();
    assert_eq!(valid, vec![true, false]);
}

#[test]
fn test_rt_tolerance_boundary() {
    let cycle = single_window();
    let fragment_mz = shared_fragments(2);

    // Delta RT exactly at the tolerance -> excluded (strict `<`), both survive.
    let valid = compete(
        &[500.0, 500.0],
        &[0, 1],
        &[0.1, 0.2],
        &[10.0, 13.0],
        &[0, 10],
        &[10, 20],
        &fragment_mz,
        cycle.view(),
    )
    .unwrap();
    assert_eq!(valid, vec![true, true]);

    // Just inside the tolerance -> the lower-priority candidate is invalidated.
    let valid = compete(
        &[500.0, 500.0],
        &[0, 1],
        &[0.1, 0.2],
        &[10.0, 12.9],
        &[0, 10],
        &[10, 20],
        &fragment_mz,
        cycle.view(),
    )
    .unwrap();
    assert_eq!(valid, vec![true, false]);
}

#[test]
fn test_mass_tolerance_boundary() {
    let cycle = single_window();
    let bases = [1_000_000.0_f32, 2_000_000.0, 3_000_000.0];

    // Three fragment pairs ~20 ppm apart (above the 15 ppm tolerance) -> no overlap.
    let above: Vec<f32> = bases
        .iter()
        .copied()
        .chain(bases.iter().map(|&b| b + b * 20.0 / 1e6))
        .collect();
    let valid = compete(
        &[500.0, 500.0],
        &[0, 1],
        &[0.1, 0.2],
        &[10.0, 10.0],
        &[0, 3],
        &[3, 6],
        &above,
        cycle.view(),
    )
    .unwrap();
    assert_eq!(valid, vec![true, true]);

    // Same pairs ~10 ppm apart (below the tolerance) -> all three count as overlaps.
    let below: Vec<f32> = bases
        .iter()
        .copied()
        .chain(bases.iter().map(|&b| b + b * 10.0 / 1e6))
        .collect();
    let valid = compete(
        &[500.0, 500.0],
        &[0, 1],
        &[0.1, 0.2],
        &[10.0, 10.0],
        &[0, 3],
        &[3, 6],
        &below,
        cycle.view(),
    )
    .unwrap();
    assert_eq!(valid, vec![true, false]);
}
