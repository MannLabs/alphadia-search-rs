//! These tests cover the two competition criteria (shared ions and retention time), the
//! tolerance and threshold limits, the winner of a conflict, and the rejected inputs.

use super::algorithm::{compete_for_fragments, WindowBounds};
use numpy::ndarray::{Array4, ArrayView4};

/// One window that contains all candidates. Only RT and overlap decide the result.
fn single_window() -> Array4<f32> {
    Array4::from_shape_vec((1, 1, 1, 2), vec![0.0, 10_000.0]).unwrap()
}

/// Two windows, `[90, 110)` and `[190, 210)`.
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

/// `n` candidates that share the same ten ions.
fn shared_fragments(n: usize) -> Vec<f32> {
    (0..n).flat_map(|_| (100..110).map(|v| v as f32)).collect()
}

#[test]
fn test_fragment_overlap() {
    let cycle = single_window();

    // All ten ions are shared. The worse candidate loses.
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

    // One shared ion is less than the threshold.
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

    // No shared ions.
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
    // All candidates share their ions. RT alone decides which candidates compete.
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
    // The same six candidates, in reverse order (worst first).
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
    // The candidate order is reversed, so the expected mask is also reversed.
    assert_eq!(valid, vec![true, false, true, false, true, true]);
}

#[test]
fn test_priority_decides_the_winner_not_position() {
    // The second candidate has the better proba.
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
    // The two proba are equal, so the lower precursor_idx wins.
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
    // The fragments and the RT are the same, but the windows are different.
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
    // The stop index is greater than the length of fragment_mz.
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

    // The start index is greater than the stop index.
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
fn test_nan_precursor_mz_is_rejected() {
    let error = compete(
        &[500.0, f32::NAN],
        &[0, 1],
        &[0.1, 0.2],
        &[10.0, 10.0],
        &[0, 10],
        &[10, 20],
        &shared_fragments(2),
        single_window().view(),
    )
    .unwrap_err();
    assert!(error.contains("precursor_mz"), "{error}");
}

#[test]
fn test_nan_proba_is_rejected() {
    let error = compete(
        &[500.0, 500.0],
        &[0, 1],
        &[f64::NAN, 0.2],
        &[10.0, 10.0],
        &[0, 10],
        &[10, 20],
        &shared_fragments(2),
        single_window().view(),
    )
    .unwrap_err();
    assert!(error.contains("proba"), "{error}");
}

#[test]
fn test_nan_rt_observed_is_rejected() {
    let error = compete(
        &[500.0, 500.0],
        &[0, 1],
        &[0.1, 0.2],
        &[10.0, f32::NAN],
        &[0, 10],
        &[10, 20],
        &shared_fragments(2),
        single_window().view(),
    )
    .unwrap_err();
    assert!(error.contains("rt_observed"), "{error}");
}

#[test]
fn test_nan_fragment_mz_is_rejected() {
    let mut fragment_mz = shared_fragments(2);
    fragment_mz[3] = f32::NAN;
    let error = compete(
        &[500.0, 500.0],
        &[0, 1],
        &[0.1, 0.2],
        &[10.0, 10.0],
        &[0, 10],
        &[10, 20],
        &fragment_mz,
        single_window().view(),
    )
    .unwrap_err();
    assert!(error.contains("fragment_mz"), "{error}");
}

#[test]
fn test_nan_cycle_is_rejected() {
    let cycle = Array4::from_shape_vec((1, 1, 1, 2), vec![0.0, f32::NAN]).unwrap();
    let error = compete(
        &[500.0, 500.0],
        &[0, 1],
        &[0.1, 0.2],
        &[10.0, 10.0],
        &[0, 10],
        &[10, 20],
        &shared_fragments(2),
        cycle.view(),
    )
    .unwrap_err();
    assert!(error.contains("cycle"), "{error}");
}

#[test]
fn test_overlap_threshold_boundary() {
    let cycle = single_window();

    // Two shared ions. This is one less than the threshold.
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

    // Exactly at the threshold.
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

    // Exactly at the tolerance. The tolerance is exclusive.
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

    // Just inside the tolerance.
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

    // Three pairs at approximately 20 ppm. This is outside the 15 ppm tolerance.
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

    // The same pairs at approximately 10 ppm. This is inside the tolerance.
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
