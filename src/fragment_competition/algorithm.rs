//! Fragment competition sweep. Port of
//! `alphadia/fragcomp/fragcomp.py::_compete_for_fragments` /
//! `_get_fragment_overlap`.
//!
//! Candidates are grouped into disjoint DIA isolation windows (the caller sorts all
//! arrays by `(window_idx, proba, precursor_idx)` first, so each window is a
//! contiguous slice) and each window is swept independently and in parallel. Within a
//! window the sweep is sequential and order-sensitive: once a candidate is
//! invalidated it can no longer invalidate others, so among two conflicting
//! candidates the one that comes *first* in the caller's sort order survives. The
//! caller sorts by ascending `proba`, so the lowest-probability candidate in a
//! conflicting pair is processed first and wins.

use rayon::prelude::*;

/// Two candidates compete for the same fragment signal once they share this many
/// fragment ions within the mass tolerance.
const MIN_OVERLAPPING_FRAGMENTS: usize = 3;

/// Count fragment ions in `frag_mz_1` that match one in `frag_mz_2` within
/// `mass_tol_ppm` (ppm computed relative to the `frag_mz_1` ion, matching the
/// original numpy broadcast).
fn fragment_overlap(frag_mz_1: &[f32], frag_mz_2: &[f32], mass_tol_ppm: f32) -> usize {
    frag_mz_1
        .iter()
        .map(|&mz1| {
            frag_mz_2
                .iter()
                .filter(|&&mz2| (mz1 - mz2).abs() / mz1 * 1e6 < mass_tol_ppm)
                .count()
        })
        .sum()
}

/// Sweep one DIA window and return its `valid` mask.
///
/// `rt`, `frag_start_idx` and `frag_stop_idx` are the window's slice of the
/// full per-PSM arrays, in the caller's priority order. `fragment_mz` is the full
/// (not window-local) fragment array, since `frag_start_idx`/`frag_stop_idx` are
/// offsets into it.
fn compete_within_window(
    rt: &[f32],
    frag_start_idx: &[i64],
    frag_stop_idx: &[i64],
    fragment_mz: &[f32],
    rt_tol_seconds: f32,
    mass_tol_ppm: f32,
) -> Vec<bool> {
    let n = rt.len();
    let mut valid = vec![true; n];

    for i in 0..n {
        if !valid[i] {
            continue;
        }
        let frags_i = &fragment_mz[frag_start_idx[i] as usize..frag_stop_idx[i] as usize];
        for j in 0..n {
            if i == j || !valid[j] {
                continue;
            }
            if (rt[i] - rt[j]).abs() >= rt_tol_seconds {
                continue;
            }
            let frags_j = &fragment_mz[frag_start_idx[j] as usize..frag_stop_idx[j] as usize];
            if fragment_overlap(frags_i, frags_j, mass_tol_ppm) >= MIN_OVERLAPPING_FRAGMENTS {
                valid[j] = false;
            }
        }
    }

    valid
}

/// Contiguous `(start, stop)` ranges of equal `window_idx` values. `window_idx` must
/// already be sorted ascending.
fn contiguous_window_bounds(window_idx: &[i64]) -> Vec<(usize, usize)> {
    let n = window_idx.len();
    if n == 0 {
        return Vec::new();
    }

    let mut bounds = Vec::new();
    let mut start = 0;
    for i in 1..n {
        if window_idx[i] != window_idx[start] {
            bounds.push((start, i));
            start = i;
        }
    }
    bounds.push((start, n));
    bounds
}

/// Run fragment competition over all PSMs and return a `valid` mask of the same
/// length and order as the inputs.
///
/// All arrays must already be sorted by `(window_idx, proba, precursor_idx)`
/// ascending (see module docs for why the order matters). `fragment_mz` is the full
/// fragment array; `frag_start_idx`/`frag_stop_idx` are offsets into it.
pub fn compete_for_fragments(
    window_idx: &[i64],
    rt_observed: &[f32],
    frag_start_idx: &[i64],
    frag_stop_idx: &[i64],
    fragment_mz: &[f32],
    rt_tol_seconds: f32,
    mass_tol_ppm: f32,
) -> Result<Vec<bool>, String> {
    validate_lengths(window_idx, rt_observed, frag_start_idx, frag_stop_idx)?;

    let valid: Vec<bool> = contiguous_window_bounds(window_idx)
        .into_par_iter()
        .map(|(start, stop)| {
            compete_within_window(
                &rt_observed[start..stop],
                &frag_start_idx[start..stop],
                &frag_stop_idx[start..stop],
                fragment_mz,
                rt_tol_seconds,
                mass_tol_ppm,
            )
        })
        .flatten()
        .collect();

    Ok(valid)
}

fn validate_lengths(
    window_idx: &[i64],
    rt_observed: &[f32],
    frag_start_idx: &[i64],
    frag_stop_idx: &[i64],
) -> Result<(), String> {
    let n = window_idx.len();
    if rt_observed.len() != n || frag_start_idx.len() != n || frag_stop_idx.len() != n {
        return Err(format!(
            "window_idx, rt_observed, frag_start_idx and frag_stop_idx must have the same length, got {}, {}, {} and {}",
            n,
            rt_observed.len(),
            frag_start_idx.len(),
            frag_stop_idx.len()
        ));
    }
    Ok(())
}
