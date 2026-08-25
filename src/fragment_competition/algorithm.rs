//! Fragment competition sweep. Ports the numba kernel that used to live in
//! `alphadia/fragcomp/fragcomp.py`.
//!
//! An invalidated candidate can no longer invalidate others, so the order a window
//! is swept in decides who survives. Grouping and ranking therefore happen here
//! rather than in the caller, which used to have to sort its dataframe just so.

use numpy::ndarray::{s, ArrayView4};
use rayon::prelude::*;
use std::cmp::Ordering;

/// Below this many shared ions the overlap is assumed coincidental.
const MIN_OVERLAPPING_FRAGMENTS: usize = 3;

/// Number of shared ions between two candidates.
///
/// ppm is taken relative to `frag_mz_1` only, so the count is not quite symmetric.
/// Kept that way to match the numpy broadcast this replaces.
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

/// m/z bounds of each DIA isolation window.
///
/// Built from the cycle array, shape `(1, n_windows, n_scans, 2)`. A window spans
/// the widest isolation range over its scans.
pub struct WindowBounds {
    lower: Vec<f32>,
    upper: Vec<f32>,
}

impl WindowBounds {
    pub fn from_cycle(cycle: ArrayView4<'_, f32>) -> Result<Self, String> {
        let shape = cycle.shape();
        if shape[3] < 2 {
            return Err(format!(
                "cycle must have 2 entries in its last axis, got shape {shape:?}"
            ));
        }
        let n_windows = shape[1];
        let mut lower = Vec::with_capacity(n_windows);
        let mut upper = Vec::with_capacity(n_windows);
        for w in 0..n_windows {
            lower.push(
                cycle
                    .slice(s![0, w, .., 0])
                    .iter()
                    .copied()
                    .fold(f32::INFINITY, f32::min),
            );
            upper.push(
                cycle
                    .slice(s![0, w, .., 1])
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max),
            );
        }
        Ok(Self { lower, upper })
    }

    /// Falls back to window 0 when nothing matches, as `np.argmax` did on an
    /// all-false row. Precursors outside every window are rare and get dropped later.
    fn index_of(&self, mz: f32) -> usize {
        self.lower
            .iter()
            .zip(&self.upper)
            .position(|(&lo, &hi)| mz >= lo && mz < hi)
            .unwrap_or(0)
    }
}

/// Sweep one window. `rows` is in priority order, and the result aligns with it.
fn compete_within_window(
    rows: &[usize],
    rt_observed: &[f32],
    frag_start_idx: &[i64],
    frag_stop_idx: &[i64],
    fragment_mz: &[f32],
    rt_tol_seconds: f32,
    mass_tol_ppm: f32,
) -> Vec<bool> {
    let n = rows.len();
    let mut valid = vec![true; n];
    let frags =
        |row: usize| &fragment_mz[frag_start_idx[row] as usize..frag_stop_idx[row] as usize];

    for i in 0..n {
        if !valid[i] {
            continue;
        }
        let frags_i = frags(rows[i]);
        for j in 0..n {
            if i == j || !valid[j] {
                continue;
            }
            // Positive test so a NaN retention time never competes.
            let within_rt_tolerance =
                (rt_observed[rows[i]] - rt_observed[rows[j]]).abs() < rt_tol_seconds;
            if !within_rt_tolerance {
                continue;
            }
            if fragment_overlap(frags_i, frags(rows[j]), mass_tol_ppm) >= MIN_OVERLAPPING_FRAGMENTS
            {
                valid[j] = false;
            }
        }
    }

    valid
}

/// Group rows by window, best `proba` first. `precursor_idx` breaks ties so the
/// outcome does not depend on the order the caller passed.
fn windows_in_priority_order(
    window_idx: &[usize],
    proba: &[f64],
    precursor_idx: &[i64],
) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..window_idx.len()).collect();
    order.sort_by(|&a, &b| {
        window_idx[a]
            .cmp(&window_idx[b])
            .then_with(|| proba[a].partial_cmp(&proba[b]).unwrap_or(Ordering::Equal))
            .then_with(|| precursor_idx[a].cmp(&precursor_idx[b]))
    });

    let mut windows: Vec<Vec<usize>> = Vec::new();
    for row in order {
        match windows.last() {
            Some(last) if window_idx[last[0]] == window_idx[row] => {
                windows.last_mut().unwrap().push(row)
            }
            _ => windows.push(vec![row]),
        }
    }
    windows
}

/// Returns a `valid` mask in the order the candidates were passed.
///
/// `fragment_mz` holds every candidate's ions; `frag_start_idx`/`frag_stop_idx`
/// slice into it.
#[allow(clippy::too_many_arguments)]
pub fn compete_for_fragments(
    precursor_mz: &[f32],
    precursor_idx: &[i64],
    proba: &[f64],
    rt_observed: &[f32],
    frag_start_idx: &[i64],
    frag_stop_idx: &[i64],
    fragment_mz: &[f32],
    bounds: &WindowBounds,
    rt_tol_seconds: f32,
    mass_tol_ppm: f32,
) -> Result<Vec<bool>, String> {
    validate_lengths(
        precursor_mz,
        precursor_idx,
        proba,
        rt_observed,
        frag_start_idx,
        frag_stop_idx,
    )?;
    validate_fragment_ranges(frag_start_idx, frag_stop_idx, fragment_mz.len())?;

    let window_idx: Vec<usize> = precursor_mz.iter().map(|&mz| bounds.index_of(mz)).collect();
    let windows = windows_in_priority_order(&window_idx, proba, precursor_idx);

    let swept: Vec<Vec<bool>> = windows
        .par_iter()
        .map(|rows| {
            compete_within_window(
                rows,
                rt_observed,
                frag_start_idx,
                frag_stop_idx,
                fragment_mz,
                rt_tol_seconds,
                mass_tol_ppm,
            )
        })
        .collect();

    let mut valid = vec![true; precursor_mz.len()];
    for (rows, window_valid) in windows.iter().zip(swept) {
        for (&row, is_valid) in rows.iter().zip(window_valid) {
            valid[row] = is_valid;
        }
    }
    Ok(valid)
}

fn validate_lengths(
    precursor_mz: &[f32],
    precursor_idx: &[i64],
    proba: &[f64],
    rt_observed: &[f32],
    frag_start_idx: &[i64],
    frag_stop_idx: &[i64],
) -> Result<(), String> {
    let n = precursor_mz.len();
    let lengths = [
        precursor_idx.len(),
        proba.len(),
        rt_observed.len(),
        frag_start_idx.len(),
        frag_stop_idx.len(),
    ];
    if lengths.iter().any(|&len| len != n) {
        return Err(format!(
            "per-candidate arrays must all have length {n}, got precursor_idx={}, proba={}, rt_observed={}, frag_start_idx={}, frag_stop_idx={}",
            lengths[0], lengths[1], lengths[2], lengths[3], lengths[4]
        ));
    }
    Ok(())
}

/// The sweep slices `fragment_mz` unchecked, so every range has to be valid up front.
fn validate_fragment_ranges(
    frag_start_idx: &[i64],
    frag_stop_idx: &[i64],
    fragment_mz_len: usize,
) -> Result<(), String> {
    for (i, (&start, &stop)) in frag_start_idx.iter().zip(frag_stop_idx).enumerate() {
        if start < 0 || stop < start || stop as usize > fragment_mz_len {
            return Err(format!(
                "invalid fragment range at index {i}: [{start}, {stop}) is out of bounds for fragment_mz of length {fragment_mz_len}"
            ));
        }
    }
    Ok(())
}
