//! Fragment competition sweep. Port of
//! `alphadia/fragcomp/fragcomp.py::_compete_for_fragments` /
//! `_get_fragment_overlap`.
//!
//! Candidates are assigned to the DIA isolation window containing their precursor
//! m/z and compete only within that window. Each window is swept independently and
//! in parallel, in priority order: ascending `proba` (lower is better), ties broken
//! by ascending `precursor_idx`. The sweep is order-sensitive — once a candidate is
//! invalidated it can no longer invalidate others — so of two conflicting candidates
//! the higher-priority one survives.
//!
//! Callers pass candidates in any order and get a mask back in that same order;
//! grouping and ranking happen here, not in the caller's sort.

use numpy::ndarray::{s, ArrayView4};
use rayon::prelude::*;
use std::cmp::Ordering;

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

/// Per-window m/z bounds, reduced from the DIA cycle.
///
/// `cycle` is the alphaDIA cycle array of shape `(1, n_windows, n_scans, 2)`; a
/// window spans the widest isolation range over its scans.
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

    /// Index of the first window containing `mz`.
    ///
    /// Falls back to window 0 when no window matches, mirroring the `np.argmax` the
    /// Python implementation used on an all-false row.
    fn index_of(&self, mz: f32) -> usize {
        self.lower
            .iter()
            .zip(&self.upper)
            .position(|(&lo, &hi)| mz >= lo && mz < hi)
            .unwrap_or(0)
    }
}

/// Sweep one DIA window, given its member rows in priority order.
///
/// Returns the surviving rows' validity aligned with `rows`.
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
            // Phrased as a positive test so a non-finite RT never competes, matching
            // the `delta_rt < rt_tol_seconds` guard in the Python implementation.
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

/// Group rows by DIA window, ordered within each window by descending priority
/// (ascending `proba`, ties broken by ascending `precursor_idx`).
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

/// Run fragment competition over all candidates and return a `valid` mask in the
/// same order as the inputs.
///
/// `fragment_mz` is the full fragment array; `frag_start_idx`/`frag_stop_idx` are
/// offsets into it.
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

/// Check that every `[frag_start_idx[i], frag_stop_idx[i])` range is a valid, in-bounds
/// slice of `fragment_mz`, since `compete_within_window` indexes it unchecked.
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
