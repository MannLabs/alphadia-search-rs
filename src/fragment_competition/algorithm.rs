//! This module groups the candidates into isolation windows and sorts each window by score.
//! If two candidates claim the same fragment ions, the better candidate invalidates the worse
//! candidate. `DOCS.md` gives the data layout and an example.
//!
//! A candidate that loses does not compete again. It cannot invalidate another candidate, and
//! no other candidate can invalidate it. The order of the comparisons therefore decides which
//! candidates survive. For this reason, this module groups and sorts the candidates itself. It
//! does not use the row order of the caller.

use numpy::ndarray::ArrayView4;
use rayon::prelude::*;

/// Two candidates compete only if they share this number of ions or more. Fewer shared ions
/// are a coincidence.
///
/// This value is an empirical threshold, not a physical constant. With fewer than three
/// ions, a shared series can occur by chance. No caller needs a different value, so this
/// module keeps the value constant.
const MIN_OVERLAPPING_FRAGMENTS: usize = 3;

/// Counts the ions that two candidates have in common, within `mass_tol_ppm`.
///
/// The ppm distance is relative to `frag_mz_1`, so the count is not fully symmetric. The
/// caller gives the ions of the candidate that claims the signal as `frag_mz_1`. This makes
/// the reference clear.
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

/// The m/z range of each DIA isolation window.
///
/// The cycle array has the shape `(1, n_windows, n_scans, 2)` and gives the limits of each
/// scan. The range of a window covers all of its scans.
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
        let (n_windows, n_scans) = (shape[1], shape[2]);
        let mut lower = Vec::with_capacity(n_windows);
        let mut upper = Vec::with_capacity(n_windows);
        for window in 0..n_windows {
            let mut window_lower = f32::INFINITY;
            let mut window_upper = f32::NEG_INFINITY;
            for scan in 0..n_scans {
                let (scan_lower, scan_upper) =
                    (cycle[[0, window, scan, 0]], cycle[[0, window, scan, 1]]);
                // The min and max operations below ignore a NaN limit. The window then
                // becomes too large.
                if scan_lower.is_nan() || scan_upper.is_nan() {
                    return Err(format!(
                        "cycle must not contain NaN, found one at window {window}, scan {scan}"
                    ));
                }
                window_lower = window_lower.min(scan_lower);
                window_upper = window_upper.max(scan_upper);
            }
            lower.push(window_lower);
            upper.push(window_upper);
        }
        Ok(Self { lower, upper })
    }

    /// The window whose range `[lower, upper)` contains `mz`.
    ///
    /// If no window contains the m/z, this function gives window 0. The precursor then
    /// competes with the candidates of window 0. This occurs only if the m/z is outside the
    /// acquisition range.
    fn index_of(&self, mz: f32) -> usize {
        self.lower
            .iter()
            .zip(&self.upper)
            .position(|(&lo, &hi)| mz >= lo && mz < hi)
            .unwrap_or(0)
    }
}

/// Runs the competition inside one window. The rows in `rows` are in priority order. The
/// result has the same order.
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
            let delta_rt = (rt_observed[rows[i]] - rt_observed[rows[j]]).abs();
            if delta_rt >= rt_tol_seconds {
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

/// Groups the rows by window, with the lowest (best) `proba` first. `precursor_idx` breaks a
/// tie. The result therefore does not depend on the row order of the caller.
fn windows_in_priority_order(
    window_idx: &[usize],
    proba: &[f64],
    precursor_idx: &[i64],
) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..window_idx.len()).collect();
    order.sort_by(|&a, &b| {
        window_idx[a]
            .cmp(&window_idx[b])
            .then_with(|| proba[a].total_cmp(&proba[b]))
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

/// Gives a `valid` mask in the candidate order of the caller.
///
/// `fragment_mz` holds the ions of all candidates. `frag_start_idx` and `frag_stop_idx` give
/// the range of each candidate. The function gives an error if the array lengths do not
/// agree, if a fragment range is outside `fragment_mz`, or if a float input contains NaN.
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
    validate_no_nan("precursor_mz", precursor_mz.iter().map(|&mz| mz.into()))?;
    validate_no_nan("proba", proba.iter().copied())?;
    validate_no_nan("rt_observed", rt_observed.iter().map(|&rt| rt.into()))?;
    validate_no_nan("fragment_mz", fragment_mz.iter().map(|&mz| mz.into()))?;

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

/// The competition does not check the limits when it makes a slice of `fragment_mz`. Each
/// range must therefore be correct before the competition starts.
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

/// A float input that contains NaN is an error. A NaN loses each comparison, and it has no
/// position in the sort order. One NaN can therefore change which candidates survive.
fn validate_no_nan(name: &str, values: impl IntoIterator<Item = f64>) -> Result<(), String> {
    for (index, value) in values.into_iter().enumerate() {
        if value.is_nan() {
            return Err(format!(
                "{name} must not contain NaN, found one at index {index}"
            ));
        }
    }
    Ok(())
}
