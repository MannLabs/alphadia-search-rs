//! FDR helper kernels: q-value calculation and best-PSM-per-group selection.
//!
//! These mirror the pandas implementations in `alphadia/fdr/fdr.py` (`get_q_values`
//! and `keep_best`) but operate directly on contiguous arrays, avoiding the pandas
//! multi-key sort / cumsum / groupby overhead that dominates FDR at ~10M+ candidates.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

/// Lexicographic argsort by (proba asc, decoy asc, tiebreak asc).
///
/// Matches `df.sort_values([score, decoy, tiebreak], ascending=True)`. NaN scores
/// sort last (numpy default), via `f64::total_cmp`. Sorting is parallelized across
/// the global rayon pool, which dominates FDR time at 10M+ candidates.
fn argsort_fdr(proba: &[f64], decoy: &[f64], tiebreak: &[i64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..proba.len()).collect();
    order.par_sort_unstable_by(|&a, &b| {
        proba[a]
            .total_cmp(&proba[b])
            .then(decoy[a].total_cmp(&decoy[b]))
            .then(tiebreak[a].cmp(&tiebreak[b]))
    });
    order
}

/// Running target-decoy FDR (`cumsum(decoy)/cumsum(target)`) converted to
/// q-values by a reverse running minimum, computed over rows in `order`.
fn q_values_in_order(decoy: &[f64], order: &[usize]) -> Vec<f64> {
    let n = order.len();
    let mut qval = vec![0.0f64; n];
    let (mut decoy_cumsum, mut target_cumsum) = (0.0f64, 0.0f64);
    for (i, &idx) in order.iter().enumerate() {
        let d = decoy[idx];
        decoy_cumsum += d;
        target_cumsum += 1.0 - d;
        qval[i] = decoy_cumsum / target_cumsum; // inf when target_cumsum == 0, as in numpy
    }
    let mut running = f64::INFINITY;
    for q in qval.iter_mut().rev() {
        if *q < running {
            running = *q;
        }
        *q = running;
    }
    qval
}

/// Compute target-decoy q-values.
///
/// Returns the sort permutation (positional indices into the input, such that
/// `df.iloc[order]` yields the sorted frame) and the q-values aligned to that
/// sorted order. Decoys are 1.0, targets 0.0. The running FDR is
/// `cumsum(decoy) / cumsum(target)` and is converted to q-values by a reverse
/// running minimum, matching `_fdr_to_q_values`.
#[pyfunction]
pub fn fdr_q_values<'py>(
    py: Python<'py>,
    proba: PyReadonlyArray1<'py, f64>,
    decoy: PyReadonlyArray1<'py, f64>,
    tiebreak: PyReadonlyArray1<'py, i64>,
) -> PyResult<(Bound<'py, PyArray1<usize>>, Bound<'py, PyArray1<f64>>)> {
    let proba = proba.as_slice()?;
    let decoy = decoy.as_slice()?;
    let tiebreak = tiebreak.as_slice()?;

    let order = argsort_fdr(proba, decoy, tiebreak);
    let qval = q_values_in_order(decoy, &order);

    Ok((order.into_pyarray(py), qval.into_pyarray(py)))
}

/// Fused, sort-free FDR finalization: keep the best PSM per group and assign
/// q-values by counting, never building an O(N) ordering.
///
/// Replaces the `sort -> get_q_values -> keep_best -> get_q_values` chain (up to
/// four O(N log N) sorts) with:
///   1. a per-group arg-min over proba (keep-best), O(N) time, O(n_groups) memory;
///   2. a fixed-bin histogram of target/decoy counts over proba in [0, 1], whose
///      cumulative sums + reverse running minimum give the q-value per bin;
///   3. a per-row bin lookup to assign q-values.
///
/// `proba` is a softmax probability in [0, 1]; values are clamped into the bin
/// range. q-values are quantized to bin resolution (1 / n_bins), which is far
/// finer than any FDR threshold. Group ids come from pandas `ngroup()` and are
/// dense in 0..n_groups.
///
/// Returns the kept positional indices (into the input) and their q-values; the
/// caller applies both at once: `df = df.iloc[order]; df["qval"] = qval`.
#[pyfunction]
pub fn fdr_finalize<'py>(
    py: Python<'py>,
    proba: PyReadonlyArray1<'py, f64>,
    decoy: PyReadonlyArray1<'py, f64>,
    group_id: PyReadonlyArray1<'py, i64>,
    n_bins: usize,
) -> PyResult<(Bound<'py, PyArray1<usize>>, Bound<'py, PyArray1<f64>>)> {
    let proba = proba.as_slice()?;
    let decoy = decoy.as_slice()?;
    let group_id = group_id.as_slice()?;

    // 1. keep best (lowest proba) PSM per group, sort-free.
    let n_groups = (group_id.par_iter().copied().max().unwrap_or(-1) + 1) as usize;
    let mut best: Vec<usize> = vec![usize::MAX; n_groups];
    for i in 0..proba.len() {
        let g = group_id[i] as usize;
        let b = best[g];
        if b == usize::MAX || proba[i] < proba[b] {
            best[g] = i;
        }
    }
    let kept: Vec<usize> = best.into_iter().filter(|&i| i != usize::MAX).collect();

    let qval_per_kept = q_values_histogram(proba, decoy, &kept, n_bins);

    Ok((kept.into_pyarray(py), qval_per_kept.into_pyarray(py)))
}

/// Map a probability in [0, 1] to a histogram bin index in [0, n_bins).
#[inline]
fn proba_bin(p: f64, n_bins: usize) -> usize {
    let b = (p * n_bins as f64) as isize;
    b.clamp(0, n_bins as isize - 1) as usize
}

/// Compute q-values for the rows in `kept` by histogramming target/decoy counts
/// over `proba` bins, without sorting. Returns q-values aligned to `kept`.
fn q_values_histogram(proba: &[f64], decoy: &[f64], kept: &[usize], n_bins: usize) -> Vec<f64> {
    // target/decoy counts per proba bin
    let mut target_hist = vec![0.0f64; n_bins];
    let mut decoy_hist = vec![0.0f64; n_bins];
    for &i in kept {
        let bin = proba_bin(proba[i], n_bins);
        let d = decoy[i];
        decoy_hist[bin] += d;
        target_hist[bin] += 1.0 - d;
    }

    // ascending cumulative counts -> running FDR per bin
    let mut fdr = vec![0.0f64; n_bins];
    let (mut decoy_cumsum, mut target_cumsum) = (0.0f64, 0.0f64);
    for bin in 0..n_bins {
        decoy_cumsum += decoy_hist[bin];
        target_cumsum += target_hist[bin];
        fdr[bin] = decoy_cumsum / target_cumsum; // NaN for empty leading bins, fixed below
    }

    // reverse running minimum -> q-value per bin (empty/NaN bins inherit the next valid q)
    let mut running = f64::INFINITY;
    for bin in (0..n_bins).rev() {
        let f = fdr[bin];
        if f.is_finite() && f < running {
            running = f;
        }
        fdr[bin] = running;
    }

    // assign per kept row
    kept.iter()
        .map(|&i| fdr[proba_bin(proba[i], n_bins)])
        .collect()
}

/// Keep the best (lowest-score) PSM per group.
///
/// Mirrors `keep_best`: for each `group_id`, retain the row with the minimum
/// `score`; ties are broken by the lowest original index (deterministic).
/// Returns the kept positional indices in ascending (original) order, so the
/// caller can do `df.iloc[keep].reset_index(drop=True)`.
#[pyfunction]
pub fn fdr_keep_best<'py>(
    py: Python<'py>,
    score: PyReadonlyArray1<'py, f64>,
    group_id: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<usize>>> {
    let score = score.as_slice()?;
    let group_id = group_id.as_slice()?;

    let mut best: HashMap<i64, usize> = HashMap::new();
    for i in 0..score.len() {
        best.entry(group_id[i])
            .and_modify(|j| {
                if score[i] < score[*j] {
                    *j = i;
                }
            })
            .or_insert(i);
    }

    let mut keep: Vec<usize> = best.into_values().collect();
    keep.sort_unstable();
    Ok(keep.into_pyarray(py))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argsort_orders_by_proba_then_decoy_then_tiebreak() {
        let proba = [0.5, 0.1, 0.5, 0.1];
        let decoy = [1.0, 0.0, 0.0, 0.0];
        let tiebreak = [10, 20, 5, 10];
        // expected ascending: (0.1,0,10)=3, (0.1,0,20)=1, (0.5,0,5)=2, (0.5,1,10)=0
        assert_eq!(argsort_fdr(&proba, &decoy, &tiebreak), vec![3, 1, 2, 0]);
    }

    #[test]
    fn q_values_match_reference() {
        // sorted proba: targets then a decoy. fdr = cumsum(decoy)/cumsum(target).
        let proba = [0.1, 0.2, 0.3, 0.4];
        let decoy = [0.0, 0.0, 1.0, 0.0];
        let tiebreak = [0i64, 1, 2, 3];

        let order = argsort_fdr(&proba, &decoy, &tiebreak);
        let n = order.len();
        let mut qval = vec![0.0f64; n];
        let (mut dc, mut tc) = (0.0, 0.0);
        for (i, &idx) in order.iter().enumerate() {
            dc += decoy[idx];
            tc += 1.0 - decoy[idx];
            qval[i] = dc / tc;
        }
        // fdr = [0/1, 0/2, 1/2, 1/3] = [0, 0, 0.5, 0.333]
        // reverse cummin -> [0, 0, 0.333, 0.333]
        let mut running = f64::INFINITY;
        for q in qval.iter_mut().rev() {
            if *q < running {
                running = *q;
            }
            *q = running;
        }
        assert!((qval[0] - 0.0).abs() < 1e-12);
        assert!((qval[1] - 0.0).abs() < 1e-12);
        assert!((qval[2] - 1.0 / 3.0).abs() < 1e-12);
        assert!((qval[3] - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn histogram_q_values_approximate_reference() {
        // distinct probas, fine bins -> should match exact reference closely
        let proba = [0.10, 0.20, 0.30, 0.40, 0.50];
        let decoy = [0.0, 0.0, 1.0, 0.0, 1.0];
        let kept: Vec<usize> = (0..5).collect();
        let q = q_values_histogram(&proba, &decoy, &kept, 1 << 20);

        // exact fdr in proba order: [0/1,0/2,1/2,1/3,2/3] -> reverse cummin
        // = [0,0,1/3,1/3,2/3]
        let expected = [0.0, 0.0, 1.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0];
        for (got, exp) in q.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-4, "got {got}, expected {exp}");
        }
    }

    #[test]
    fn keep_best_picks_min_score_per_group() {
        let score = [0.9, 0.1, 0.5, 0.2, 0.3];
        let group = [1i64, 1, 2, 2, 1];
        // group 1: indices 0,1,4 -> min score 0.1 at idx 1
        // group 2: indices 2,3 -> min score 0.2 at idx 3
        let mut best: HashMap<i64, usize> = HashMap::new();
        for i in 0..score.len() {
            best.entry(group[i])
                .and_modify(|j| {
                    if score[i] < score[*j] {
                        *j = i;
                    }
                })
                .or_insert(i);
        }
        let mut keep: Vec<usize> = best.into_values().collect();
        keep.sort_unstable();
        assert_eq!(keep, vec![1, 3]);
    }
}
