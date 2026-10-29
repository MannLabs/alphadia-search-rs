//! Mixture-model deconvolution of shared fragment signal.
//!
//! In one isolation window at one cycle the observed peaks are a superposition of the fragment
//! spectra of the co-isolated precursors: `y = Σ_p a_p s_p + noise`, with `s_p` the library
//! spectrum of precursor `p` (normalised to sum 1) and `a_p ≥ 0` its abundance at that cycle.
//! The precursors with a candidate whose peak window covers the cycle form the system (targets
//! and decoys alike); the ranks of one precursor share one abundance, since they explain the
//! same peaks with the same spectrum. Rows are the observed peaks of the involved precursors
//! (fragments within the mass tolerance of each other are one peak) and, with
//! `absent_evidence`, a zero row for every library fragment of an involved precursor without
//! signal: a precursor that would have to put intensity where nothing was observed cannot take
//! a shared peak for free. Non-negative least squares gives the abundances, and every candidate's
//! share of a shared peak is `a_p s_pf / Σ_q a_q s_qf`. A peak no other precursor matched keeps
//! share 1. The soft claims (`fragment_claim`) are computed from these shares.

use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};

use crate::candidate::CandidateCollection;
use crate::dense_xic_observation::DenseXICObservation;
use crate::speclib_flat::SpecLibFlat;
use crate::traits::DIADataTrait;
use numpy::ndarray::Array2;

/// Coordinate-descent sweeps of the per-cycle non-negative least squares.
pub const NNLS_SWEEPS: usize = 20;

/// One candidate as the deconvolution sees it.
struct Slot {
    precursor_idx: usize,
    window: Option<usize>,
    cycle_start: usize,
    cycle_stop: usize,
    fragment_mz: Vec<f32>,
    /// library intensities, normalised to sum 1
    pattern: Vec<f32>,
    /// observed intensities, fragments × cycles
    xic: Array2<f32>,
}

/// Non-negative least squares `min ||y - S a||²`, `a >= 0`, by coordinate descent.
/// `rows[b] = (y_b, [(k, s_kb), ...])`.
pub fn nnls(rows: &[(f32, Vec<(usize, f32)>)], n: usize, sweeps: usize) -> Vec<f32> {
    let mut a = vec![0.0f32; n];
    let mut norm = vec![0.0f32; n];
    // the rows each column appears in, so a sweep touches every row once per column entry
    let mut rows_of: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (b, (_, cols)) in rows.iter().enumerate() {
        for &(k, s) in cols {
            norm[k] += s * s;
            rows_of[k].push(b);
        }
    }
    for _ in 0..sweeps {
        for k in 0..n {
            if norm[k] <= 0.0 {
                continue;
            }
            // exact minimiser in a_k with the other abundances fixed
            let mut num = 0.0f32;
            for &b in &rows_of[k] {
                let (y, cols) = &rows[b];
                let mut s_k = 0.0f32;
                let mut rest = 0.0f32;
                for &(j, s) in cols {
                    if j == k {
                        s_k = s;
                    } else {
                        rest += a[j] * s;
                    }
                }
                num += s_k * (y - rest);
            }
            a[k] = (num / norm[k]).max(0.0);
        }
    }
    a
}

/// Shares of the candidates of one window at one cycle. `active` are the slots covering the
/// cycle; returns `(slot index, fragment, share)` for the fragments of shared peaks.
fn cycle_shares(
    slots: &[Slot],
    active: &[usize],
    cycle: usize,
    mass_tolerance: f32,
    absent_evidence: bool,
) -> Vec<(usize, usize, f32)> {
    // every observed fragment of the active candidates at this cycle, sorted by m/z
    let mut entries: Vec<(f32, usize, usize, f32)> = Vec::new(); // (mz, slot, fragment, y)
    for &k in active {
        let s = &slots[k];
        let t = cycle - s.cycle_start;
        for (f, &mz) in s.fragment_mz.iter().enumerate() {
            let y = s.xic[[f, t]];
            if y > 0.0 {
                entries.push((mz, k, f, y));
            }
        }
    }
    if entries.len() < 2 {
        return Vec::new();
    }
    entries.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    // neighbours within the tolerance form one peak; a peak is shared when it holds fragments
    // of at least two different precursors
    let mut members: Vec<Vec<usize>> = vec![vec![0]];
    for i in 1..entries.len() {
        let (mz_prev, mz) = (entries[i - 1].0, entries[i].0);
        if (mz - mz_prev).abs() > mz * mass_tolerance * 1e-6 {
            members.push(Vec::new());
        }
        members.last_mut().unwrap().push(i);
    }
    let precursor_of = |i: usize| slots[entries[i].1].precursor_idx;
    let shared: Vec<usize> = (0..members.len())
        .filter(|&g| {
            let first = precursor_of(members[g][0]);
            members[g].iter().any(|&i| precursor_of(i) != first)
        })
        .collect();
    if shared.is_empty() {
        return Vec::new();
    }
    // one column per precursor involved in a shared peak; its ranks share the abundance
    let mut involved: Vec<usize> = shared
        .iter()
        .flat_map(|&g| members[g].iter().map(|&i| precursor_of(i)))
        .collect();
    involved.sort_unstable();
    involved.dedup();
    let col: HashMap<usize, usize> = involved.iter().enumerate().map(|(c, &p)| (p, c)).collect();
    // one representative slot per involved precursor for its pattern and absent fragments;
    // ordered, so that the rows of the fit and with them the abundances are reproducible
    let mut representative: BTreeMap<usize, usize> = BTreeMap::new();
    for &k in active {
        if col.contains_key(&slots[k].precursor_idx) {
            representative.entry(slots[k].precursor_idx).or_insert(k);
        }
    }

    // the column entries of one peak: every involved precursor once, with its pattern value
    let peak_columns = |g: usize| -> Vec<(usize, f32)> {
        let mut cols: Vec<(usize, f32)> = Vec::new();
        for &i in &members[g] {
            let (_, k, f, _) = entries[i];
            if let Some(&c) = col.get(&slots[k].precursor_idx) {
                let s = slots[k].pattern[f];
                match cols.iter_mut().find(|(cc, _)| *cc == c) {
                    Some(entry) => entry.1 = entry.1.max(s),
                    None => cols.push((c, s)),
                }
            }
        }
        cols
    };
    let mut rows: Vec<(f32, Vec<(usize, f32)>)> = Vec::new();
    let mut peak_cols: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();
    for g in 0..members.len() {
        let cols = peak_columns(g);
        if cols.is_empty() {
            continue;
        }
        let y = members[g]
            .iter()
            .map(|&i| entries[i].3)
            .fold(0.0f32, f32::max);
        if shared.binary_search(&g).is_ok() {
            peak_cols.insert(g, cols.clone());
        }
        rows.push((y, cols));
    }
    if absent_evidence {
        for (&p, &k) in &representative {
            let s = &slots[k];
            let t = cycle - s.cycle_start;
            for f in 0..s.fragment_mz.len() {
                if s.xic[[f, t]] <= 0.0 && s.pattern[f] > 0.0 {
                    rows.push((0.0, vec![(col[&p], s.pattern[f])]));
                }
            }
        }
    }
    let a = nnls(&rows, involved.len(), NNLS_SWEEPS);
    let mut out = Vec::new();
    for &g in &shared {
        let cols = &peak_cols[&g];
        let total: f32 = cols.iter().map(|&(c, s)| a[c] * s).sum();
        let n_precursors = cols.len() as f32;
        for &i in &members[g] {
            let (_, k, f, _) = entries[i];
            let share = if total > 0.0 {
                a[col[&slots[k].precursor_idx]] * slots[k].pattern[f] / total
            } else {
                1.0 / n_precursors
            };
            out.push((k, f, share.min(1.0)));
        }
    }
    out
}

/// The share matrices (fragments × cycles, 1 where nothing is shared) of every candidate, in the
/// order of the collection; `None` for candidates that share no peak with another precursor.
pub fn compute_shares<T: DIADataTrait + Sync>(
    dia_data: &T,
    lib: &SpecLibFlat,
    candidates: &CandidateCollection,
    top_k_fragments: usize,
    mass_tolerance: f32,
    absent_evidence: bool,
) -> Vec<Option<Array2<f32>>> {
    let slots: Vec<Slot> = candidates
        .par_iter()
        .map(|candidate| {
            let precursor = lib.get_precursor_by_idx_filtered(
                candidate.precursor_idx,
                true,
                true,
                top_k_fragments,
            );
            match precursor {
                Some(p)
                    if candidate.cycle_stop > candidate.cycle_start
                        && !p.fragment_mz.is_empty() =>
                {
                    let window = dia_data.get_valid_observations(p.mz).first().cloned();
                    let xic = DenseXICObservation::new(
                        dia_data,
                        p.mz,
                        candidate.cycle_start,
                        candidate.cycle_stop,
                        mass_tolerance,
                        &p.fragment_mz,
                    )
                    .dense_xic;
                    let total: f32 = p.fragment_intensity.iter().sum();
                    let pattern = p
                        .fragment_intensity
                        .iter()
                        .map(|&v| if total > 0.0 { v / total } else { 0.0 })
                        .collect();
                    Slot {
                        precursor_idx: candidate.precursor_idx,
                        window,
                        cycle_start: candidate.cycle_start,
                        cycle_stop: candidate.cycle_stop,
                        fragment_mz: p.fragment_mz,
                        pattern,
                        xic,
                    }
                }
                _ => Slot {
                    precursor_idx: candidate.precursor_idx,
                    window: None,
                    cycle_start: 0,
                    cycle_stop: 0,
                    fragment_mz: Vec::new(),
                    pattern: Vec::new(),
                    xic: Array2::zeros((0, 0)),
                },
            }
        })
        .collect();

    let mut by_window: HashMap<usize, Vec<usize>> = HashMap::new();
    for (k, s) in slots.iter().enumerate() {
        if let Some(w) = s.window {
            by_window.entry(w).or_default().push(k);
        }
    }
    let windows: Vec<Vec<usize>> = by_window.into_values().collect();
    let updates: Vec<(usize, usize, usize, f32)> = windows
        .par_iter()
        .flat_map_iter(|ks| {
            let mut by_cycle: HashMap<usize, Vec<usize>> = HashMap::new();
            for &k in ks {
                for t in slots[k].cycle_start..slots[k].cycle_stop {
                    by_cycle.entry(t).or_default().push(k);
                }
            }
            let mut out = Vec::new();
            for (t, active) in by_cycle {
                if active.len() < 2 {
                    continue;
                }
                for (k, f, share) in
                    cycle_shares(&slots, &active, t, mass_tolerance, absent_evidence)
                {
                    out.push((k, f, t, share));
                }
            }
            out
        })
        .collect();

    let mut shares: Vec<Option<Array2<f32>>> = (0..slots.len()).map(|_| None).collect();
    for (k, f, t, share) in updates {
        let s = &slots[k];
        let m = shares[k].get_or_insert_with(|| {
            Array2::from_elem((s.fragment_mz.len(), s.cycle_stop - s.cycle_start), 1.0)
        });
        m[[f, t - s.cycle_start]] = share;
    }
    shares
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::Array2;

    /// A slot with one cycle (cycle 0) of observed intensities.
    fn slot(precursor_idx: usize, fragments: &[(f32, f32, f32)]) -> Slot {
        // (m/z, library intensity, observed intensity)
        let total: f32 = fragments.iter().map(|f| f.1).sum();
        Slot {
            precursor_idx,
            window: Some(0),
            cycle_start: 0,
            cycle_stop: 1,
            fragment_mz: fragments.iter().map(|f| f.0).collect(),
            pattern: fragments.iter().map(|f| f.1 / total).collect(),
            xic: Array2::from_shape_vec(
                (fragments.len(), 1),
                fragments.iter().map(|f| f.2).collect(),
            )
            .unwrap(),
        }
    }

    fn share_of(shares: &[(usize, usize, f32)], k: usize, f: usize) -> Option<f32> {
        shares.iter().find(|s| s.0 == k && s.1 == f).map(|s| s.2)
    }

    #[test]
    fn nnls_recovers_non_negative_abundances() {
        // given y = 2 * s_0 + 0 * s_1 on three rows
        let rows = vec![
            (2.0, vec![(0, 1.0)]),
            (1.0, vec![(0, 0.5), (1, 0.5)]),
            (0.0, vec![(1, 1.0)]),
        ];
        // when
        let a = nnls(&rows, 2, 50);
        // then the second column does not go negative
        assert!((a[0] - 2.0).abs() < 1e-3, "{a:?}");
        assert!(a[1].abs() < 1e-3, "{a:?}");
    }

    #[test]
    fn a_borrower_loses_the_shared_peak_to_the_precursor_that_explains_it() {
        // given a real precursor R with four observed fragments and a borrower N that matches
        // two of them and predicts two fragments where nothing was observed
        let slots = vec![
            slot(
                0,
                &[
                    (300.0, 1.0, 100.0),
                    (400.0, 3.0, 300.0),
                    (500.0, 3.0, 300.0),
                    (600.0, 3.0, 300.0),
                ],
            ),
            slot(
                1,
                &[
                    (300.0, 1.0, 100.0),
                    (400.0, 1.0, 300.0),
                    (700.0, 1.0, 0.0),
                    (800.0, 1.0, 0.0),
                ],
            ),
        ];
        // when
        let shares = cycle_shares(&slots, &[0, 1], 0, 10.0, true);
        // then R keeps nearly all of the shared peaks and N keeps nearly nothing
        for f in [0, 1] {
            assert!(share_of(&shares, 0, f).unwrap() > 0.9, "{shares:?}");
            assert!(share_of(&shares, 1, f).unwrap() < 0.1, "{shares:?}");
        }
        // and the unshared fragments get no share entry
        assert!(share_of(&shares, 0, 2).is_none());
    }

    #[test]
    fn absent_fragments_are_evidence_against_a_borrower() {
        // given R whose library under-predicts the first shared peak, so the fit leaves a
        // residual there, and a borrower N with two absent fragments
        let slots = vec![
            slot(
                0,
                &[
                    (300.0, 1.0, 300.0),
                    (400.0, 3.0, 300.0),
                    (500.0, 3.0, 300.0),
                    (600.0, 3.0, 300.0),
                ],
            ),
            slot(
                1,
                &[(300.0, 1.0, 300.0), (700.0, 1.0, 0.0), (800.0, 1.0, 0.0)],
            ),
        ];
        // when the fit uses or ignores the absent fragments
        let with = cycle_shares(&slots, &[0, 1], 0, 10.0, true);
        let without = cycle_shares(&slots, &[0, 1], 0, 10.0, false);
        // then N takes the residual only when its missing fragments do not count against it
        let (n_with, n_without) = (
            share_of(&with, 1, 0).unwrap(),
            share_of(&without, 1, 0).unwrap(),
        );
        assert!(n_with < n_without, "{n_with} vs {n_without}");
        assert!(n_without > 0.5, "{n_without}");
    }

    #[test]
    fn ranks_of_one_precursor_share_one_abundance() {
        // given two ranks of precursor 0 and one candidate of precursor 1 on the same peaks
        let a = [(300.0, 1.0, 200.0), (400.0, 1.0, 200.0)];
        let slots = vec![
            slot(0, &a),
            slot(0, &a),
            slot(1, &[(300.0, 1.0, 200.0), (900.0, 1.0, 0.0)]),
        ];
        // when
        let shares = cycle_shares(&slots, &[0, 1, 2], 0, 10.0, true);
        // then both ranks get the same share of the shared peak, not half of the precursor's
        let (r0, r1) = (
            share_of(&shares, 0, 0).unwrap(),
            share_of(&shares, 1, 0).unwrap(),
        );
        assert!((r0 - r1).abs() < 1e-6);
        assert!(r0 > 0.9, "{shares:?}");
        // and ranks of one precursor alone share nothing
        assert!(cycle_shares(&slots, &[0, 1], 0, 10.0, true).is_empty());
    }

    #[test]
    fn precursors_without_common_peaks_get_no_shares() {
        // given two precursors with disjoint fragments
        let slots = vec![
            slot(0, &[(300.0, 1.0, 100.0), (400.0, 1.0, 100.0)]),
            slot(1, &[(500.0, 1.0, 100.0), (600.0, 1.0, 100.0)]),
        ];
        // when / then
        assert!(cycle_shares(&slots, &[0, 1], 0, 10.0, true).is_empty());
    }

    #[test]
    fn a_target_and_its_decoy_are_one_label_blind_system() {
        use crate::fragment_claim::tests::{
            candidate, dia_data, library, ALL_MZ, FRAGMENTS_A, WINDOW_LOW,
        };
        // given target T and a decoy D that shares T's most intense fragment and predicts two
        // fragments that were not observed
        let shared_mz = FRAGMENTS_A[0].0;
        let decoy: [(f32, f32); 3] = [(shared_mz, 5.0), (910.0, 4.0), (920.0, 3.0)];
        let dia = dia_data(&[(WINDOW_LOW, &ALL_MZ)]);
        let lib = library(&[(0, 500.0, &FRAGMENTS_A), (1, 505.0, &decoy)]);
        let position = |precursor_idx: usize| {
            lib.get_precursor_by_idx_filtered(precursor_idx, true, true, 12)
                .unwrap()
                .fragment_mz
                .iter()
                .position(|&mz| mz == shared_mz)
                .unwrap()
        };
        let (f_target, f_decoy) = (position(0), position(1));
        let shares_in = |order: [usize; 2]| {
            let candidates = CandidateCollection::from_vec(
                order.iter().map(|&p| candidate(p, 1, 1.0, 2)).collect(),
            );
            let shares = compute_shares(&dia, &lib, &candidates, 12, 10.0, true);
            let of = |p: usize| shares[order.iter().position(|&q| q == p).unwrap()].clone();
            (of(0).unwrap(), of(1).unwrap())
        };
        // when
        let (target, decoy) = shares_in([0, 1]);
        // then at every cycle the target keeps the shared peak and the decoy keeps almost nothing
        for t in 0..target.ncols() {
            assert!(target[[f_target, t]] > 0.9, "{target:?}");
            assert!(decoy[[f_decoy, t]] < 0.1, "{decoy:?}");
        }
        // and the fit does not depend on the candidate order
        let (target_swapped, decoy_swapped) = shares_in([1, 0]);
        assert_eq!(target, target_swapped);
        assert_eq!(decoy, decoy_swapped);
    }
}
