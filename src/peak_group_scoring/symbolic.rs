//! Compact scale-free symbolic scoring features discovered by `symdia`
//! (GPU TensorGP + Adversarial Sparse KAN + Physical Co-Elution Primitives)
//! across multi-raw post-calibration DIA regimes.
//!
//! All 8 features are pure MS2 (zero MS1 scan lookups), intensity-scale-free,
//! leakage-audited (`decoy-vs-entrapment AUC ~ 0.500-0.508`), reuse the single-pass
//! `DenseXICMZObservation` buffer without duplicate MS2 XIC scans, and look up
//! 1 Da spectral crowding in O(1).

use numpy::ndarray::Array2;

use crate::candidate::Candidate;
use crate::dense_xic_observation::DenseXICMZObservation;
use crate::peak_group_scoring::feature_block::feature_block;
use crate::peak_group_scoring::library_agreement::entropy_similarity;
use crate::precursor::Precursor;
use crate::quadrupole_observation::lookup_1da_crowding;
use crate::traits::DIADataTrait;

/// Half-width (in cycles) around `candidate.cycle_center` for MS2 context extraction (T = 15).
pub const SYMBOLIC_HALF_WINDOW: usize = 7;
const WINDOW_LEN: usize = 2 * SYMBOLIC_HALF_WINDOW + 1;
const CORE_HALF_WIDTH: usize = 2;
const CORE_LO: usize = SYMBOLIC_HALF_WINDOW - CORE_HALF_WIDTH; // 5
const CORE_HI: usize = SYMBOLIC_HALF_WINDOW + CORE_HALF_WIDTH + 1; // 10

const EPS: f32 = 1e-6;

// Adversarial KAN robust standardization parameters fitted on multi-regime calibrated candidates.
const MED_COEL_CLEAN_TOP3: f32 = 0.696878;
const SCL_COEL_CLEAN_TOP3: f32 = 0.182273;
const MED_CLIQUE_SUM: f32 = 1.857143;
const SCL_CLIQUE_SUM: f32 = 1.911732;
const MED_CTX_SPEC_HITS: f32 = 1.388889;
const SCL_CTX_SPEC_HITS: f32 = 0.746103;
const MED_NOV_TOP5_PPM: f32 = 0.459167;
const SCL_NOV_TOP5_PPM: f32 = 0.143157;

feature_block! {
    /// 8 MS2-only scale-free, leakage-audited `symdia` features from calibrated TensorGP and Adversarial KAN.
    pub struct SymbolicFeatures {
        /// TensorGP: top-5 mean of crowding-normalized clean-template correlation over exp(-0.25*|ppm|)-weighted XICs.
        pub sym_tgp_crowding_ppm_top5: f32,
        /// TensorGP: top-3 mean of crowding-normalized clean-template correlation over sqrt-XICs.
        pub sym_tgp_crowding_sqrt_top3: f32,
        /// Spectral entropy similarity to the library after dropping the 2 most excess-intense fragments.
        pub sym_interf_trimmed2_entropy: f32,
        /// Spectral entropy similarity of co-eluting clean fragments against the full library.
        pub sym_lib_clean_entropy_full: f32,
        /// Adversarial KAN sigmoid-gated interaction over clean co-elution, clique sum, m/z crowding specificity, and ppm-weighted correlation.
        pub sym_kan_coel_specificity: f32,
        /// m/z specificity-weighted count of clean co-eluting fragments: sum(1 / (1 + crowd_1da / 10)).
        pub sym_ctx_specificity_hits: f32,
        /// Count of active fragments with ppm-gated clean-template correlation > 0.65.
        pub sym_nov_n_clean_ppm: f32,
        /// Count of clean co-eluting fragments with core intensity-weighted |ppm| <= calibrated inner tolerance.
        pub sym_mz_n_clean_within_tol: f32,
    }
}

#[inline]
fn sanitize(v: f32) -> f32 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

#[inline]
fn robust_z(val: f32, median: f32, scale: f32) -> f32 {
    ((val - median) / scale.max(EPS)).clamp(-3.5, 3.5)
}

/// Pearson correlation matching `_safe_pearson` (`den > 1e-7 -> num / (den + 1e-6)`).
fn safe_pearson(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    if n == 0 || b.len() != n {
        return 0.0;
    }
    let inv_n = 1.0 / n as f32;
    let a_mean = a.iter().sum::<f32>() * inv_n;
    let b_mean = b.iter().sum::<f32>() * inv_n;
    let mut num = 0.0f32;
    let mut va = 0.0f32;
    let mut vb = 0.0f32;
    for i in 0..n {
        let da = a[i] - a_mean;
        let db = b[i] - b_mean;
        num += da * db;
        va += da * da;
        vb += db * db;
    }
    let den = (va * vb).sqrt();
    if den > 1e-7 {
        (num / (den + EPS)).clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

fn spectral_entropy_sim(p: &[f32], q: &[f32]) -> f32 {
    let p_sum: f32 = p.iter().copied().filter(|&x| x > 0.0).sum();
    let q_sum: f32 = q.iter().copied().filter(|&x| x > 0.0).sum();
    if p_sum <= 1e-7 || q_sum <= 1e-7 {
        return 0.0;
    }
    let pf: Vec<f64> = p.iter().map(|&x| x.max(0.0) as f64).collect();
    let qf: Vec<f64> = q.iter().map(|&x| x.max(0.0) as f64).collect();
    (entropy_similarity(&pf, &qf) as f32).clamp(0.0, 1.0)
}

fn topk_mean(values: &[f32], k: usize) -> f32 {
    if values.is_empty() || k == 0 {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let take = k.min(sorted.len());
    sorted[..take].iter().sum::<f32>() / take as f32
}

fn kth_largest(values: &[f32], k: usize) -> f32 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    sorted[k.saturating_sub(1).min(sorted.len().saturating_sub(1))]
}

fn quantile_linear(sorted: &[f32], q: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pos = q * (sorted.len() - 1) as f32;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f32;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Copy `[center-7..center+8)` MS2 intensities and signed ppm errors from an already extracted union buffer.
fn copy_window_from_union(
    union_dense_xic: &Array2<f32>,
    union_dense_mz: &Array2<f32>,
    union_lo: usize,
    fragment_mz: &[f32],
    cycle_center: usize,
    n_cycles: usize,
) -> (Vec<[f32; WINDOW_LEN]>, Vec<[f32; WINDOW_LEN]>) {
    let n_frags = fragment_mz.len();
    let mut ms2_xic = vec![[0.0f32; WINDOW_LEN]; n_frags];
    let mut ms2_err = vec![[0.0f32; WINDOW_LEN]; n_frags];
    if n_frags == 0 || n_cycles == 0 || cycle_center >= n_cycles {
        return (ms2_xic, ms2_err);
    }

    let win_start = cycle_center as isize - SYMBOLIC_HALF_WINDOW as isize;
    let sym_lo = win_start.max(0) as usize;
    let sym_hi = (cycle_center + SYMBOLIC_HALF_WINDOW + 1).min(n_cycles);
    let max_cols = union_dense_xic.shape()[1];
    let available_hi = (union_lo + max_cols).min(sym_hi);
    if sym_lo < union_lo || sym_lo >= available_hi {
        return (ms2_xic, ms2_err);
    }

    let t_offset = (sym_lo as isize - win_start) as usize;
    let src_start = sym_lo - union_lo;
    let len = available_hi - sym_lo;

    for (f, &target_mz) in fragment_mz.iter().enumerate() {
        for c in 0..len {
            let inten = union_dense_xic[[f, src_start + c]];
            let obs_mz = union_dense_mz[[f, src_start + c]];
            ms2_xic[f][t_offset + c] = inten;
            if inten > 0.0 && obs_mz > 0.0 && target_mz > 0.0 {
                ms2_err[f][t_offset + c] = ((obs_mz - target_mz) / target_mz) * 1e6;
            }
        }
    }
    (ms2_xic, ms2_err)
}

/// Compute all 8 MS2-only `symdia` features reusing the single-pass `union_dense` buffer and O(1) crowding tables.
#[allow(clippy::too_many_arguments)]
pub fn calculate_symbolic_features_from_dense<T: DIADataTrait>(
    dia_data: &T,
    precursor: &Precursor,
    candidate: &Candidate,
    mass_tolerance: f32,
    union_dense_xic: &Array2<f32>,
    union_dense_mz: &Array2<f32>,
    union_lo: usize,
    contributing_obs_indices: &[usize],
    crowding_tables: &[Option<Vec<u8>>],
) -> SymbolicFeatures {
    let n_frags = precursor.fragment_mz.len();
    if n_frags == 0 {
        return SymbolicFeatures::default();
    }

    let n_cycles = dia_data.rt_index().rt.len();
    let (ms2_xic, ms2_err) = copy_window_from_union(
        union_dense_xic,
        union_dense_mz,
        union_lo,
        &precursor.fragment_mz,
        candidate.cycle_center,
        n_cycles,
    );

    let lib_raw: Vec<f32> = precursor
        .fragment_intensity
        .iter()
        .map(|&v| v.max(0.0))
        .collect();
    let lib_sum: f32 = lib_raw.iter().sum::<f32>() + EPS;
    let lib_norm: Vec<f32> = lib_raw.iter().map(|&v| v / lib_sum).collect();

    let mut obs_core = Vec::with_capacity(n_frags);
    let mut has_sig = Vec::with_capacity(n_frags);
    let mut xic_norm = Vec::with_capacity(n_frags);
    for row in &ms2_xic {
        let s: f32 = row[CORE_LO..CORE_HI].iter().sum();
        obs_core.push(s);
        has_sig.push(s > 0.0);
        let row_max = row.iter().copied().fold(0.0f32, f32::max);
        let inv = 1.0 / (row_max + EPS);
        let mut rn = [0.0f32; WINDOW_LEN];
        for t in 0..WINDOW_LEN {
            rn[t] = row[t] * inv;
        }
        xic_norm.push(rn);
    }

    let mut frag_crowd = vec![0.0f32; n_frags];
    for (f, &fmz) in precursor.fragment_mz.iter().enumerate() {
        let mut c_sum = 0.0f32;
        for &obs_idx in contributing_obs_indices {
            if let Some(Some(tbl)) = crowding_tables.get(obs_idx) {
                c_sum += lookup_1da_crowding(tbl, candidate.cycle_center, fmz);
            }
        }
        frag_crowd[f] = c_sum;
    }

    let inv_t = 1.0 / WINDOW_LEN as f32;
    let mut xu = Vec::with_capacity(n_frags);
    for f in 0..n_frags {
        let mean = ms2_xic[f].iter().sum::<f32>() * inv_t;
        let mut centered = [0.0f32; WINDOW_LEN];
        let mut sq = 0.0f32;
        for t in 0..WINDOW_LEN {
            let d = ms2_xic[f][t] - mean;
            centered[t] = d;
            sq += d * d;
        }
        let norm = sq.sqrt();
        if norm > 1e-7 {
            let inv = 1.0 / (norm + EPS);
            for v in &mut centered {
                *v *= inv;
            }
        } else {
            centered = [0.0f32; WINDOW_LEN];
        }
        xu.push(centered);
    }

    let mut frag_clique_score = vec![0.0f32; n_frags];
    for i in 0..n_frags {
        if !has_sig[i] || n_frags <= 1 {
            continue;
        }
        let mut top1 = -1.0f32;
        let mut top2 = -1.0f32;
        for j in 0..n_frags {
            if i == j || !has_sig[j] {
                continue;
            }
            let mut dot = 0.0f32;
            for t in 0..WINDOW_LEN {
                dot += xu[i][t] * xu[j][t];
            }
            let c = dot.clamp(-1.0, 1.0);
            if c > top1 {
                top2 = top1;
                top1 = c;
            } else if c > top2 {
                top2 = c;
            }
        }
        let avg_top2 = if n_frags > 2 {
            0.5 * (top1 + top2)
        } else {
            top1
        };
        frag_clique_score[i] = avg_top2.max(0.0);
    }
    let coel_clique_score_sum: f32 = frag_clique_score.iter().sum();

    let mut clique_w_sum = 0.0f32;
    let mut clique_w = vec![0.0f32; n_frags];
    for f in 0..n_frags {
        if has_sig[f] {
            let w = frag_clique_score[f] * frag_clique_score[f] * (lib_norm[f] + 0.05);
            clique_w[f] = w;
            clique_w_sum += w;
        }
    }
    let mut clean_template = [0.0f32; WINDOW_LEN];
    for f in 0..n_frags {
        let wn = clique_w[f] / (clique_w_sum + EPS);
        for t in 0..WINDOW_LEN {
            clean_template[t] += xic_norm[f][t] * wn;
        }
    }
    let corr_clean: Vec<f32> = (0..n_frags)
        .map(|f| {
            if has_sig[f] {
                safe_pearson(&ms2_xic[f], &clean_template)
            } else {
                0.0
            }
        })
        .collect();
    let clean_mask: Vec<bool> = (0..n_frags)
        .map(|f| has_sig[f] && corr_clean[f] > 0.60)
        .collect();
    let coel_clean_top3_corr = topk_mean(&corr_clean, 3);

    let mut w_lib_sum = 0.0f32;
    let mut w_lib = vec![0.0f32; n_frags];
    for f in 0..n_frags {
        if has_sig[f] {
            w_lib[f] = lib_norm[f];
            w_lib_sum += lib_norm[f];
        }
    }
    let mut lib_tpl = [0.0f32; WINDOW_LEN];
    for f in 0..n_frags {
        let wn = w_lib[f] / (w_lib_sum + EPS);
        for t in 0..WINDOW_LEN {
            lib_tpl[t] += xic_norm[f][t] * wn;
        }
    }

    let mut tgp_ppm_terms = Vec::with_capacity(n_frags);
    let mut tgp_sqrt_terms = Vec::with_capacity(n_frags);
    let mut corr_exp_ppm = Vec::with_capacity(n_frags);
    let mut nov_n_clean_ppm = 0.0f32;

    for f in 0..n_frags {
        let mut x_exp = [0.0f32; WINDOW_LEN];
        let mut x_sqrt = [0.0f32; WINDOW_LEN];
        for t in 0..WINDOW_LEN {
            let v = ms2_xic[f][t].max(0.0);
            x_exp[t] = v * (-0.25 * ms2_err[f][t].abs()).exp();
            x_sqrt[t] = v.sqrt();
        }
        let ce = safe_pearson(&x_exp, &lib_tpl).max(0.0);
        let cs = safe_pearson(&x_sqrt, &lib_tpl).max(0.0);
        let crowd_div = 1.0 / (1.0 + frag_crowd[f] / 10.0);
        tgp_ppm_terms.push(ce * crowd_div);
        tgp_sqrt_terms.push(cs * crowd_div);
        corr_exp_ppm.push(ce);
        if has_sig[f] && ce > 0.65 {
            nov_n_clean_ppm += 1.0;
        }
    }
    let sym_tgp_crowding_ppm_top5 = topk_mean(&tgp_ppm_terms, 5);
    let sym_tgp_crowding_sqrt_top3 = topk_mean(&tgp_sqrt_terms, 3);
    let nov_top5_corr_exp_ppm = topk_mean(&corr_exp_ppm, 5);

    let tol_thresh = (0.57 * mass_tolerance).max(4.0);
    let mut sym_ctx_specificity_hits = 0.0f32;
    let mut sym_mz_n_clean_within_tol = 0.0f32;
    let mut obs_clean = vec![0.0f32; n_frags];
    for f in 0..n_frags {
        if clean_mask[f] {
            obs_clean[f] = obs_core[f];
            sym_ctx_specificity_hits += 1.0 / (1.0 + frag_crowd[f] / 10.0);
            let mut err_w_sum = 0.0f32;
            for t in CORE_LO..CORE_HI {
                err_w_sum += ms2_err[f][t] * ms2_xic[f][t];
            }
            let mean_ppm = (err_w_sum / (obs_core[f] + EPS)).abs();
            if mean_ppm <= tol_thresh {
                sym_mz_n_clean_within_tol += 1.0;
            }
        }
    }
    let sym_lib_clean_entropy_full = spectral_entropy_sim(&obs_clean, &lib_norm);

    let mut active_ratios: Vec<f32> = (0..n_frags)
        .filter(|&f| has_sig[f])
        .map(|f| obs_core[f] / (lib_norm[f] + 1e-4))
        .collect();
    active_ratios.sort_by(|a, b| a.total_cmp(b));
    let alpha_hat = quantile_linear(&active_ratios, 0.35);
    let n_active = active_ratios.len();

    let excess: Vec<f32> = (0..n_frags)
        .map(|f| {
            if has_sig[f] {
                (obs_core[f] + 1.0) / (alpha_hat * lib_norm[f] + 1.0)
            } else {
                -1.0
            }
        })
        .collect();
    let drop_k = n_frags.saturating_sub(3).min(2);
    let cutoff = if drop_k > 0 {
        kth_largest(&excess, drop_k)
    } else {
        f32::INFINITY
    };
    let mut obs_strip = vec![0.0f32; n_frags];
    let mut lib_strip = vec![0.0f32; n_frags];
    for f in 0..n_frags {
        if n_active <= 4 || excess[f] < cutoff {
            obs_strip[f] = obs_core[f];
            lib_strip[f] = lib_norm[f];
        }
    }
    let sym_interf_trimmed2_entropy = spectral_entropy_sim(&obs_strip, &lib_strip);

    let z_c3 = robust_z(
        coel_clean_top3_corr,
        MED_COEL_CLEAN_TOP3,
        SCL_COEL_CLEAN_TOP3,
    );
    let z_csum = robust_z(coel_clique_score_sum, MED_CLIQUE_SUM, SCL_CLIQUE_SUM);
    let z_hits = robust_z(
        sym_ctx_specificity_hits,
        MED_CTX_SPEC_HITS,
        SCL_CTX_SPEC_HITS,
    );
    let z_nov5 = robust_z(nov_top5_corr_exp_ppm, MED_NOV_TOP5_PPM, SCL_NOV_TOP5_PPM);
    let sym_kan_coel_specificity = (-0.3685 * z_csum - 0.7647 * z_hits - 0.3671 * z_nov5.max(0.0)
        + 1.1899)
        - 0.9707 / (1.0 + (-2.0 * z_c3).exp());

    SymbolicFeatures {
        sym_tgp_crowding_ppm_top5: sanitize(sym_tgp_crowding_ppm_top5),
        sym_tgp_crowding_sqrt_top3: sanitize(sym_tgp_crowding_sqrt_top3),
        sym_interf_trimmed2_entropy: sanitize(sym_interf_trimmed2_entropy),
        sym_lib_clean_entropy_full: sanitize(sym_lib_clean_entropy_full),
        sym_kan_coel_specificity: sanitize(sym_kan_coel_specificity),
        sym_ctx_specificity_hits: sanitize(sym_ctx_specificity_hits),
        sym_nov_n_clean_ppm: sanitize(nov_n_clean_ppm),
        sym_mz_n_clean_within_tol: sanitize(sym_mz_n_clean_within_tol),
    }
}

/// Convenience entry point computing symbolic features standalone (used in unit tests).
pub fn calculate_symbolic_features<T: DIADataTrait>(
    dia_data: &T,
    precursor: &Precursor,
    candidate: &Candidate,
    mass_tolerance: f32,
) -> SymbolicFeatures {
    let n_cycles = dia_data.rt_index().rt.len();
    let sym_lo = candidate
        .cycle_center
        .saturating_sub(SYMBOLIC_HALF_WINDOW)
        .min(n_cycles);
    let sym_hi = (candidate.cycle_center + SYMBOLIC_HALF_WINDOW + 1).min(n_cycles);
    if sym_lo >= sym_hi || precursor.fragment_mz.is_empty() {
        return SymbolicFeatures::default();
    }
    let union_dense = DenseXICMZObservation::new(
        dia_data,
        precursor.mz,
        sym_lo,
        sym_hi,
        mass_tolerance,
        &precursor.fragment_mz,
    );
    calculate_symbolic_features_from_dense(
        dia_data,
        precursor,
        candidate,
        mass_tolerance,
        &union_dense.dense_xic,
        &union_dense.dense_mz,
        sym_lo,
        &union_dense.contributing_obs_indices,
        &[],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mz_index::MZIndex;
    use crate::peak_group_scoring::feature_block::FeatureBlock;
    use crate::rt_index::RTIndex;
    use crate::traits::QuadrupoleObservationTrait;
    use numpy::ndarray::{Array1, ArrayViewMut1};
    use std::collections::HashSet;

    struct MockObs {
        iso: [f32; 2],
        mz_peaks: Vec<(f32, usize, f32)>,
        table: Option<Vec<u8>>,
    }

    impl QuadrupoleObservationTrait for MockObs {
        fn fill_xic_slice(
            &self,
            _mz_index: &MZIndex,
            dense_xic: &mut ArrayViewMut1<f32>,
            cycle_start_idx: usize,
            cycle_stop_idx: usize,
            mass_tolerance: f32,
            mz: f32,
        ) {
            let tol = mz * mass_tolerance * 1e-6;
            for &(pmz, cyc, inten) in &self.mz_peaks {
                if cyc >= cycle_start_idx && cyc < cycle_stop_idx && (pmz - mz).abs() <= tol {
                    dense_xic[cyc - cycle_start_idx] += inten;
                }
            }
        }

        fn fill_xic_and_mz_slice(
            &self,
            _mz_index: &MZIndex,
            dense_xic: &mut ArrayViewMut1<f32>,
            dense_mz: &mut ArrayViewMut1<f32>,
            cycle_start_idx: usize,
            cycle_stop_idx: usize,
            mass_tolerance: f32,
            mz: f32,
        ) {
            let tol = mz * mass_tolerance * 1e-6;
            for &(pmz, cyc, inten) in &self.mz_peaks {
                if cyc >= cycle_start_idx && cyc < cycle_stop_idx && (pmz - mz).abs() <= tol {
                    let i = cyc - cycle_start_idx;
                    let prev = dense_xic[i];
                    dense_xic[i] = prev + inten;
                    if inten > 0.0 {
                        dense_mz[i] = if prev == 0.0 {
                            pmz
                        } else {
                            (dense_mz[i] * prev + pmz * inten) / (prev + inten)
                        };
                    }
                }
            }
        }

        fn build_1da_crowding_table(&self, _mz_index: &MZIndex) -> Option<Vec<u8>> {
            self.table.clone()
        }
    }

    struct MockDIA {
        mz_index: MZIndex,
        rt_index: RTIndex,
        obs: Vec<MockObs>,
    }

    impl DIADataTrait for MockDIA {
        type QuadrupoleObservation = MockObs;

        fn get_valid_observations(&self, precursor_mz: f32) -> Vec<usize> {
            self.obs
                .iter()
                .enumerate()
                .filter(|(_, o)| precursor_mz >= o.iso[0] && precursor_mz <= o.iso[1])
                .map(|(i, _)| i)
                .collect()
        }

        fn mz_index(&self) -> &MZIndex {
            &self.mz_index
        }

        fn rt_index(&self) -> &RTIndex {
            &self.rt_index
        }

        fn quadrupole_observations(&self) -> &[Self::QuadrupoleObservation] {
            &self.obs
        }

        fn memory_footprint_bytes(&self) -> usize {
            0
        }
    }

    fn make_rt_index(rt_vals: Vec<f32>) -> RTIndex {
        let mut rt_index = RTIndex::new();
        rt_index.rt = Array1::from_vec(rt_vals);
        rt_index
    }

    fn make_mock_precursor() -> Precursor {
        Precursor {
            precursor_idx: 0,
            mz: 500.0,
            mz_library: 500.0,
            rt: 100.0,
            rt_library: 100.0,
            naa: 10,
            charge: 2,
            fragment_mz: vec![300.0, 400.0, 500.0, 600.0, 700.0, 800.0],
            fragment_mz_library: vec![300.0, 400.0, 500.0, 600.0, 700.0, 800.0],
            fragment_intensity: vec![1.0, 0.9, 0.8, 0.7, 0.5, 0.4],
            fragment_cardinality: vec![1; 6],
            fragment_charge: vec![1; 6],
            fragment_loss_type: vec![0; 6],
            fragment_number: vec![4, 5, 6, 7, 8, 9],
            fragment_position: vec![3, 4, 5, 6, 7, 8],
            fragment_type: vec![1, 1, 1, 0, 0, 1],
        }
    }

    #[test]
    fn symbolic_block_has_8_unique_names() {
        let names = SymbolicFeatures::NAMES;
        assert_eq!(names.len(), 8);
        let uniq: HashSet<&str> = names.iter().copied().collect();
        assert_eq!(uniq.len(), 8);
    }

    #[test]
    fn symbolic_features_are_finite_on_zero_input() {
        let rt_vals: Vec<f32> = (0..25).map(|i| i as f32 * 2.0).collect();
        let dia = MockDIA {
            mz_index: MZIndex::new(),
            rt_index: make_rt_index(rt_vals),
            obs: vec![],
        };
        let prec = make_mock_precursor();
        let cand = Candidate {
            precursor_idx: 0,
            rank: 0,
            score: 0.0,
            scan_center: 0,
            scan_start: 0,
            scan_stop: 0,
            cycle_center: 12,
            cycle_start: 8,
            cycle_stop: 17,
        };
        let sf = calculate_symbolic_features(&dia, &prec, &cand, 7.0);
        assert!(sf.values().iter().all(|v| v.is_finite()));
        assert_eq!(sf.sym_tgp_crowding_ppm_top5, 0.0);
        assert_eq!(sf.sym_ctx_specificity_hits, 0.0);
    }

    #[test]
    fn clean_gaussian_peak_scores_stronger_than_noise_on_discriminative_features() {
        let rt_vals: Vec<f32> = (0..30).map(|i| i as f32 * 2.0).collect();
        let prec = make_mock_precursor();

        let mut ms2_peaks = Vec::new();
        for cyc in 8..23 {
            let dt = (cyc as f32 - 15.0) / 1.8;
            let shape = (-0.5 * dt * dt).exp();
            for (f, &fmz) in prec.fragment_mz.iter().enumerate() {
                let lib_i = prec.fragment_intensity[f];
                ms2_peaks.push((fmz, cyc, 10_000.0 * lib_i * shape));
            }
        }
        let clean_dia = MockDIA {
            mz_index: MZIndex::new(),
            rt_index: make_rt_index(rt_vals.clone()),
            obs: vec![MockObs {
                iso: [490.0, 510.0],
                mz_peaks: ms2_peaks,
                table: None,
            }],
        };

        let mut noise_peaks = Vec::new();
        for cyc in 8..23 {
            if cyc % 2 == 0 {
                noise_peaks.push((prec.fragment_mz[0], cyc, 500.0));
            } else {
                noise_peaks.push((prec.fragment_mz[5], cyc, 500.0));
            }
        }
        let noise_dia = MockDIA {
            mz_index: MZIndex::new(),
            rt_index: make_rt_index(rt_vals),
            obs: vec![MockObs {
                iso: [490.0, 510.0],
                mz_peaks: noise_peaks,
                table: None,
            }],
        };

        let cand = Candidate {
            precursor_idx: 0,
            rank: 0,
            score: 10.0,
            scan_center: 0,
            scan_start: 0,
            scan_stop: 0,
            cycle_center: 15,
            cycle_start: 10,
            cycle_stop: 21,
        };
        let sf_clean = calculate_symbolic_features(&clean_dia, &prec, &cand, 7.0);
        let sf_noise = calculate_symbolic_features(&noise_dia, &prec, &cand, 7.0);

        assert!(sf_clean.sym_tgp_crowding_ppm_top5 > 0.8);
        assert!(sf_clean.sym_tgp_crowding_ppm_top5 > sf_noise.sym_tgp_crowding_ppm_top5);
        assert!(sf_clean.sym_tgp_crowding_sqrt_top3 > sf_noise.sym_tgp_crowding_sqrt_top3);
        assert!(sf_clean.sym_interf_trimmed2_entropy > sf_noise.sym_interf_trimmed2_entropy);
        assert!(sf_clean.sym_lib_clean_entropy_full > sf_noise.sym_lib_clean_entropy_full);
        assert!(sf_clean.sym_ctx_specificity_hits > sf_noise.sym_ctx_specificity_hits);
        assert!(sf_clean.sym_nov_n_clean_ppm > sf_noise.sym_nov_n_clean_ppm);
        assert!(sf_clean.sym_mz_n_clean_within_tol > sf_noise.sym_mz_n_clean_within_tol);
        assert_ne!(
            sf_clean.sym_kan_coel_specificity,
            sf_noise.sym_kan_coel_specificity
        );
    }
}
