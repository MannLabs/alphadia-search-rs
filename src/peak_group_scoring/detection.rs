//! Fragment detection block for sparsely sampled, ion-counting MS2 data (final feature set only), from Tomas Jelinek's
//! alphadia-search-rs exp/frag-shadow 65fe5f8, exp/det-llr 9d09ddc and exp/faint-detection f63e876.
//!
//! At low input on Astral a peptide's peak is about one MS2 scan wide and a fragment is either seen (a few ions) or
//! not. Correlations over the candidate window then compare one spike with noise and collapse for faint peptides as
//! for decoys. The block holds three features that do not need several points across the peak:
//! - `shadow_lib_share` (`shadow.rs`): the share of coherent fragments that are the M+1 isotope of another ion,
//! - `det_faint_hits`, `det_faint_miss` (`faint.rs`): the hit and miss terms of an abundance-aware likelihood ratio of
//!   the fragments seen and missed at the apex scan, against each channel's background hit rate.
//!
//! The background hit rate of a channel is measured on flank scans of the same channel, away from the peak, so a hit
//! in a quiet channel counts more than a hit in a channel that common fragments of co-isolated peptides light up all
//! the time. The detection XIC is taken over `FLANK_MAX` cycles on both sides of the candidate apex; hits are
//! intensity > 0. A channel's background rate is (flank hits + PRIOR_WEIGHT * b0) / (flank scans + PRIOR_WEIGHT) with
//! b0 the run's mean flank hit rate, estimated over all candidates alike (no labels).
//!
//! The block is scored only with `ScoringParameters::detection_features`; otherwise every value is 0. All values are
//! finite.

use numpy::ndarray::Array2;

use crate::peak_group_scoring::feature_block::feature_block;

/// Candidates sampled to estimate the run-wide flank hit rate and the background-hit intensity density.
pub const DETECTION_PRIOR_SAMPLE: usize = 20_000;
/// Flank scans start this many cycles from the apex: a peak is ~1 scan wide, so these hold background only.
pub const FLANK_MIN: usize = 4;
/// Flank scans end this many cycles from the apex.
pub const FLANK_MAX: usize = 9;
/// Pseudo-scans of the run-wide rate b0 added to each channel's flank count.
const PRIOR_WEIGHT: f32 = 2.0;
/// Bounds that keep the log-likelihoods finite.
const MIN_PROBABILITY: f32 = 1e-6;

feature_block! {
    /// The fragment detection features of one candidate.
    pub struct DetectionFeatures {
        /// library-weighted share of the coherent fragments that are the M+1 isotope of another ion (`shadow.rs`)
        pub shadow_lib_share: f32,
        /// hit terms of the faint-detection likelihood ratio at its best abundance (`faint.rs`)
        pub det_faint_hits: f32,
        /// miss terms of the faint-detection likelihood ratio at its best abundance, <= 0 (`faint.rs`)
        pub det_faint_miss: f32,
    }
}

impl DetectionFeatures {
    /// Feature values without `ScoringParameters::detection_features`, or for a candidate whose traces do not fit.
    pub fn missing() -> Self {
        Self::default()
    }
}

/// The cycle range [start, stop) of the detection XIC around `apex` in a run of `num_cycles` cycles.
pub fn window(apex: usize, num_cycles: usize) -> (usize, usize) {
    (
        apex.saturating_sub(FLANK_MAX),
        (apex + FLANK_MAX + 1).min(num_cycles),
    )
}

/// Flank hits and flank scans per fragment of a detection XIC whose apex is column `apex_col`.
pub fn flank_counts(xic: &Array2<f32>, apex_col: usize) -> (Vec<u32>, u32) {
    let flank: Vec<usize> = (0..xic.ncols())
        .filter(|&c| {
            let distance = c.abs_diff(apex_col);
            (FLANK_MIN..=FLANK_MAX).contains(&distance)
        })
        .collect();
    let hits = (0..xic.nrows())
        .map(|f| flank.iter().filter(|&&c| xic[[f, c]] > 0.0).count() as u32)
        .collect();
    (hits, flank.len() as u32)
}

/// Intensities of all flank hits of a detection XIC whose apex is column `apex_col`.
pub fn flank_hit_intensities(xic: &Array2<f32>, apex_col: usize) -> Vec<f32> {
    let mut out = Vec::new();
    for c in 0..xic.ncols() {
        if (FLANK_MIN..=FLANK_MAX).contains(&c.abs_diff(apex_col)) {
            out.extend(xic.column(c).iter().filter(|&&v| v > 0.0));
        }
    }
    out
}

/// Each fragment channel's background hit rate: its flank hit share, shrunk towards the run's rate `b0`.
pub fn background_rates(xic: &Array2<f32>, apex_col: usize, b0: f32) -> Vec<f32> {
    let (flank_hits, flank_scans) = flank_counts(xic, apex_col);
    flank_hits
        .iter()
        .map(|&k| clamp((k as f32 + PRIOR_WEIGHT * b0) / (flank_scans as f32 + PRIOR_WEIGHT)))
        .collect()
}

fn clamp(p: f32) -> f32 {
    p.clamp(MIN_PROBABILITY, 1.0 - MIN_PROBABILITY)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 12 fragments over 19 cycles with the apex in column 9.
    fn empty() -> Array2<f32> {
        Array2::zeros((12, 2 * FLANK_MAX + 1))
    }

    #[test]
    fn window_is_clamped_to_the_run() {
        assert_eq!(window(3, 100), (0, 13));
        assert_eq!(window(50, 100), (41, 60));
        assert_eq!(window(95, 100), (86, 100));
    }

    #[test]
    fn flank_counts_skip_the_peak_region() {
        let mut xic = empty();
        xic[[0, 9]] = 5.0; // apex
        xic[[0, 11]] = 5.0; // too close
        xic[[0, 13]] = 5.0; // flank (distance 4)
        xic[[0, 0]] = 5.0; // flank (distance 9)
        let (hits, scans) = flank_counts(&xic, 9);
        assert_eq!(scans, 12);
        assert_eq!(hits[0], 2);
        assert_eq!(hits[1], 0);
    }

    #[test]
    fn background_rates_shrink_flank_shares_towards_the_run_rate() {
        let mut xic = empty();
        for c in [0, 1, 2, 3, 4, 5, 13, 14, 15, 16, 17, 18] {
            xic[[0, c]] = 5.0;
        }
        let rates = background_rates(&xic, 9, 0.1);
        // 12 of 12 flank scans hit: (12 + 2 x 0.1) / 14; no flank hit: 2 x 0.1 / 14
        assert!((rates[0] - 12.2 / 14.0).abs() < 1e-5);
        assert!((rates[1] - 0.2 / 14.0).abs() < 1e-5);
    }

    #[test]
    fn missing_values_are_all_zero() {
        use crate::peak_group_scoring::feature_block::FeatureBlock;
        assert!(DetectionFeatures::missing()
            .values()
            .iter()
            .all(|&v| v == 0.0));
        assert_eq!(DetectionFeatures::NAMES.len(), 3);
    }
}
