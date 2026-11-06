//! Fragment detection features for sparsely sampled, ion-counting MS2 data.
//!
//! At low input on Astral a peptide's peak is about one MS2 scan wide and a fragment is either seen (a few ions) or
//! not. Correlations over the candidate window then compare one spike with noise. These features instead ask which
//! fragment channels are hit at the apex scan, against how often each channel is hit anyway: the background hit rate
//! of a channel is measured on flank scans of the same channel, away from the peak, so a hit in a quiet channel counts
//! more than a hit in a channel that common fragments of co-isolated peptides light up all the time.
//!
//! The XIC is taken over `FLANK_MAX` cycles on both sides of the candidate apex. Hits are intensity > 0. A channel's
//! background rate is (flank hits + PRIOR_WEIGHT * b0) / (flank scans + PRIOR_WEIGHT) with b0 the run's mean flank hit
//! rate, estimated over all candidates alike (no labels).

use numpy::ndarray::Array2;

/// Flank scans start this many cycles from the apex: a peak is ~1 scan wide, so these hold background only.
pub const FLANK_MIN: usize = 4;
/// Flank scans end this many cycles from the apex.
pub const FLANK_MAX: usize = 9;
/// Pseudo-scans of the run-wide rate b0 added to each channel's flank count.
const PRIOR_WEIGHT: f32 = 2.0;
/// Expected signal on the two scans next to the apex relative to the apex scan.
const NEIGHBOUR_WEIGHT: f32 = 0.1;
/// Bounds that keep the log-likelihoods finite.
const MIN_PROBABILITY: f32 = 1e-6;
/// Expected ion counts of the strongest fragment over which the signal model is maximised (log-spaced).
const NUM_SCALES: usize = 15;
const MIN_SCALE: f32 = 0.03;
const MAX_SCALE: f32 = 100.0;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct DetectionFeatures {
    /// max over the signal scale of the hit/miss log-likelihood ratio, signal + background vs background, over the
    /// apex scan and its two neighbours
    pub det_llr: f32,
    /// the hit terms of `det_llr` at its best scale (no evidence from missed fragments)
    pub det_llr_hits: f32,
    /// library-share-weighted sum over the fragments hit at the apex scan of -ln(background rate)
    pub det_bg_surprise: f32,
}

impl DetectionFeatures {
    /// Feature values for a candidate whose apex lies outside the extracted window.
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

fn clamp(p: f32) -> f32 {
    p.clamp(MIN_PROBABILITY, 1.0 - MIN_PROBABILITY)
}

/// Detection features of one candidate.
///
/// `xic`: [fragment, cycle] over `window(apex, ...)`, `apex_col`: the apex column in it, `library_intensity`: per
/// fragment, `b0`: the run's mean flank hit rate.
pub fn detection_features(
    xic: &Array2<f32>,
    apex_col: usize,
    library_intensity: &[f32],
    b0: f32,
) -> DetectionFeatures {
    let n_fragments = xic.nrows();
    if n_fragments == 0 || apex_col >= xic.ncols() || library_intensity.len() != n_fragments {
        return DetectionFeatures::missing();
    }
    let top = library_intensity.iter().cloned().fold(0.0f32, f32::max);
    let total: f32 = library_intensity.iter().sum();
    if top <= 0.0 {
        return DetectionFeatures::missing();
    }

    let (flank_hits, flank_scans) = flank_counts(xic, apex_col);
    let background: Vec<f32> = flank_hits
        .iter()
        .map(|&k| clamp((k as f32 + PRIOR_WEIGHT * b0) / (flank_scans as f32 + PRIOR_WEIGHT)))
        .collect();

    let scans: Vec<(usize, f32)> = [
        (apex_col.checked_sub(1), NEIGHBOUR_WEIGHT),
        (Some(apex_col), 1.0),
        (Some(apex_col + 1), NEIGHBOUR_WEIGHT),
    ]
    .into_iter()
    .filter_map(|(c, g)| c.filter(|&c| c < xic.ncols()).map(|c| (c, g)))
    .collect();

    let log_step = (MAX_SCALE / MIN_SCALE).ln() / (NUM_SCALES - 1) as f32;
    let mut best = (f32::NEG_INFINITY, 0.0f32);
    for s in 0..NUM_SCALES {
        let scale = MIN_SCALE * (log_step * s as f32).exp();
        let mut llr = 0.0f32;
        let mut hits = 0.0f32;
        for f in 0..n_fragments {
            let b = background[f];
            let relative = library_intensity[f] / top;
            for &(c, g) in &scans {
                let p1 = clamp(1.0 - (1.0 - b) * (-scale * relative * g).exp());
                if xic[[f, c]] > 0.0 {
                    let term = (p1 / b).ln();
                    llr += term;
                    hits += term;
                } else {
                    llr += ((1.0 - p1) / (1.0 - b)).ln();
                }
            }
        }
        if llr > best.0 {
            best = (llr, hits);
        }
    }

    let det_bg_surprise = (0..n_fragments)
        .filter(|&f| xic[[f, apex_col]] > 0.0)
        .map(|f| library_intensity[f] / total * -background[f].ln())
        .sum();

    DetectionFeatures {
        det_llr: best.0,
        det_llr_hits: best.1,
        det_bg_surprise,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 12 fragments over 19 cycles with the apex in column 9.
    fn empty() -> Array2<f32> {
        Array2::zeros((12, 2 * FLANK_MAX + 1))
    }

    fn library() -> Vec<f32> {
        (0..12).map(|f| 1.0 - f as f32 * 0.07).collect()
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
    fn strong_fragments_hit_at_the_apex_score_above_an_empty_candidate() {
        let mut xic = empty();
        for f in 0..8 {
            xic[[f, 9]] = 300.0;
        }
        let hit = detection_features(&xic, 9, &library(), 0.09);
        let none = detection_features(&empty(), 9, &library(), 0.09);
        assert!(hit.det_llr > 5.0);
        assert!(hit.det_llr > none.det_llr);
        assert!(hit.det_llr_hits > 0.0);
        assert!(hit.det_bg_surprise > 1.0);
        assert_eq!(none.det_bg_surprise, 0.0);
    }

    #[test]
    fn hits_in_busy_channels_count_less() {
        let mut quiet = empty();
        let mut busy = empty();
        for f in 0..4 {
            quiet[[f, 9]] = 300.0;
            busy[[f, 9]] = 300.0;
            for c in [0, 1, 2, 3, 4, 5, 13, 14, 15, 16, 17, 18] {
                busy[[f, c]] = 300.0;
            }
        }
        let q = detection_features(&quiet, 9, &library(), 0.09);
        let b = detection_features(&busy, 9, &library(), 0.09);
        assert!(q.det_llr > b.det_llr);
        assert!(q.det_bg_surprise > b.det_bg_surprise);
    }

    #[test]
    fn missing_the_strongest_fragments_costs_more_than_missing_weak_ones() {
        let mut weak_missing = empty();
        let mut strong_missing = empty();
        for f in 0..6 {
            weak_missing[[f, 9]] = 300.0;
            strong_missing[[11 - f, 9]] = 300.0;
        }
        let w = detection_features(&weak_missing, 9, &library(), 0.09);
        let s = detection_features(&strong_missing, 9, &library(), 0.09);
        assert!(w.det_llr > s.det_llr);
    }

    #[test]
    fn invalid_inputs_give_missing_values() {
        assert_eq!(
            detection_features(&empty(), 30, &library(), 0.09),
            DetectionFeatures::missing()
        );
        assert_eq!(
            detection_features(&empty(), 9, &[1.0, 2.0], 0.09),
            DetectionFeatures::missing()
        );
    }
}
