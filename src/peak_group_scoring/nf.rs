//! Unfragmented-precursor ("NF") features: the precursor ion that survives fragmentation, read in the candidate's own
//! MS2 scans.
//!
//! The NF trace is sampled in the same scans as the fragments and only sees the narrow isolation window, so at low
//! input it is far less crowded than the full MS1 scan and has the same time grid as the fragment profile. It is
//! measured by the MS2 analyzer, so its m/z gets the MS2 calibration of the precursor's own fragments, not the MS1
//! calibration. Decoys are read at their own m/z, never at their target's.

use numpy::ndarray::{s, Array2};

use crate::peak_group_scoring::ms1::C13_MASS_DELTA;
use crate::peak_group_scoring::utils::calculate_correlation_safe;

/// Cycles on each side of the fragment apex that count as the apex.
const APEX_HALF_WIDTH: usize = 1;
/// Added before logarithms so that an empty trace stays finite.
const LOG_EPSILON: f32 = 1.0;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct NfFeatures {
    /// Pearson correlation of the NF trace with the median fragment profile over the candidate window
    pub nf_corr: f32,
    /// log10 of the apex NF intensity minus log10 of the apex fragment intensity summed over fragments
    pub nf_log_ratio: f32,
    /// 1 if the NF trace has signal at the apex, else 0
    pub nf_present: f32,
    /// Pearson correlation of the NF M+1 trace with the NF trace; 0 if M+1 lies outside the isolation window
    pub nf_iso_corr: f32,
}

impl NfFeatures {
    /// Feature values for a candidate whose traces do not match the fragment profile.
    pub fn missing() -> Self {
        Self::default()
    }
}

/// The library precursor m/z moved by the median relative calibration shift of the precursor's fragments.
pub fn ms2_calibrated_mz(mz_library: f32, fragment_mz: &[f32], fragment_mz_library: &[f32]) -> f32 {
    let mut ratios: Vec<f32> = fragment_mz
        .iter()
        .zip(fragment_mz_library)
        .filter(|(_, &library)| library > 0.0)
        .map(|(&calibrated, &library)| calibrated / library)
        .collect();
    if ratios.is_empty() {
        return mz_library;
    }
    ratios.sort_by(|a, b| a.total_cmp(b));
    let mid = ratios.len() / 2;
    let median = if ratios.len() % 2 == 0 {
        (ratios[mid - 1] + ratios[mid]) / 2.0
    } else {
        ratios[mid]
    };
    mz_library * median
}

/// m/z of the M+1 isotope of a precursor of `charge` at `mz`.
pub fn m1_mz(mz: f32, charge: u8) -> f32 {
    mz + C13_MASS_DELTA / charge as f32
}

/// NF features of one candidate.
///
/// `nf`: the NF trace, `nf1`: the NF M+1 trace (None if it was not isolated with the precursor), `fragment_xic`:
/// [fragment, cycle] raw fragment intensities, `fragment_profile`: the median fragment profile, all over the same
/// cycles.
pub fn nf_features(
    nf: &[f32],
    nf1: Option<&[f32]>,
    fragment_xic: &Array2<f32>,
    fragment_profile: &[f32],
) -> NfFeatures {
    let n_cycles = fragment_profile.len();
    if n_cycles == 0 || nf.len() != n_cycles || fragment_xic.ncols() != n_cycles {
        return NfFeatures::missing();
    }

    let apex = fragment_profile
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let start = apex.saturating_sub(APEX_HALF_WIDTH);
    let stop = (apex + APEX_HALF_WIDTH + 1).min(n_cycles);

    let nf_apex: f32 = nf[start..stop].iter().sum();
    let fragment_apex: f32 = fragment_xic.slice(s![.., start..stop]).sum();

    let nf_iso_corr = match nf1 {
        Some(nf1) if nf1.len() == n_cycles => calculate_correlation_safe(nf1, nf),
        _ => 0.0,
    };

    NfFeatures {
        nf_corr: calculate_correlation_safe(nf, fragment_profile),
        nf_log_ratio: (nf_apex + LOG_EPSILON).log10() - (fragment_apex + LOG_EPSILON).log10(),
        nf_present: if nf_apex > 0.0 { 1.0 } else { 0.0 },
        nf_iso_corr,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::arr2;

    fn fragments() -> Array2<f32> {
        arr2(&[
            [0.0, 0.0, 10.0, 40.0, 10.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 20.0, 5.0, 0.0, 0.0],
        ])
    }

    const PROFILE: [f32; 7] = [0.0, 0.0, 0.25, 1.0, 0.25, 0.0, 0.0];

    #[test]
    fn calibrated_mz_follows_the_median_fragment_shift() {
        let library = [300.0, 500.0, 700.0];
        let calibrated = [
            300.0 * (1.0 + 4e-6),
            500.0 * (1.0 + 5e-6),
            700.0 * (1.0 + 9e-6),
        ];
        let mz = ms2_calibrated_mz(600.0, &calibrated, &library);
        assert!(((mz / 600.0 - 1.0) * 1e6 - 5.0).abs() < 0.1);
        assert_eq!(ms2_calibrated_mz(600.0, &[], &[]), 600.0);
    }

    #[test]
    fn m1_uses_charge_spacing() {
        assert!((m1_mz(500.0, 2) - (500.0 + C13_MASS_DELTA / 2.0)).abs() < 1e-4);
        assert!((m1_mz(500.0, 3) - (500.0 + C13_MASS_DELTA / 3.0)).abs() < 1e-4);
    }

    #[test]
    fn coeluting_precursor_scores_well() {
        let nf = [0.0, 0.0, 3.0, 10.0, 3.0, 0.0, 0.0];
        let nf1 = [0.0, 0.0, 2.0, 6.0, 2.0, 0.0, 0.0];
        let f = nf_features(&nf, Some(&nf1), &fragments(), &PROFILE);
        assert!(f.nf_corr > 0.99);
        assert!(f.nf_iso_corr > 0.99);
        assert_eq!(f.nf_present, 1.0);
        assert!((f.nf_log_ratio - (17.0f32.log10() - 91.0f32.log10())).abs() < 1e-5);
    }

    #[test]
    fn absent_precursor_gives_zero_correlations_and_a_negative_ratio() {
        let nf = [0.0; 7];
        let f = nf_features(&nf, Some(&nf), &fragments(), &PROFILE);
        assert_eq!(f.nf_corr, 0.0);
        assert_eq!(f.nf_iso_corr, 0.0);
        assert_eq!(f.nf_present, 0.0);
        assert!(f.nf_log_ratio < -1.5);
    }

    #[test]
    fn signal_off_the_apex_is_not_present() {
        let nf = [8.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let f = nf_features(&nf, None, &fragments(), &PROFILE);
        assert_eq!(f.nf_present, 0.0);
        assert!(f.nf_corr < 0.0);
        assert_eq!(f.nf_iso_corr, 0.0);
    }

    #[test]
    fn mismatched_lengths_give_missing_values() {
        let nf = [1.0; 5];
        assert_eq!(
            nf_features(&nf, None, &fragments(), &PROFILE),
            NfFeatures::missing()
        );
    }
}
