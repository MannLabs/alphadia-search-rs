//! MS1 precursor-isotope features of a candidate peak group (final feature set only).
//!
//! The isotope XICs M-1 .. M+(n-1) are extracted from the MS1 survey scans over the candidate's cycles at the MS1 m/z
//! the library sets per precursor (alphaDIA: every precursor at its own m/z, shifted by its MS1 calibration), at the
//! calibrated MS1 tolerance. The block combines four views on them:
//! - the whole candidate window against the median fragment profile (`ms1_profile`),
//! - the isotope ratio M+1/M0 at the fragment apex (`ms1_iso_m1_err`, `calculate_ms1_apex_features` here),
//! - the share of apex scans with an M0 / M+1 peak close to the calibrated m/z (`ms1_accuracy`),
//! - the M0 / M+1 points of each MS1 scan at its own time within the cycle (`ms1_fine`).
//!
//! The ten features are the ones a feature selection kept from a larger pool (own MS1 profile features and Tomas
//! Jelinek's exp/* MS1 features) on HeLa 2-12 min gradients, plasma and ProteoBench.
//!
//! A candidate without library isotopes (`SpecLibFlat::set_isotopes` not called) or a run without MS1 scans gets 0 in
//! every feature. All values are finite, never NaN, because NaN rows are dropped before the classifier.

use numpy::ndarray::Array2;

use crate::peak_group_scoring::feature_block::feature_block;

/// Mass difference between the 13C and 12C isotopes in Dalton.
pub const C13_MASS_DELTA: f32 = 1.003_354_8;
/// Isotope traces extracted below the monoisotopic one (M-1: another precursor's isotope on our M0).
pub const ISOTOPES_BELOW: usize = 1;
/// Cycles on each side of the fragment apex that count as the apex.
pub const APEX_HALF_WIDTH: usize = 1;
/// Added before logarithms so that an empty trace stays finite.
const LOG_EPSILON: f32 = 1.0;

feature_block! {
    /// The MS1 features of one candidate.
    pub struct Ms1Features {
        /// 1 if the M0 XIC has signal at the apex of the fragment profile, else 0
        pub ms1_signal_at_apex: f32,
        /// mean Pearson correlation of the M+1 and M+2 XICs with the median fragment profile
        pub ms1_isotope_correlation: f32,
        /// cosine of the profile-fitted M0..M+2 intensities with the library isotope abundances
        pub ms1_isotope_cosine: f32,
        /// profile-fitted M-1 intensity over that of M-1 and M0 (M0 is another precursor's M+1)
        pub ms1_m_minus_1_frac: f32,
        /// log10(1 + profile-fitted M0 intensity at the profile maximum)
        pub ms1_log_intensity: f32,
        /// log2 of the observed apex M+1/M0 ratio minus log2 of the library ratio
        pub ms1_iso_m1_err: f32,
        /// share of the apex points (MS1 scan x apex cycle) with an M0 peak within `ACCURATE_PPM`
        pub ms1_m0_accurate_share: f32,
        /// share of the apex points with an M+1 peak within `ACCURATE_PPM`
        pub ms1_m1_accurate_share: f32,
        /// Pearson correlation of the per-scan M0 points with a Gaussian of the fragment FWHM at the fragment apex time
        pub ms1_fine_shape: f32,
        /// Pearson correlation of the per-scan M0 and M+1 points
        pub ms1_fine_iso_corr: f32,
    }
}

impl Ms1Features {
    /// Feature values for a candidate without MS1 data or without library isotopes.
    pub fn missing() -> Self {
        Self::default()
    }
}

/// m/z of the isotope traces M-ISOTOPES_BELOW .. M+(num_isotopes-1) of a precursor.
pub fn isotope_mz(mono_mz: f32, charge: u8, num_isotopes: usize) -> Vec<f32> {
    (0..ISOTOPES_BELOW + num_isotopes)
        .map(|k| mono_mz + (k as f32 - ISOTOPES_BELOW as f32) * C13_MASS_DELTA / charge as f32)
        .collect()
}

pub fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

/// The apex view on the isotope XICs (Tomas Jelinek's MS1 block, alphadia-search-rs c4ece50 / 59d1892 / 6a014e1):
/// the isotopes summed over the apex cycles of the fragment profile. `iso_m1_err` is the MS1 block's `ms1_iso_m1_err`;
/// the others are what alphaDIA's MS1 calibration pre-pass needs to pick confident precursors and their mass error.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Ms1ApexFeatures {
    /// cosine of the apex isotope intensities M0.. with the library isotope abundances
    pub iso_cosine: f32,
    /// number of isotopes M0.. with signal at the apex
    pub n_isotopes: f32,
    /// log10 of the monoisotopic intensity summed over the apex cycles
    pub log_m0_apex: f32,
    /// log2 of the observed apex M+1/M0 ratio minus log2 of the library ratio; 0 without M0 or library M+1
    pub iso_m1_err: f32,
    /// intensity-weighted signed mass error of M0 at the apex in ppm; 0 without M0 signal
    pub ppm_signed: f32,
}

/// The apex features from the isotope XICs (`dense_xic`, `dense_mz`: [trace, cycle], traces as in `isotope_mz`), all
/// 0 if the traces do not fit.
///
/// `fragment_profile` is the candidate's median fragment profile over the same cycles.
pub fn calculate_ms1_apex_features(
    dense_xic: &Array2<f32>,
    dense_mz: &Array2<f32>,
    mono_mz: f32,
    isotope_intensity: &[f32],
    fragment_profile: &[f32],
) -> Ms1ApexFeatures {
    let num_isotopes = isotope_intensity.len();
    let n_cycles = dense_xic.ncols();
    if num_isotopes == 0
        || n_cycles == 0
        || dense_xic.nrows() != ISOTOPES_BELOW + num_isotopes
        || fragment_profile.len() != n_cycles
    {
        return Ms1ApexFeatures::default();
    }

    let apex = fragment_profile
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let start = apex.saturating_sub(APEX_HALF_WIDTH);
    let stop = (apex + APEX_HALF_WIDTH + 1).min(n_cycles);

    let apex_sum = |trace: usize| -> f32 { (start..stop).map(|c| dense_xic[[trace, c]]).sum() };
    let apex_isotopes: Vec<f32> = (0..num_isotopes)
        .map(|k| apex_sum(ISOTOPES_BELOW + k))
        .collect();
    let m0 = apex_isotopes[0];

    let dot: f32 = apex_isotopes
        .iter()
        .zip(isotope_intensity)
        .map(|(a, i)| a * i)
        .sum();
    let norm = apex_isotopes.iter().map(|a| a * a).sum::<f32>().sqrt()
        * isotope_intensity.iter().map(|i| i * i).sum::<f32>().sqrt();
    let iso_cosine = if norm > 0.0 { dot / norm } else { 0.0 };

    let n_isotopes = apex_isotopes.iter().filter(|&&a| a > 0.0).count() as f32;

    let iso_m1_err =
        if m0 > 0.0 && num_isotopes > 1 && isotope_intensity[0] > 0.0 && isotope_intensity[1] > 0.0
        {
            ((apex_isotopes[1] + LOG_EPSILON) / (m0 + LOG_EPSILON)).log2()
                - (isotope_intensity[1] / isotope_intensity[0]).log2()
        } else {
            0.0
        };

    let mono_row = ISOTOPES_BELOW;
    let mut error_sum = 0.0f32;
    let mut weight_sum = 0.0f32;
    for c in start..stop {
        let intensity = dense_xic[[mono_row, c]];
        if intensity > 0.0 {
            error_sum += (dense_mz[[mono_row, c]] - mono_mz) / mono_mz * 1e6 * intensity;
            weight_sum += intensity;
        }
    }
    let ppm_signed = if weight_sum > 0.0 {
        error_sum / weight_sum
    } else {
        0.0
    };

    Ms1ApexFeatures {
        iso_cosine: finite_or_zero(iso_cosine),
        n_isotopes,
        log_m0_apex: finite_or_zero((m0 + LOG_EPSILON).log10()),
        iso_m1_err: finite_or_zero(iso_m1_err),
        ppm_signed: finite_or_zero(ppm_signed),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::peak_group_scoring::feature_block::FeatureBlock;
    use numpy::ndarray::arr2;

    #[test]
    fn isotope_mz_starts_one_isotope_below_and_uses_charge_spacing() {
        let mz = isotope_mz(500.0, 2, 3);
        assert_eq!(mz.len(), 4);
        assert!((mz[0] - (500.0 - C13_MASS_DELTA / 2.0)).abs() < 1e-4);
        assert!((mz[1] - 500.0).abs() < 1e-4);
        assert!((mz[3] - (500.0 + C13_MASS_DELTA)).abs() < 1e-4);
    }

    #[test]
    fn coeluting_envelope_at_the_apex_scores_well() {
        let abundance = [1.0, 0.6, 0.2];
        let profile = [0.0, 0.0, 1.0, 3.0, 1.0, 0.0, 0.0];
        let xic = arr2(&[
            [5.0, 5.0, 0.0, 0.0, 0.0, 5.0, 5.0],
            [1.0, 1.0, 10.0, 30.0, 10.0, 1.0, 1.0],
            [0.0, 0.0, 6.0, 18.0, 6.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 6.0, 2.0, 0.0, 0.0],
        ]);
        let mz = xic.mapv(|x| if x > 0.0 { 500.001 } else { 0.0 });

        let f = calculate_ms1_apex_features(&xic, &mz, 500.0, &abundance, &profile);

        assert!(f.iso_cosine > 0.99);
        assert_eq!(f.n_isotopes, 3.0);
        assert!((f.log_m0_apex - 51.0f32.log10()).abs() < 1e-5);
        assert!(f.iso_m1_err.abs() < 0.05);
        assert!((f.ppm_signed - 2.0).abs() < 0.1);
    }

    #[test]
    fn too_intense_m1_gives_a_positive_isotope_error() {
        let profile = [1.0, 3.0, 1.0];
        let xic = arr2(&[[0.0; 3], [10.0, 30.0, 10.0], [20.0, 60.0, 20.0]]);
        let mz = Array2::<f32>::zeros((3, 3));

        let f = calculate_ms1_apex_features(&xic, &mz, 500.0, &[1.0, 0.5], &profile);

        // observed M+1/M0 about 2, library 0.5: log2(2) - log2(0.5) = 2
        assert!((f.iso_m1_err - 2.0).abs() < 0.05);
    }

    #[test]
    fn signal_away_from_the_apex_does_not_count() {
        let profile = [0.0, 1.0, 3.0, 1.0, 0.0, 0.0, 0.0];
        let xic = arr2(&[
            [0.0; 7],
            [0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 40.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 24.0],
        ]);
        let mz = Array2::<f32>::zeros((3, 7));

        let f = calculate_ms1_apex_features(&xic, &mz, 500.0, &[1.0, 0.6], &profile);

        assert_eq!(f.n_isotopes, 0.0);
        assert_eq!(f.log_m0_apex, 0.0);
        assert_eq!(f.iso_m1_err, 0.0);
        assert_eq!(f.ppm_signed, 0.0);
    }

    #[test]
    fn no_isotopes_mismatched_or_empty_traces_give_zero() {
        let xic = Array2::<f32>::zeros((1, 3));
        let mz = Array2::<f32>::zeros((1, 3));
        let zero = Ms1ApexFeatures::default();
        assert_eq!(
            calculate_ms1_apex_features(&xic, &mz, 500.0, &[], &[1.0, 2.0, 1.0]),
            zero
        );
        assert_eq!(
            calculate_ms1_apex_features(&xic, &mz, 500.0, &[1.0, 0.5], &[1.0, 2.0, 1.0]),
            zero
        );
        let empty = Array2::<f32>::zeros((3, 5));
        let empty_mz = Array2::<f32>::zeros((3, 5));
        assert_eq!(
            calculate_ms1_apex_features(&empty, &empty_mz, 500.0, &[1.0, 0.5], &[0.0; 5]),
            zero
        );
    }

    #[test]
    fn missing_values_are_all_zero() {
        assert!(Ms1Features::missing().values().iter().all(|&v| v == 0.0));
        assert_eq!(Ms1Features::NAMES.len(), 10);
    }
}
