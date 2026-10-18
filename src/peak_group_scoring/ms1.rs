//! MS1 precursor-isotope features of a candidate peak group.
//!
//! The isotope XICs M-1 .. M+(n-1) are extracted from the MS1 survey scans over the candidate's cycles at the MS1 m/z
//! the library sets per precursor. alphaDIA gives a decoy the m/z and isotope envelope of its target, so MS1 evidence
//! at the precursor mass cannot tell a decoy from a target, only MS1 that agrees with the fragment peak counts.
//! The features look at the cycles around the fragment apex only: the candidate window is several peak widths wide and
//! mostly background.

use numpy::ndarray::Array2;

use crate::peak_group_scoring::utils::calculate_correlation_safe;

/// Mass difference between the 13C and 12C isotopes in Dalton.
pub const C13_MASS_DELTA: f32 = 1.003_354_8;
/// Isotope traces extracted below the monoisotopic one (M-1: another precursor's isotope on our M0).
pub const ISOTOPES_BELOW: usize = 1;
/// Cycles on each side of the fragment apex that count as the apex.
const APEX_HALF_WIDTH: usize = 1;
/// Cycles on each side of the fragment apex excluded from the background.
const BACKGROUND_EXCLUSION: usize = 2;
/// Added before logarithms so that an empty trace stays finite.
const LOG_EPSILON: f32 = 1.0;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Ms1Features {
    /// Cosine of the apex isotope intensities M0.. with the library isotope abundances
    pub ms1_iso_cosine: f32,
    /// Number of isotopes M0.. with signal at the apex
    pub ms1_n_isotopes: f32,
    /// log10 of the monoisotopic intensity summed over the apex cycles
    pub ms1_log_m0_apex: f32,
    /// log10 of the mean apex M0 intensity minus log10 of the mean M0 intensity away from the apex
    pub ms1_contrast: f32,
    /// Share of the apex M-1 intensity in M-1 + M0: an isotope of another precursor on our M0
    pub ms1_m_minus1_share: f32,
    /// Pearson correlation of the M0 XIC with the median fragment profile over the candidate window
    pub ms1_frag_corr: f32,
    /// log2 of the observed apex M1/M0 ratio minus log2 of the library ratio
    pub ms1_iso_m1_err: f32,
    /// Intensity-weighted signed mass error of M0 at the apex in ppm; 0 without M0 signal
    pub ms1_ppm_signed: f32,
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

/// MS1 features from the isotope XICs (`dense_xic`, `dense_mz`: [trace, cycle], traces as in `isotope_mz`).
///
/// `fragment_profile` is the candidate's median fragment profile over the same cycles.
pub fn ms1_features(
    dense_xic: &Array2<f32>,
    dense_mz: &Array2<f32>,
    mono_mz: f32,
    isotope_intensity: &[f32],
    fragment_profile: &[f32],
) -> Ms1Features {
    let num_isotopes = isotope_intensity.len();
    let n_cycles = dense_xic.ncols();
    if num_isotopes == 0
        || n_cycles == 0
        || dense_xic.nrows() != ISOTOPES_BELOW + num_isotopes
        || fragment_profile.len() != n_cycles
    {
        return Ms1Features::missing();
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
    let m_minus1 = apex_sum(0);
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
    let ms1_iso_cosine = if norm > 0.0 { dot / norm } else { 0.0 };

    let ms1_n_isotopes = apex_isotopes.iter().filter(|&&a| a > 0.0).count() as f32;

    let mono_row = ISOTOPES_BELOW;
    let apex_mean = m0 / (stop - start) as f32;
    let background: Vec<f32> = (0..n_cycles)
        .filter(|&c| c + BACKGROUND_EXCLUSION < apex || c > apex + BACKGROUND_EXCLUSION)
        .map(|c| dense_xic[[mono_row, c]])
        .collect();
    let background_mean = if background.is_empty() {
        0.0
    } else {
        background.iter().sum::<f32>() / background.len() as f32
    };
    let ms1_contrast = (apex_mean + LOG_EPSILON).log10() - (background_mean + LOG_EPSILON).log10();

    let ms1_m_minus1_share = if m_minus1 + m0 > 0.0 {
        m_minus1 / (m_minus1 + m0)
    } else {
        0.0
    };

    let mono_xic = dense_xic.row(mono_row).to_vec();
    let ms1_frag_corr = calculate_correlation_safe(&mono_xic, fragment_profile);

    let ms1_iso_m1_err =
        if m0 > 0.0 && num_isotopes > 1 && isotope_intensity[0] > 0.0 && isotope_intensity[1] > 0.0
        {
            ((apex_isotopes[1] + LOG_EPSILON) / (m0 + LOG_EPSILON)).log2()
                - (isotope_intensity[1] / isotope_intensity[0]).log2()
        } else {
            0.0
        };

    let mut error_sum = 0.0f32;
    let mut weight_sum = 0.0f32;
    for c in start..stop {
        let intensity = dense_xic[[mono_row, c]];
        if intensity > 0.0 {
            error_sum += (dense_mz[[mono_row, c]] - mono_mz) / mono_mz * 1e6 * intensity;
            weight_sum += intensity;
        }
    }
    let ms1_ppm_signed = if weight_sum > 0.0 {
        error_sum / weight_sum
    } else {
        0.0
    };

    Ms1Features {
        ms1_iso_cosine,
        ms1_n_isotopes,
        ms1_log_m0_apex: (m0 + LOG_EPSILON).log10(),
        ms1_contrast,
        ms1_m_minus1_share,
        ms1_frag_corr,
        ms1_iso_m1_err,
        ms1_ppm_signed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

        let f = ms1_features(&xic, &mz, 500.0, &abundance, &profile);

        assert!(f.ms1_iso_cosine > 0.99);
        assert_eq!(f.ms1_n_isotopes, 3.0);
        assert!((f.ms1_log_m0_apex - 51.0f32.log10()).abs() < 1e-5);
        assert!(f.ms1_contrast > 0.8);
        assert_eq!(f.ms1_m_minus1_share, 0.0);
        assert!(f.ms1_frag_corr > 0.95);
        assert!(f.ms1_iso_m1_err.abs() < 0.05);
        assert!((f.ms1_ppm_signed - 2.0).abs() < 0.1);
    }

    #[test]
    fn signal_away_from_the_apex_does_not_count() {
        let abundance = [1.0, 0.6];
        let profile = [0.0, 1.0, 3.0, 1.0, 0.0, 0.0, 0.0];
        let xic = arr2(&[
            [0.0; 7],
            [0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 40.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 24.0],
        ]);
        let mz = Array2::<f32>::zeros((3, 7));

        let f = ms1_features(&xic, &mz, 500.0, &abundance, &profile);

        assert_eq!(f.ms1_n_isotopes, 0.0);
        assert_eq!(f.ms1_log_m0_apex, 0.0);
        assert!(f.ms1_contrast < 0.0);
        assert_eq!(f.ms1_ppm_signed, 0.0);
    }

    #[test]
    fn another_precursors_isotope_raises_the_m_minus1_share() {
        let abundance = [1.0, 0.6];
        let profile = [1.0, 3.0, 1.0];
        let xic = arr2(&[[30.0, 90.0, 30.0], [10.0, 30.0, 10.0], [6.0, 18.0, 6.0]]);
        let mz = Array2::<f32>::zeros((3, 3));

        let f = ms1_features(&xic, &mz, 500.0, &abundance, &profile);

        assert!((f.ms1_m_minus1_share - 0.75).abs() < 1e-5);
    }

    #[test]
    fn no_isotopes_or_mismatched_traces_give_missing_values() {
        let xic = Array2::<f32>::zeros((1, 3));
        let mz = Array2::<f32>::zeros((1, 3));
        assert_eq!(
            ms1_features(&xic, &mz, 500.0, &[], &[1.0, 2.0, 1.0]),
            Ms1Features::missing()
        );
        assert_eq!(
            ms1_features(&xic, &mz, 500.0, &[1.0, 0.5], &[1.0, 2.0, 1.0]),
            Ms1Features::missing()
        );
    }
}
