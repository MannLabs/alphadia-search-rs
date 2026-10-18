//! MS1 precursor-isotope features of a candidate peak group.
//!
//! The isotope XICs are extracted from the MS1 survey scans over the candidate's cycles, at the
//! (calibrated) precursor m/z plus k * C13 spacing / charge for the library's isotope abundances.

use numpy::ndarray::Array2;

use crate::peak_group_scoring::utils::calculate_correlation_safe;
use crate::utils::weighted_mean_nonzero_axis_1;

/// Mass difference between the 13C and 12C isotopes in Dalton.
pub const C13_MASS_DELTA: f32 = 1.003_354_8;
/// Added before the log10 so that an empty MS1 trace stays finite.
const LOG_EPSILON: f32 = 1.0;

#[derive(Debug, Clone, PartialEq)]
pub struct Ms1Features {
    /// log10 of the library-abundance-weighted sum of isotope intensities over the peak group
    pub log10_ms1_intensity: f32,
    /// Pearson correlation of the library isotope abundances with the observed isotope intensities
    pub ms1_isotope_correlation: f32,
    /// Pearson correlation of the isotope-summed MS1 XIC with the median fragment profile
    pub ms1_profile_correlation: f32,
    /// Mean Pearson correlation of the M+1, M+2, ... XICs with the monoisotopic XIC
    pub ms1_isotope_coelution: f32,
    /// Abundance-weighted mean absolute isotope mass error in ppm; the tolerance if nothing matched
    pub ms1_mass_error: f32,
}

impl Ms1Features {
    /// Feature values for a candidate without MS1 data or without library isotopes.
    pub fn missing(mass_tolerance: f32) -> Self {
        Self {
            log10_ms1_intensity: LOG_EPSILON.log10(),
            ms1_isotope_correlation: 0.0,
            ms1_profile_correlation: 0.0,
            ms1_isotope_coelution: 0.0,
            ms1_mass_error: mass_tolerance,
        }
    }
}

/// m/z of the first `num_isotopes` isotopes of a precursor.
pub fn isotope_mz(mono_mz: f32, charge: u8, num_isotopes: usize) -> Vec<f32> {
    (0..num_isotopes)
        .map(|k| mono_mz + k as f32 * C13_MASS_DELTA / charge as f32)
        .collect()
}

/// MS1 features from the isotope XICs (`dense_xic`, `dense_mz`: [isotope, cycle]).
///
/// `fragment_profile` is the candidate's median fragment profile over the same cycles.
pub fn ms1_features(
    dense_xic: &Array2<f32>,
    dense_mz: &Array2<f32>,
    isotope_mz: &[f32],
    isotope_intensity: &[f32],
    fragment_profile: &[f32],
    mass_tolerance: f32,
) -> Ms1Features {
    let num_isotopes = isotope_mz.len();
    if num_isotopes == 0 || dense_xic.ncols() == 0 {
        return Ms1Features::missing(mass_tolerance);
    }

    let observed: Vec<f32> = dense_xic.rows().into_iter().map(|row| row.sum()).collect();
    let weighted_intensity: f32 = observed
        .iter()
        .zip(isotope_intensity)
        .map(|(o, i)| o * i)
        .sum();

    let summed_xic: Vec<f32> = (0..dense_xic.ncols())
        .map(|c| dense_xic.column(c).sum())
        .collect();

    let mono_xic = dense_xic.row(0).to_vec();
    let coelution: Vec<f32> = (1..num_isotopes)
        .map(|k| calculate_correlation_safe(&mono_xic, &dense_xic.row(k).to_vec()))
        .collect();
    let ms1_isotope_coelution = if coelution.is_empty() {
        0.0
    } else {
        coelution.iter().sum::<f32>() / coelution.len() as f32
    };

    let observed_mz = weighted_mean_nonzero_axis_1(dense_mz, dense_xic);
    let mut error_sum = 0.0f32;
    let mut weight_sum = 0.0f32;
    for k in 0..num_isotopes {
        if observed_mz[k] > 0.0 {
            let error_ppm = (observed_mz[k] - isotope_mz[k]) / isotope_mz[k] * 1e6;
            error_sum += error_ppm.abs() * isotope_intensity[k];
            weight_sum += isotope_intensity[k];
        }
    }
    let ms1_mass_error = if weight_sum > 0.0 {
        error_sum / weight_sum
    } else {
        mass_tolerance
    };

    Ms1Features {
        log10_ms1_intensity: (weighted_intensity + LOG_EPSILON).log10(),
        ms1_isotope_correlation: calculate_correlation_safe(isotope_intensity, &observed),
        ms1_profile_correlation: calculate_correlation_safe(&summed_xic, fragment_profile),
        ms1_isotope_coelution,
        ms1_mass_error,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::arr2;

    #[test]
    fn isotope_mz_uses_charge_spacing() {
        let mz = isotope_mz(500.0, 2, 3);
        assert_eq!(mz.len(), 3);
        assert!((mz[1] - (500.0 + C13_MASS_DELTA / 2.0)).abs() < 1e-4);
        assert!((mz[2] - (500.0 + C13_MASS_DELTA)).abs() < 1e-4);
    }

    #[test]
    fn perfect_envelope_gives_high_correlations_and_small_error() {
        let isotope_mz = [500.0, 500.5017, 501.0034];
        let abundance = [1.0, 0.6, 0.2];
        let profile = [0.0, 1.0, 3.0, 1.0, 0.0];
        let xic = arr2(&[
            [0.0, 10.0, 30.0, 10.0, 0.0],
            [0.0, 6.0, 18.0, 6.0, 0.0],
            [0.0, 2.0, 6.0, 2.0, 0.0],
        ]);
        let mz = arr2(&[
            [0.0, 500.001, 500.001, 500.001, 0.0],
            [0.0, 500.5017, 500.5017, 500.5017, 0.0],
            [0.0, 501.0034, 501.0034, 501.0034, 0.0],
        ]);

        let f = ms1_features(&xic, &mz, &isotope_mz, &abundance, &profile, 4.0);

        assert!(f.ms1_isotope_correlation > 0.99);
        assert!(f.ms1_profile_correlation > 0.99);
        assert!(f.ms1_isotope_coelution > 0.99);
        assert!(f.ms1_mass_error < 1.5);
        let expected = (50.0f32 * 1.0 + 30.0 * 0.6 + 10.0 * 0.2 + 1.0).log10();
        assert!((f.log10_ms1_intensity - expected).abs() < 1e-5);
    }

    #[test]
    fn empty_trace_gives_missing_values() {
        let isotope_mz = [500.0, 500.5, 501.0];
        let abundance = [1.0, 0.6, 0.2];
        let profile = [0.0, 1.0, 3.0];
        let xic = Array2::<f32>::zeros((3, 3));
        let mz = Array2::<f32>::zeros((3, 3));

        let f = ms1_features(&xic, &mz, &isotope_mz, &abundance, &profile, 4.0);

        assert_eq!(f, Ms1Features::missing(4.0));
    }

    #[test]
    fn wrong_isotope_ratio_lowers_isotope_correlation() {
        let isotope_mz = [500.0, 500.5, 501.0];
        let abundance = [1.0, 0.6, 0.2];
        let profile = [1.0, 3.0, 1.0];
        let xic = arr2(&[[1.0, 2.0, 1.0], [5.0, 9.0, 5.0], [10.0, 20.0, 10.0]]);
        let mz = arr2(&[
            [500.0, 500.0, 500.0],
            [500.5, 500.5, 500.5],
            [501.0, 501.0, 501.0],
        ]);

        let f = ms1_features(&xic, &mz, &isotope_mz, &abundance, &profile, 4.0);

        assert!(f.ms1_isotope_correlation < 0.0);
    }

    #[test]
    fn no_isotopes_gives_missing_values() {
        let xic = Array2::<f32>::zeros((0, 3));
        let mz = Array2::<f32>::zeros((0, 3));
        let f = ms1_features(&xic, &mz, &[], &[], &[1.0, 2.0, 1.0], 4.0);
        assert_eq!(f, Ms1Features::missing(4.0));
    }
}
