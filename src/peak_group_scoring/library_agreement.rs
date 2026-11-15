//! Whether the observed intensities of a candidate's fragments match the library intensities.
//!
//! Interfered fragments distort the observed spectrum, so most of these features also compare the spectra over the
//! co-eluting fragments only, or with each fragment weighted by its correlation. In fragment XICs of ARATH entrapment
//! runs they separate real peptides from false targets beyond correlation and intensity summaries, also when trained
//! against decoys only.
//!
//! The fitted spread (`fit_`) fits each fragment XIC as `a_i * m` to the candidate's median elution profile `m`
//! (`a_i >= 0`, least squares), so `a_i * sum(m)` is its co-eluting intensity and a peak borrowed from another
//! precursor drops out. Its features describe how |fitted - library| is spread over the fragments after scaling both
//! to unit sum: evenly small for a correct candidate, a few large misses for a wrong one.

use numpy::ndarray::Array2;

use crate::constants::FragmentType;
use crate::peak_group_scoring::feature_block::feature_block;
use crate::peak_group_scoring::utils::{
    calculate_correlation_safe, shares, COELUTION_MIN_CORRELATION,
};
use crate::utils::calculate_std;

/// Minimum number of y ions for the y-ion intensity correlation
pub const MIN_Y_IONS_FOR_CORRELATION: usize = 3;
/// Cycles on each side of the median-profile apex summed for the apex spectrum
pub const LIB_COSINE_APEX_HALF_WIDTH: usize = 1;
/// Minimum number of co-eluting fragments for the library cosine over the co-eluting fragments
pub const LIB_COSINE_MIN_FRAGMENTS: usize = 3;

feature_block! {
    /// The library-agreement features of one candidate.
    pub struct LibraryAgreementFeatures {
        /// standard deviation of log2(observed share / library share) over the observed fragments
        pub log_ratio_std: f32,
        /// observed vs library intensity correlation over the y ions, 0 with fewer than MIN_Y_IONS_FOR_CORRELATION
        pub y_intensity_correlation: f32,
        /// cosine of the apex spectrum with the library over the co-eluting fragments, 0 with fewer than
        /// LIB_COSINE_MIN_FRAGMENTS of them
        pub lib_cosine_over_50: f32,
        /// spectral entropy similarity (Li et al. 2021) of observed and library intensities
        pub ent_sim: f32,
        /// the same with each observed intensity weighted by its (non-negative) correlation
        pub ent_sim_corrweighted: f32,
        /// mean of |fitted - library| over all fragments, both scaled to unit sum
        pub fit_intensity_abs_diff_mean: f32,
        /// sample standard deviation of |fitted - library| over all fragments
        pub fit_intensity_abs_diff_std: f32,
        /// maximum of |fitted - library| over all fragments
        pub fit_intensity_abs_diff_max: f32,
        /// sample standard deviation of |fitted - library| over the y ions, scaled to unit sum over all fragments
        pub fit_y_intensity_abs_diff_std: f32,
    }
}

/// The library-agreement features of one candidate.
///
/// `dense_xic` holds the fragment XICs (fragments x cycles) and `median_profile` their median profile. The slices
/// hold one value per scored fragment: library intensity, observed intensity (summed over the candidate's cycles),
/// correlation with the median profile and ion type.
pub fn calculate_library_agreement_features(
    dense_xic: &Array2<f32>,
    median_profile: &[f32],
    library_intensities: &[f32],
    observation_intensities: &[f32],
    correlations: &[f32],
    fragment_types: &[u8],
) -> LibraryAgreementFeatures {
    let n = library_intensities.len();
    if n == 0
        || observation_intensities.len() != n
        || correlations.len() != n
        || fragment_types.len() != n
    {
        return LibraryAgreementFeatures::default();
    }
    let library: Vec<f64> = library_intensities.iter().map(|&x| x as f64).collect();
    let observed: Vec<f64> = observation_intensities.iter().map(|&x| x as f64).collect();
    let observed_corrweighted: Vec<f64> = (0..n)
        .map(|i| observed[i] * (correlations[i] as f64).max(0.0))
        .collect();
    LibraryAgreementFeatures {
        log_ratio_std: log_ratio_std(library_intensities, observation_intensities),
        y_intensity_correlation: y_intensity_correlation(
            fragment_types,
            library_intensities,
            observation_intensities,
        ),
        lib_cosine_over_50: lib_cosine_over_50(
            dense_xic,
            median_profile,
            correlations,
            library_intensities,
        ),
        ent_sim: entropy_similarity(&library, &observed) as f32,
        ent_sim_corrweighted: entropy_similarity(&library, &observed_corrweighted) as f32,
        ..fitted_spread(
            dense_xic,
            median_profile,
            library_intensities,
            fragment_types,
        )
    }
}

/// The fitted spread features, the other fields 0; all 0 if the XIC rows do not match the fragments.
fn fitted_spread(
    dense_xic: &Array2<f32>,
    median_profile: &[f32],
    library_intensities: &[f32],
    fragment_types: &[u8],
) -> LibraryAgreementFeatures {
    if dense_xic.nrows() != library_intensities.len() {
        return LibraryAgreementFeatures::default();
    }
    let fitted = fit_to_profile(dense_xic, median_profile);
    let y_mask: Vec<bool> = fragment_types
        .iter()
        .map(|&t| t == FragmentType::Y)
        .collect();
    let (mean, std, max) = intensity_abs_diff_stats(&fitted, library_intensities, None);
    let (_, y_std, _) = intensity_abs_diff_stats(&fitted, library_intensities, Some(&y_mask));
    LibraryAgreementFeatures {
        fit_intensity_abs_diff_mean: finite_or_zero(mean),
        fit_intensity_abs_diff_std: finite_or_zero(std),
        fit_intensity_abs_diff_max: finite_or_zero(max),
        fit_y_intensity_abs_diff_std: finite_or_zero(y_std),
        ..Default::default()
    }
}

fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

/// Scales `values` to unit sum; all zeros if the sum is not positive.
fn unit_sum(values: &[f32]) -> Vec<f32> {
    let total: f32 = values.iter().sum();
    if total <= 0.0 {
        return vec![0.0; values.len()];
    }
    values.iter().map(|&v| v / total).collect()
}

/// Fits every row of `xic` as `a_i * profile` with `a_i >= 0` (least squares) and returns per fragment the
/// co-eluting intensity `a_i * sum(profile)`; 0 for every fragment if the profile has no signal or does not match
/// the XIC's cycles.
pub fn fit_to_profile(xic: &Array2<f32>, profile: &[f32]) -> Vec<f32> {
    let n_fragments = xic.nrows();
    let norm_sq: f32 = profile.iter().map(|&m| m * m).sum();
    let profile_sum: f32 = profile.iter().sum();

    if xic.ncols() != profile.len() || norm_sq <= 0.0 {
        return vec![0.0; n_fragments];
    }

    xic.rows()
        .into_iter()
        .map(|row| {
            let dot: f32 = row.iter().zip(profile).map(|(&x, &m)| x * m).sum();
            (dot / norm_sq).max(0.0) * profile_sum
        })
        .collect()
}

/// Mean, sample std and max of `|observed_i - library_i|` after scaling both to unit sum over all fragments, over
/// the fragments selected by `mask` (all if `None`).
pub fn intensity_abs_diff_stats(
    observed: &[f32],
    library: &[f32],
    mask: Option<&[bool]>,
) -> (f32, f32, f32) {
    let observed = unit_sum(observed);
    let library = unit_sum(library);
    let diffs: Vec<f32> = observed
        .iter()
        .zip(&library)
        .enumerate()
        .filter(|(i, _)| mask.is_none_or(|m| m[*i]))
        .map(|(_, (&o, &l))| (o - l).abs())
        .collect();

    if diffs.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mean = diffs.iter().sum::<f32>() / diffs.len() as f32;
    let max = diffs.iter().cloned().fold(0.0, f32::max);
    (mean, calculate_std(&diffs), max)
}

fn log_ratio_std(library_intensities: &[f32], observation_intensities: &[f32]) -> f32 {
    let paired: Vec<usize> = (0..library_intensities.len())
        .filter(|&i| observation_intensities[i] > 0.0 && library_intensities[i] > 0.0)
        .collect();
    if paired.len() < 2 {
        return 0.0;
    }
    let observed_total: f32 = paired.iter().map(|&i| observation_intensities[i]).sum();
    let library_paired_total: f32 = paired.iter().map(|&i| library_intensities[i]).sum();
    let log_ratios: Vec<f32> = paired
        .iter()
        .map(|&i| {
            ((observation_intensities[i] / observed_total)
                / (library_intensities[i] / library_paired_total))
                .log2()
        })
        .collect();
    let mean = log_ratios.iter().sum::<f32>() / log_ratios.len() as f32;
    (log_ratios.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / log_ratios.len() as f32).sqrt()
}

fn y_intensity_correlation(
    fragment_types: &[u8],
    library_intensities: &[f32],
    observation_intensities: &[f32],
) -> f32 {
    let y_ions: Vec<usize> = (0..fragment_types.len())
        .filter(|&i| fragment_types[i] == FragmentType::Y)
        .collect();
    if y_ions.len() < MIN_Y_IONS_FOR_CORRELATION {
        return 0.0;
    }
    let observed_y: Vec<f32> = y_ions.iter().map(|&i| observation_intensities[i]).collect();
    let library_y: Vec<f32> = y_ions.iter().map(|&i| library_intensities[i]).collect();
    calculate_correlation_safe(&observed_y, &library_y)
}

/// Interfered fragments are left out, so the score reads whether the co-eluting fragments have the expected
/// intensities (in 200 ng ARATH entrapment XICs a decoy-trained model gained 6 % real targets at 1 % decoy FDR with it).
fn lib_cosine_over_50(
    dense_xic: &Array2<f32>,
    median_profile: &[f32],
    correlations: &[f32],
    library_intensities: &[f32],
) -> f32 {
    let (n_fragments, n_cycles) = dense_xic.dim();
    if n_cycles == 0
        || median_profile.len() != n_cycles
        || correlations.len() != n_fragments
        || library_intensities.len() != n_fragments
    {
        return 0.0;
    }
    let apex = median_profile
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let start = apex.saturating_sub(LIB_COSINE_APEX_HALF_WIDTH);
    let stop = (apex + LIB_COSINE_APEX_HALF_WIDTH + 1).min(n_cycles);

    let coeluting: Vec<usize> = (0..n_fragments)
        .filter(|&i| correlations[i] > COELUTION_MIN_CORRELATION)
        .collect();
    if coeluting.len() < LIB_COSINE_MIN_FRAGMENTS {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut observed_norm = 0.0f32;
    let mut library_norm = 0.0f32;
    for &i in &coeluting {
        let apex_intensity: f32 = (start..stop).map(|c| dense_xic[[i, c]]).sum();
        dot += apex_intensity * library_intensities[i];
        observed_norm += apex_intensity * apex_intensity;
        library_norm += library_intensities[i] * library_intensities[i];
    }
    let norm = (observed_norm * library_norm).sqrt();
    if norm > 0.0 {
        dot / norm
    } else {
        0.0
    }
}

fn entropy(p: &[f64]) -> f64 {
    -p.iter()
        .filter(|&&x| x > 0.0)
        .map(|&x| x * x.ln())
        .sum::<f64>()
}

/// Spectral entropy similarity of the library and observed intensities; 0 without observed signal.
/// A fragment without signal counts as a missing peak.
fn entropy_similarity(library: &[f64], observed: &[f64]) -> f64 {
    if observed.iter().sum::<f64>() <= 0.0 {
        return 0.0;
    }
    let p = shares(library);
    let q = shares(observed);
    let mixed: Vec<f64> = p.iter().zip(&q).map(|(a, b)| (a + b) / 2.0).collect();
    1.0 - (2.0 * entropy(&mixed) - entropy(&p) - entropy(&q)) / 4f64.ln()
}
