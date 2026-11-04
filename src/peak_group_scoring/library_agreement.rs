//! Whether the observed intensities of a candidate's fragments match the library intensities.
//!
//! Interfered fragments distort the observed spectrum, so most of these features also compare the spectra over the
//! co-eluting fragments only, or with each fragment weighted by its correlation. In fragment XICs of ARATH entrapment
//! runs they separate real peptides from false targets beyond correlation and intensity summaries, also when trained
//! against decoys only.

use numpy::ndarray::Array2;

use crate::constants::FragmentType;
use crate::peak_group_scoring::utils::{
    calculate_correlation_safe, shares, COELUTION_MIN_CORRELATION,
};

/// Minimum number of y ions for the y-ion intensity correlation
pub const MIN_Y_IONS_FOR_CORRELATION: usize = 3;
/// Cycles on each side of the median-profile apex summed for the apex spectrum
pub const LIB_COSINE_APEX_HALF_WIDTH: usize = 1;
/// Minimum number of co-eluting fragments for the library cosine over the co-eluting fragments
pub const LIB_COSINE_MIN_FRAGMENTS: usize = 3;

/// The library-agreement features of one candidate.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
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
    }
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
