//! Construction of the consensus elution profile the integration strategies share.
//!
//! The first-pass profile is the median of every apex-normalised fragment trace, which is
//! also what the correlation feature is measured against. That median is robust to a
//! minority of interfered fragments but not to a systematic one: a fragment whose m/z is
//! shared with an abundant co-eluting species contributes its interference to the very
//! profile that then decides where the peak starts and ends, how it is shaped, and what
//! every fragment's amplitude is.
//!
//! A second pass fixes that cheaply. Each fragment's correlation to the first-pass profile
//! has already been computed, so the profile can simply be rebuilt from the fragments that
//! agree with it. Fragments below the correlation cut are still quantified — they are only
//! excluded from *defining* the elution profile.

use numpy::ndarray::Array2;

use crate::peak_group_scoring::utils::median_axis_0;

/// A profile built from one or two traces is not a consensus. Below this many agreeing
/// fragments the first-pass profile is kept.
pub const MIN_TEMPLATE_FRAGMENTS: usize = 3;

/// Rebuild the consensus profile from the fragments that correlate with the first pass.
///
/// `normalized_xic` is the apex-normalised XIC, `correlations` each fragment's correlation
/// to the first-pass profile. Returns `None` when the refinement is disabled or too few
/// fragments qualify, in which case the caller keeps the first-pass profile.
pub fn refine(
    normalized_xic: &Array2<f32>,
    correlations: &[f32],
    min_correlation: f32,
) -> Option<Vec<f32>> {
    if min_correlation <= 0.0 || correlations.len() != normalized_xic.nrows() {
        return None;
    }

    let selected: Vec<usize> = correlations
        .iter()
        .enumerate()
        .filter(|(idx, &correlation)| {
            correlation >= min_correlation
                && normalized_xic.row(*idx).iter().any(|&value| value != 0.0)
        })
        .map(|(idx, _)| idx)
        .collect();

    if selected.len() < MIN_TEMPLATE_FRAGMENTS {
        return None;
    }

    let mut agreeing = Array2::zeros((selected.len(), normalized_xic.ncols()));
    for (row_idx, &fragment_idx) in selected.iter().enumerate() {
        agreeing
            .row_mut(row_idx)
            .assign(&normalized_xic.row(fragment_idx));
    }

    Some(median_axis_0(&agreeing))
}
