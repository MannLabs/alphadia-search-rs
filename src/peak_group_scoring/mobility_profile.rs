//! The RT x mobility peak of a candidate, for data with an ion mobility dimension (timsTOF).
//!
//! The XICs of the other blocks sum all scans of the mobility window, so a precursor eluting at the same time but at
//! another mobility adds to them. Here each fragment keeps its scans: a patch of cycles around the apex x scans of the
//! mobility window. The fragments of a real precursor form one 2D peak, so each fragment's patch matches the consensus
//! of the others, their mobilogram apexes agree, and most of the patch signal lies inside that peak. A fragment that
//! carries an interferer's signal peaks elsewhere in the patch.
//!
//! Data without a mobility dimension, and candidates without fragment signal in the patch, get all values 0.

use numpy::ndarray::{Array2, Axis};

use crate::candidate::Candidate;
use crate::peak_group_scoring::feature_block::feature_block;
use crate::peak_group_scoring::utils::calculate_correlation_safe;
use crate::precursor::Precursor;
use crate::traits::{DIADataTrait, MobilityFilter};

/// The patch spans the cycles within this many cycles of the candidate's apex.
pub const PATCH_CYCLE_RADIUS: usize = 2;
/// Raw scans are sparse (ion counting), so each fragment's patch is smoothed along the scans with a box of this
/// half width before it is compared.
pub const SCAN_SMOOTHING_RADIUS: usize = 3;
/// A fragment whose patch correlates with the consensus of the others above this value is coherent.
pub const COHERENT_CORRELATION: f32 = 0.5;

feature_block! {
    /// The RT x mobility peak features of one candidate.
    pub struct MobilityProfileFeatures {
        /// library-intensity-weighted mean correlation of each fragment's smoothed patch with the consensus of the
        /// other fragments
        pub mob2d_corr_mean: f32,
        /// library intensity share of the fragments whose patch correlates above COHERENT_CORRELATION
        pub mob2d_coherent_share: f32,
        /// library-intensity-weighted SD (1/K0) of the fragments' mobilogram apexes around the consensus apex
        pub mob2d_apex_sd: f32,
        /// share of all fragment signal in the patch that lies within the consensus peak (apex cycle +-1, consensus
        /// mobilogram above half its maximum)
        pub mob2d_peak_share: f32,
        /// full width at half maximum (1/K0) of the consensus mobilogram at the apex
        pub mob2d_fwhm: f32,
    }
}

/// The RT x mobility features of one candidate over the scans of `filter`.
pub fn calculate_mobility_profile_features<T: DIADataTrait>(
    dia_data: &T,
    precursor: &Precursor,
    candidate: &Candidate,
    filter: MobilityFilter,
    mass_tolerance: f32,
) -> MobilityProfileFeatures {
    let Some((scan_start, scan_stop, scan_mobility)) = dia_data.mobility_scans(filter) else {
        return MobilityProfileFeatures::default();
    };
    let n_scans = scan_stop - scan_start;
    let num_cycles = dia_data.rt_index().rt.len();
    let cycle_start = candidate.cycle_center.saturating_sub(PATCH_CYCLE_RADIUS);
    let cycle_stop = (candidate.cycle_center + PATCH_CYCLE_RADIUS + 1).min(num_cycles);
    let n_cycles = cycle_stop.saturating_sub(cycle_start);
    if n_scans < 2 || n_cycles == 0 {
        return MobilityProfileFeatures::default();
    }
    let apex_row = candidate.cycle_center - cycle_start;

    let observations = dia_data.select_observations(precursor.mz, filter);
    let patches: Vec<Array2<f32>> = precursor
        .fragment_mz
        .iter()
        .map(|&mz| {
            let mut patch = Array2::<f32>::zeros((n_cycles, n_scans));
            dia_data.fill_mobility_patch(
                &observations,
                cycle_start,
                scan_start,
                mass_tolerance,
                mz,
                &mut patch,
            );
            smooth_scans(&patch, SCAN_SMOOTHING_RADIUS)
        })
        .collect();
    features_from_patches(
        &patches,
        &precursor.fragment_intensity,
        &scan_mobility,
        apex_row,
    )
}

/// The features from the smoothed patches (cycles x scans) of each fragment, their library intensities, the 1/K0 of
/// each scan column and the row of the candidate's apex cycle.
pub fn features_from_patches(
    patches: &[Array2<f32>],
    library_intensity: &[f32],
    scan_mobility: &[f32],
    apex_row: usize,
) -> MobilityProfileFeatures {
    let with_signal: Vec<usize> = (0..patches.len())
        .filter(|&f| patches[f].sum() > 0.0)
        .collect();
    if with_signal.is_empty() {
        return MobilityProfileFeatures::default();
    }
    let shape = patches[with_signal[0]].dim();

    // each fragment's patch at unit sum, so that the consensus is not one intense fragment
    let normalized: Vec<Array2<f32>> = with_signal
        .iter()
        .map(|&f| &patches[f] / patches[f].sum())
        .collect();
    let mut consensus = Array2::<f32>::zeros(shape);
    for patch in &normalized {
        consensus += patch;
    }

    let weights: Vec<f32> = with_signal
        .iter()
        .map(|&f| library_intensity[f].max(0.0))
        .collect();
    let total_weight: f32 = weights.iter().sum();
    let weight_share = |w: f32| {
        if total_weight > 0.0 {
            w / total_weight
        } else {
            1.0 / weights.len() as f32
        }
    };

    let mut corr_mean = 0.0;
    let mut coherent_share = 0.0;
    for (patch, &w) in normalized.iter().zip(&weights) {
        // leave-one-out consensus; a lone fragment has none and counts as not coherent
        let others = &consensus - patch;
        let corr = if with_signal.len() > 1 {
            calculate_correlation_safe(
                patch.as_slice().expect("standard layout"),
                others.as_slice().expect("standard layout"),
            )
        } else {
            0.0
        };
        corr_mean += weight_share(w) * corr;
        if corr > COHERENT_CORRELATION {
            coherent_share += weight_share(w);
        }
    }

    // consensus mobilogram at the apex (+-1 cycle)
    let rows = apex_row.saturating_sub(1)..(apex_row + 2).min(shape.0);
    let mobilogram = consensus
        .slice(numpy::ndarray::s![rows.clone(), ..])
        .sum_axis(Axis(0));
    let (apex_scan, &apex_value) = mobilogram
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .expect("at least 2 scans");
    let half = 0.5 * apex_value;
    let in_peak: Vec<bool> = mobilogram.iter().map(|&v| v >= half && v > 0.0).collect();
    let peak_scans = in_peak.iter().filter(|&&p| p).count();
    let scan_width = (scan_mobility[0] - scan_mobility[scan_mobility.len() - 1]).abs()
        / (scan_mobility.len() - 1) as f32;

    let mut apex_var = 0.0;
    let mut inside = 0.0;
    let mut all = 0.0;
    for (patch_idx, (&f, &w)) in with_signal.iter().zip(&weights).enumerate() {
        let fragment_mobilogram = normalized[patch_idx]
            .slice(numpy::ndarray::s![rows.clone(), ..])
            .sum_axis(Axis(0));
        if let Some((scan, _)) = fragment_mobilogram
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
        {
            let delta = scan_mobility[scan] - scan_mobility[apex_scan];
            apex_var += weight_share(w) * delta * delta;
        }
        let raw = &patches[f];
        all += raw.sum();
        for row in rows.clone() {
            for (scan, &p) in in_peak.iter().enumerate() {
                if p {
                    inside += raw[[row, scan]];
                }
            }
        }
    }

    MobilityProfileFeatures {
        mob2d_corr_mean: corr_mean,
        mob2d_coherent_share: coherent_share,
        mob2d_apex_sd: apex_var.sqrt(),
        mob2d_peak_share: if all > 0.0 { inside / all } else { 0.0 },
        mob2d_fwhm: peak_scans as f32 * scan_width,
    }
}

/// Each row smoothed with a centred box of `radius` scans on either side (shrunk at the edges).
fn smooth_scans(patch: &Array2<f32>, radius: usize) -> Array2<f32> {
    let (n_rows, n_cols) = patch.dim();
    let mut out = Array2::<f32>::zeros((n_rows, n_cols));
    for row in 0..n_rows {
        let values = patch.row(row);
        let mut prefix = vec![0.0f32; n_cols + 1];
        for col in 0..n_cols {
            prefix[col + 1] = prefix[col] + values[col];
        }
        for col in 0..n_cols {
            let lo = col.saturating_sub(radius);
            let hi = (col + radius + 1).min(n_cols);
            out[[row, col]] = (prefix[hi] - prefix[lo]) / (hi - lo) as f32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::arr2;

    fn gaussian_patch(center: f32, scale: f32) -> Array2<f32> {
        Array2::from_shape_fn((3, 20), |(row, col)| {
            let rt = [0.5, 1.0, 0.5][row];
            rt * scale * (-(col as f32 - center).powi(2) / 8.0).exp()
        })
    }

    fn mobilities() -> Vec<f32> {
        (0..20).map(|i| 1.0 - 0.001 * i as f32).collect()
    }

    #[test]
    fn test_coeluting_fragments_at_one_mobility_are_coherent() {
        let patches = vec![
            gaussian_patch(10.0, 1.0),
            gaussian_patch(10.0, 3.0),
            gaussian_patch(10.0, 2.0),
        ];
        let f = features_from_patches(&patches, &[1.0, 1.0, 1.0], &mobilities(), 1);
        assert!(f.mob2d_corr_mean > 0.99, "{f:?}");
        assert_eq!(f.mob2d_coherent_share, 1.0);
        assert!(f.mob2d_apex_sd < 1e-6);
        assert!(f.mob2d_peak_share > 0.5);
        assert!(f.mob2d_fwhm > 0.0);
    }

    #[test]
    fn test_a_fragment_at_another_mobility_lowers_the_coherence() {
        let patches = vec![
            gaussian_patch(5.0, 1.0),
            gaussian_patch(5.0, 1.0),
            gaussian_patch(15.0, 1.0),
        ];
        let f = features_from_patches(&patches, &[1.0, 1.0, 1.0], &mobilities(), 1);
        assert!(f.mob2d_coherent_share < 0.7, "{f:?}");
        assert!(f.mob2d_apex_sd > 0.004, "{f:?}");
    }

    #[test]
    fn test_no_signal_is_missing() {
        let patches = vec![Array2::zeros((3, 20))];
        assert_eq!(
            features_from_patches(&patches, &[1.0], &mobilities(), 1),
            MobilityProfileFeatures::default()
        );
    }

    #[test]
    fn test_smoothing_keeps_the_sum_inside_and_spreads_a_spike() {
        let smoothed = smooth_scans(&arr2(&[[0.0, 0.0, 3.0, 0.0, 0.0]]), 1);
        assert_eq!(smoothed.row(0).to_vec(), vec![0.0, 1.0, 1.0, 1.0, 0.0]);
    }
}
