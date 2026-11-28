//! What the feature blocks read of one candidate: its fragment XICs and their agreement with the median profile.

use numpy::ndarray::{Array2, Axis};

use crate::candidate::Candidate;
use crate::dense_xic_observation::{DenseXICMZObservation, DenseXICObservation};
use crate::peak_group_scoring::fragment_isotope::ISOTOPE_SPACING;
use crate::peak_group_scoring::utils::{
    correlation_axis_0, filter_non_zero, median_axis_0, normalize_profiles,
    COELUTION_MIN_CORRELATION,
};
use crate::precursor::Precursor;
use crate::traits::DIADataTrait;

/// The isotope XICs of a candidate's fragments: fragments x cycles each.
pub struct IsotopeXics {
    /// M+1 XICs of the isotope-eligible fragments, zero in the other rows
    pub mplus1: Array2<f32>,
    /// M-1 XICs of the isotope-eligible fragments that co-elute and have a monoisotopic signal, zero in the other
    /// rows. The isotope features read the M-1 XICs of these fragments only.
    pub mminus1: Array2<f32>,
    /// monoisotopic fragment XICs
    pub monoisotopic: Array2<f32>,
    /// monoisotopic intensity per fragment, summed over the candidate's cycles
    pub monoisotopic_intensities: Vec<f32>,
}

pub struct PeakGroupObservation {
    /// fragment XICs: fragments x cycles. With isotope XICs, the M+1 XIC of each isotope-eligible fragment is added
    /// to its monoisotopic XIC.
    pub dense_xic: Array2<f32>,
    /// observed m/z of the monoisotopic fragment XICs: fragments x cycles
    pub dense_mz: Array2<f32>,
    /// median of the normalized XICs
    pub median_profile: Vec<f32>,
    /// median of the normalized XICs with signal
    pub median_profile_filtered: Vec<f32>,
    /// number of fragment XICs with signal
    pub num_profiles_filtered: usize,
    /// correlation of each normalized XIC with `median_profile_filtered`
    pub correlations: Vec<f32>,
    /// observed intensity per fragment, summed over the candidate's cycles
    pub observation_intensities: Vec<f32>,
    /// observed intensity > 0 per fragment
    pub matched_mask: Vec<bool>,
    /// the quadrupole observations (isolation windows) the XICs were extracted from
    pub quadrupole_observation_indices: Vec<usize>,
    /// None if the observation has no isotope XICs
    pub isotope_xics: Option<IsotopeXics>,
}

impl PeakGroupObservation {
    /// The observation of a candidate.
    ///
    /// With `isotope_eligible` (one value per fragment), the observation also holds the M+1 and M-1 XICs of the
    /// fragments, and the M+1 XIC of each eligible fragment adds to its fragment XIC. The M+1 peak carries a large
    /// part of a fragment's signal, so weak fragments give more signal.
    pub fn new<T: DIADataTrait>(
        dia_data: &T,
        precursor: &Precursor,
        candidate: &Candidate,
        mass_tolerance: f32,
        isotope_eligible: Option<&[bool]>,
    ) -> Self {
        let dense_xic_mz_obs = DenseXICMZObservation::new(
            dia_data,
            precursor.mz,
            candidate.cycle_start,
            candidate.cycle_stop,
            mass_tolerance,
            &precursor.fragment_mz,
        );
        let monoisotopic = dense_xic_mz_obs.dense_xic;

        let extract_rows = |rows: &[usize], shift: f32| {
            extract_isotope_rows(dia_data, precursor, candidate, mass_tolerance, rows, shift)
        };
        let eligible_rows: Vec<usize> = isotope_eligible
            .map(|eligible| (0..eligible.len()).filter(|&i| eligible[i]).collect())
            .unwrap_or_default();

        let (dense_xic, mplus1) = match isotope_eligible {
            None => (monoisotopic.clone(), None),
            Some(_) => {
                let mplus1 = extract_rows(&eligible_rows, 1.0);
                (&monoisotopic + &mplus1, Some(mplus1))
            }
        };

        let normalized_xic = normalize_profiles(&dense_xic, 1);
        let filtered_xic = filter_non_zero(&normalized_xic);

        let median_profile = median_axis_0(&normalized_xic);
        let median_profile_filtered = median_axis_0(&filtered_xic);
        let correlations = correlation_axis_0(&median_profile_filtered, &normalized_xic);

        let observation_intensities = dense_xic.sum_axis(Axis(1)).to_vec();
        let matched_mask = observation_intensities.iter().map(|&x| x > 0.0).collect();

        let isotope_xics = mplus1.map(|mplus1| {
            let monoisotopic_intensities = monoisotopic.sum_axis(Axis(1)).to_vec();
            let coeluting_rows: Vec<usize> = eligible_rows
                .iter()
                .copied()
                .filter(|&i| {
                    correlations[i] > COELUTION_MIN_CORRELATION && monoisotopic_intensities[i] > 0.0
                })
                .collect();
            IsotopeXics {
                mplus1,
                mminus1: extract_rows(&coeluting_rows, -1.0),
                monoisotopic,
                monoisotopic_intensities,
            }
        });

        Self {
            dense_xic,
            dense_mz: dense_xic_mz_obs.dense_mz,
            median_profile,
            median_profile_filtered,
            num_profiles_filtered: filtered_xic.shape()[0],
            correlations,
            observation_intensities,
            matched_mask,
            quadrupole_observation_indices: dense_xic_mz_obs.contributing_obs_indices,
            isotope_xics,
        }
    }

    /// Monoisotopic intensity > 0 per fragment. Only these fragments have an observed m/z and a mass error.
    pub fn monoisotopic_matched_mask(&self) -> Vec<bool> {
        match &self.isotope_xics {
            None => self.matched_mask.clone(),
            Some(isotopes) => isotopes
                .monoisotopic_intensities
                .iter()
                .map(|&x| x > 0.0)
                .collect(),
        }
    }

    /// The monoisotopic fragment XICs, which `dense_mz` belongs to.
    pub fn monoisotopic_xic(&self) -> &Array2<f32> {
        self.isotope_xics
            .as_ref()
            .map_or(&self.dense_xic, |isotopes| &isotopes.monoisotopic)
    }
}

/// The XICs at `shift` isotope spacings from the fragments in `rows`: fragments x cycles, zero in the other rows.
fn extract_isotope_rows<T: DIADataTrait>(
    dia_data: &T,
    precursor: &Precursor,
    candidate: &Candidate,
    mass_tolerance: f32,
    rows: &[usize],
    shift: f32,
) -> Array2<f32> {
    let num_cycles = candidate.cycle_stop - candidate.cycle_start;
    let mut xic = Array2::zeros((precursor.fragment_mz.len(), num_cycles));
    if rows.is_empty() {
        return xic;
    }
    let mz: Vec<f32> = rows
        .iter()
        .map(|&i| {
            precursor.fragment_mz[i]
                + shift * ISOTOPE_SPACING / precursor.fragment_charge[i].max(1) as f32
        })
        .collect();
    let extracted = DenseXICObservation::new(
        dia_data,
        precursor.mz,
        candidate.cycle_start,
        candidate.cycle_stop,
        mass_tolerance,
        &mz,
    )
    .dense_xic;
    for (k, &i) in rows.iter().enumerate() {
        xic.row_mut(i).assign(&extracted.row(k));
    }
    xic
}
