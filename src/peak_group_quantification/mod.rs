//! Peak group quantification operates only on candidates scored at 1% FDR.
//!
//! Avoids storing memory-intensive fragment-level properties for all possible candidates
//! (e.g., 3M precursors × 3 candidates × target+decoy = 18M candidates) by quantifying
//! only the pre-filtered high-confidence subset.

use pyo3::prelude::*;
use rayon::prelude::*;
use std::time::Instant;

use crate::candidate::{Candidate, CandidateCollection};
use crate::dense_xic_observation::DenseXICMZObservation;
use crate::dia_data::DIAData;
use crate::peak_group_scoring::utils::{
    calculate_correlation_safe, filter_non_zero, median_axis_0, normalize_profiles,
};
use crate::precursor_quantified::PrecursorQuantified;
use crate::traits::DIADataTrait;
use crate::utils::calculate_fragment_mz_and_errors;
use crate::{SpecLibFlat, SpecLibFlatQuantified};
use numpy::ndarray::Axis;

pub mod parameters;
pub mod tests;
pub use parameters::QuantificationParameters;

#[pyclass]
pub struct PeakGroupQuantification {
    params: QuantificationParameters,
}

#[pymethods]
impl PeakGroupQuantification {
    #[new]
    pub fn new(params: QuantificationParameters) -> Self {
        Self { params }
    }

    pub fn quantify(
        &self,
        dia_data: &DIAData,
        lib: &SpecLibFlat,
        candidates: &CandidateCollection,
    ) -> SpecLibFlatQuantified {
        self.quantify_generic(dia_data, lib, candidates)
    }
}

impl PeakGroupQuantification {
    fn quantify_generic<T: DIADataTrait + Sync>(
        &self,
        dia_data: &T,
        lib: &SpecLibFlat,
        candidates: &CandidateCollection,
    ) -> SpecLibFlatQuantified {
        let start_time = Instant::now();

        let quantified_precursors: Vec<PrecursorQuantified> = candidates
            .par_iter()
            .filter_map(|candidate| {
                match lib.get_precursor_by_idx_filtered(
                    candidate.precursor_idx,
                    true,
                    true, // Filter Y1 ions by default
                    self.params.top_k_fragments,
                ) {
                    Some(precursor) => self.quantify_precursor(dia_data, &precursor, candidate),
                    None => None,
                }
            })
            .collect();

        let duration = start_time.elapsed();
        println!(
            "Peak group quantification completed in {:.2}s for {} precursors",
            duration.as_secs_f64(),
            quantified_precursors.len()
        );

        SpecLibFlatQuantified::from_precursor_quantified_vec(quantified_precursors)
    }

    fn quantify_precursor<T: DIADataTrait>(
        &self,
        dia_data: &T,
        precursor: &crate::precursor::Precursor,
        candidate: &Candidate,
    ) -> Option<PrecursorQuantified> {
        let cycle_start = candidate.cycle_start;
        let cycle_stop = candidate.cycle_stop;
        let cycle_center = candidate.cycle_center;

        let num_cycles = cycle_stop - cycle_start;

        if num_cycles == 0 || cycle_center < cycle_start || cycle_center >= cycle_stop {
            return None;
        }

        let dense_xic_mz_obs = DenseXICMZObservation::new(
            dia_data,
            precursor.mz,
            cycle_start,
            cycle_stop,
            self.params.tolerance_ppm,
            &precursor.fragment_mz,
        );

        let num_fragments = precursor.fragment_mz.len();
        let mut fragment_correlation_observed = vec![0.0f32; num_fragments];

        let _center_cycle_idx = cycle_center - cycle_start;

        // Calculate observed m/z values and mass errors for all fragments
        let (fragment_mz_observed, fragment_mass_error_observed) = calculate_fragment_mz_and_errors(
            &dense_xic_mz_obs.dense_mz,
            &dense_xic_mz_obs.dense_xic,
            &precursor.fragment_mz,
        );

        let normalized_xic = normalize_profiles(&dense_xic_mz_obs.dense_xic, 1);

        // Filter to only non-zero profiles for median calculation (same as in scoring)
        let filtered_xic = filter_non_zero(&normalized_xic);
        let median_profile = median_axis_0(&filtered_xic);

        for fragment_idx in 0..num_fragments {
            let fragment_profile = normalized_xic.row(fragment_idx);
            let correlation = calculate_correlation_safe(
                fragment_profile.as_slice().unwrap_or(&[]),
                &median_profile,
            );
            fragment_correlation_observed[fragment_idx] = correlation;
        }

        // Calculate observed intensities from the dense XIC observation (sum across cycles)
        let observation_intensities = dense_xic_mz_obs.dense_xic.sum_axis(Axis(1));

        let rt_observed = dia_data.rt_index().rt[cycle_center];

        let precursor_quantified = PrecursorQuantified {
            precursor_idx: precursor.precursor_idx,
            mz_library: precursor.mz_library,
            mz: precursor.mz,
            rt_library: precursor.rt_library,
            rt: precursor.rt,
            naa: precursor.naa,
            rank: candidate.rank,
            rt_observed,
            // Clone is necessary because we only have a borrowed reference (&Precursor) to the precursor,
            // but PrecursorQuantified needs to own its Vec<T> data. Since Vec<T> contains heap-allocated
            // data, we must clone to create new owned copies rather than trying to move from a borrowed value.
            fragment_mz_library: precursor.fragment_mz_library.clone(),
            fragment_mz: precursor.fragment_mz.clone(),
            fragment_intensity: observation_intensities.to_vec(),
            fragment_cardinality: precursor.fragment_cardinality.clone(),
            fragment_charge: precursor.fragment_charge.clone(),
            fragment_loss_type: precursor.fragment_loss_type.clone(),
            fragment_number: precursor.fragment_number.clone(),
            fragment_position: precursor.fragment_position.clone(),
            fragment_type: precursor.fragment_type.clone(),
            fragment_mz_observed,
            fragment_correlation_observed,
            fragment_mass_error_observed,
        };

        precursor_quantified.filter_fragments_by_intensity(0.0)
    }
}
