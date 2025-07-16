use rayon::prelude::*;
use std::iter::zip;
use numpy::ndarray::s;

use crate::dia_data::AlphaRawView;
use crate::mz_index::MZIndex;
use crate::quadrupole_observation::QuadrupoleObservation;
use crate::rt_index::RTIndex;
use crate::dia_data::DIAData;

pub struct DIADataBuilder;

impl DIADataBuilder {
    pub fn from_alpha_raw(alpha_raw_view: &AlphaRawView) -> DIAData {
        let mz_index = MZIndex::new();
        let num_quadrupole_observations = alpha_raw_view.spectrum_delta_scan_idx.iter().max().unwrap() + 1;

        let rt_index = RTIndex::from_alpha_raw(&alpha_raw_view);

        // Parallel iteration over quadrupole observations
        let quadrupole_observations: Vec<QuadrupoleObservation> = (0..num_quadrupole_observations)
            .into_par_iter()
            .map(|i| Self::build_quadrupole_observation(&alpha_raw_view, i, &mz_index))
            .collect();
        
        DIAData {
            mz_index,
            rt_index,
            quadrupole_observations,
        }
    }

    fn build_quadrupole_observation(alpha_raw_view: &AlphaRawView, delta_scan_idx: i64, mz_index: &MZIndex) -> QuadrupoleObservation {
        let mut quad_obs = QuadrupoleObservation::new(mz_index);
        let mut num_cycles = 0;
        let mut _num_peaks = 0;
        
        // Find the first valid scan to get isolation window
        for i in 0..alpha_raw_view.spectrum_delta_scan_idx.len() {
            if alpha_raw_view.spectrum_delta_scan_idx[i] == delta_scan_idx {
                if num_cycles == 0 {
                    // Set isolation window from the first valid scan
                    quad_obs.isolation_window = [
                        alpha_raw_view.isolation_lower_mz[i],
                        alpha_raw_view.isolation_upper_mz[i]
                    ];
                }

                // Get the peak indices for this scan
                let peak_start_idx = alpha_raw_view.spectrum_peak_start_idx[i] as usize;
                let peak_stop_idx = alpha_raw_view.spectrum_peak_stop_idx[i] as usize;
                let cycle_idx = alpha_raw_view.spectrum_cycle_idx[i] as u16;

                // Get the mz and intensity slices for this scan
                let mz_slice = alpha_raw_view.peak_mz.slice(s![peak_start_idx..peak_stop_idx]);
                let intensity_slice = alpha_raw_view.peak_intensity.slice(s![peak_start_idx..peak_stop_idx]);

                // Add each peak to the appropriate XIC slice
                for (mz_val, intensity_val) in zip(mz_slice.iter(), intensity_slice.iter()) {
                    quad_obs.add_peak(*mz_val, *intensity_val, cycle_idx, mz_index);
                    _num_peaks += 1;
                }

                num_cycles += 1;
            }
        }

        quad_obs.num_cycles = num_cycles;

        quad_obs
    }
}

#[cfg(test)]
mod tests; 