use std::iter::zip;
use numpy::ndarray::s;
use numpy::ndarray::{ArrayBase, ViewRepr, Dim};

use crate::xic_index::{XICSlice, MZIndex};

#[derive(Debug)]
pub struct QuadrupoleObservation {
    pub isolation_window: [f32; 2],
    pub xic_slices: Vec<XICSlice>,
}

impl QuadrupoleObservation {
    pub fn new(mz_index: &MZIndex) -> Self {
        Self {
            isolation_window: [0.0, 0.0],
            xic_slices: vec![XICSlice::empty(); mz_index.len()],
        }
    }

    pub fn add_peak(&mut self, mz: f32, intensity: f32, cycle_idx: u16, mz_index: &MZIndex) {
        let closest_idx = mz_index.find_closest_index(mz);
        self.xic_slices[closest_idx].cycle_index.push(cycle_idx);
        self.xic_slices[closest_idx].intensity.push(intensity);
    }

    pub fn from_alpha_raw(alpha_raw_view: &crate::AlphaRawView, delta_scan_idx: i64, mz_index: &MZIndex) -> Self {
        let mut quad_obs = Self::new(mz_index);
        let mut num_valid_scans = 0;
        let mut num_peaks = 0;
        
        // Find the first valid scan to get isolation window
        for i in 0..alpha_raw_view.spectrum_delta_scan_idx.len() {
            if alpha_raw_view.spectrum_delta_scan_idx[i] == delta_scan_idx {
                if num_valid_scans == 0 {
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
                    num_peaks += 1;
                }

                num_valid_scans += 1;
            }
        }

        println!("Quadrupole observation idx: {}, Cycles: {}, Peaks: {}", delta_scan_idx, num_valid_scans, num_peaks);

        quad_obs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xic_index::MZIndex;

    #[test]
    fn test_quadrupole_observation() {
        // Create a new MZIndex
        let mz_index = MZIndex::new();
        
        // Create a new QuadrupoleObservation
        let mut quad_obs = QuadrupoleObservation::new(&mz_index);
        
        // Test initial state
        assert_eq!(quad_obs.isolation_window, [0.0, 0.0]);
        assert_eq!(quad_obs.xic_slices.len(), mz_index.len());
        
        // Add some test peaks
        let test_mz = 500.0; // Example m/z value
        let test_intensity = 1000.0;
        let test_cycle_idx = 1;
        
        quad_obs.add_peak(test_mz, test_intensity, test_cycle_idx, &mz_index);
        
        // Find the closest index for our test m/z
        let closest_idx = mz_index.find_closest_index(test_mz);
        
        // Verify the peak was added to the correct XIC slice
        assert_eq!(quad_obs.xic_slices[closest_idx].cycle_index.len(), 1);
        assert_eq!(quad_obs.xic_slices[closest_idx].intensity.len(), 1);
        assert_eq!(quad_obs.xic_slices[closest_idx].cycle_index[0], test_cycle_idx);
        assert_eq!(quad_obs.xic_slices[closest_idx].intensity[0], test_intensity);
        
        // Add another peak to the same m/z bin
        quad_obs.add_peak(test_mz, test_intensity * 2.0, test_cycle_idx + 1, &mz_index);
        
        // Verify both peaks are in the same XIC slice
        assert_eq!(quad_obs.xic_slices[closest_idx].cycle_index.len(), 2);
        assert_eq!(quad_obs.xic_slices[closest_idx].intensity.len(), 2);
        assert_eq!(quad_obs.xic_slices[closest_idx].cycle_index[1], test_cycle_idx + 1);
        assert_eq!(quad_obs.xic_slices[closest_idx].intensity[1], test_intensity * 2.0);
    }
} 