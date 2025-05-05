use numpy::ndarray::ArrayViewMut1;

use crate::xic_slice::XICSlice;
use crate::mz_index::MZIndex;

#[derive(Debug, Clone)]
pub struct QuadrupoleObservation {
    pub isolation_window: [f32; 2],
    pub num_cycles: usize,
    pub xic_slices: Vec<XICSlice>,
}

impl QuadrupoleObservation {
    pub fn new(mz_index: &MZIndex) -> Self {
        Self {
            isolation_window: [0.0, 0.0],
            num_cycles: 0,
            xic_slices: vec![XICSlice::empty(); mz_index.len()],
        }
    }

    pub fn add_peak(&mut self, mz: f32, intensity: f32, cycle_idx: u16, mz_index: &MZIndex) {
        let closest_idx = mz_index.find_closest_index(mz);
        self.xic_slices[closest_idx].cycle_index.push(cycle_idx);
        self.xic_slices[closest_idx].intensity.push(intensity);
    }

    pub fn fill_xic_slice(&self, mz_index: &MZIndex, dense_xic: &mut ArrayViewMut1<f32>, mass_tolerance: f32, mz: f32) {

        let delta_mz = mz * mass_tolerance * 1e-6;
        let lower_mz = mz - delta_mz;
        let upper_mz = mz + delta_mz;

        for mz_idx in mz_index.mz_range_indices(lower_mz, upper_mz) {
            let xic_slice = &self.xic_slices[mz_idx];
            for (cycle_idx, intensity) in xic_slice.cycle_index.iter().zip(xic_slice.intensity.iter()) {
                dense_xic[*cycle_idx as usize] += intensity;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mz_index::MZIndex;

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