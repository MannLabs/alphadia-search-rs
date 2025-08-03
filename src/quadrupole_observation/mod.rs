use numpy::ndarray::ArrayViewMut1;

use crate::mz_index::MZIndex;
use crate::xic_slice::XICSlice;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct QuadrupoleObservation {
    pub isolation_window: [f32; 2],
    pub num_cycles: usize,
    pub xic_slices: Vec<XICSlice>,
}

#[allow(dead_code)]
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

    pub fn fill_xic_slice(
        &self,
        mz_index: &MZIndex,
        dense_xic: &mut ArrayViewMut1<f32>,
        cycle_start_idx: usize,
        cycle_stop_idx: usize,
        mass_tolerance: f32,
        mz: f32,
    ) {
        let delta_mz = mz * mass_tolerance * 1e-6;
        let lower_mz = mz - delta_mz;
        let upper_mz = mz + delta_mz;

        for mz_idx in mz_index.mz_range_indices(lower_mz, upper_mz) {
            let xic_slice = &self.xic_slices[mz_idx];

            // Find the position of cycle_start_idx or the insertion point
            let start_pos = match xic_slice
                .cycle_index
                .binary_search(&(cycle_start_idx as u16))
            {
                Ok(idx) => idx,  // Exact match
                Err(idx) => idx, // Insertion point
            };

            // Process only the cycles within the specified range
            for i in start_pos..xic_slice.cycle_index.len() {
                let cycle_idx = xic_slice.cycle_index[i] as usize;

                // Stop once we reach cycle_stop_idx
                if cycle_idx >= cycle_stop_idx {
                    break;
                }

                let intensity = xic_slice.intensity[i];
                dense_xic[cycle_idx - cycle_start_idx] += intensity;
            }
        }
    }
}

// Implement the QuadrupoleObservationTrait for QuadrupoleObservation
impl crate::traits::QuadrupoleObservationTrait for QuadrupoleObservation {
    fn fill_xic_slice(
        &self,
        mz_index: &crate::mz_index::MZIndex,
        dense_xic: &mut numpy::ndarray::ArrayViewMut1<f32>,
        cycle_start_idx: usize,
        cycle_stop_idx: usize,
        mass_tolerance: f32,
        mz: f32,
    ) {
        self.fill_xic_slice(
            mz_index,
            dense_xic,
            cycle_start_idx,
            cycle_stop_idx,
            mass_tolerance,
            mz,
        )
    }
}

#[cfg(test)]
mod tests;
