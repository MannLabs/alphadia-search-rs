use crate::mz_index::MZIndex;
use numpy::ndarray::ArrayViewMut1;

#[cfg(test)]
mod tests;

/// QuadrupoleObservation structure that achieves >99.9% memory overhead reduction
///
/// Instead of millions of XICSlice objects with individual Vec allocations,
/// this uses consolidated arrays with index-based slicing.
#[derive(Debug, Clone)]
pub struct QuadrupoleObservation {
    pub isolation_window: [f32; 2],
    pub num_cycles: usize,

    /// Start indices for each mz_index slice. Length = mz_index.len() + 1
    /// The stop index for slice[i] is slice_starts[i+1]
    pub slice_starts: Vec<u32>,

    /// All cycle indices concatenated from all slices
    pub cycle_indices: Vec<u16>,

    /// All intensities concatenated from all slices
    pub intensities: Vec<f32>,
}

impl QuadrupoleObservation {
    /// Create a new empty observation with exact pre-allocation
    pub fn new_with_capacity(
        isolation_window: [f32; 2],
        num_cycles: usize,
        mz_index_len: usize,
        total_peaks: usize,
    ) -> Self {
        let mut slice_starts = Vec::with_capacity(mz_index_len + 1);
        slice_starts.push(0); // First slice starts at 0

        Self {
            isolation_window,
            num_cycles,
            slice_starts,
            cycle_indices: Vec::with_capacity(total_peaks),
            intensities: Vec::with_capacity(total_peaks),
        }
    }

    /// Get the cycle indices and intensities for a specific mz_index
    pub fn get_slice_data(&self, mz_idx: usize) -> (&[u16], &[f32]) {
        let start = self.slice_starts[mz_idx] as usize;
        let stop = self.slice_starts[mz_idx + 1] as usize;

        (
            &self.cycle_indices[start..stop],
            &self.intensities[start..stop],
        )
    }

    /// Add peak data for a specific mz_index (used during building)
    pub fn add_peak_data(&mut self, cycle_idx: u16, intensity: f32) {
        self.cycle_indices.push(cycle_idx);
        self.intensities.push(intensity);
    }

    /// Finalize a slice by recording its end position
    pub fn finalize_slice(&mut self) {
        self.slice_starts.push(self.cycle_indices.len() as u32);
    }

    /// Optimized fill_xic_slice method using direct array access
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
            // Direct slice access using optimized indexing
            let start = self.slice_starts[mz_idx] as usize;
            let stop = self.slice_starts[mz_idx + 1] as usize;

            let cycle_indices = &self.cycle_indices[start..stop];
            let intensities = &self.intensities[start..stop];

            // Binary search for start position
            let start_pos = cycle_indices
                .binary_search(&(cycle_start_idx as u16))
                .unwrap_or_else(|idx| idx);

            // Process cycles within range
            for i in start_pos..cycle_indices.len() {
                let cycle_idx = cycle_indices[i] as usize;

                if cycle_idx >= cycle_stop_idx {
                    break;
                }

                dense_xic[cycle_idx - cycle_start_idx] += intensities[i];
            }
        }
    }

    /// Calculate memory footprint of this optimized observation
    pub fn memory_footprint_bytes(&self) -> usize {
        let mut total_size = 0;

        // Fixed size components
        total_size += std::mem::size_of::<[f32; 2]>(); // isolation_window
        total_size += std::mem::size_of::<usize>(); // num_cycles

        // Vec overheads - only 3 total!
        total_size += std::mem::size_of::<Vec<u32>>(); // slice_starts
        total_size += std::mem::size_of::<Vec<u16>>(); // cycle_indices
        total_size += std::mem::size_of::<Vec<f32>>(); // intensities

        // Actual data
        total_size += self.slice_starts.len() * std::mem::size_of::<u32>();
        total_size += self.cycle_indices.len() * std::mem::size_of::<u16>();
        total_size += self.intensities.len() * std::mem::size_of::<f32>();

        total_size
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

    fn fill_xic_and_mz_slice(
        &self,
        mz_index: &crate::mz_index::MZIndex,
        dense_xic: &mut numpy::ndarray::ArrayViewMut1<f32>,
        dense_mz: &mut numpy::ndarray::ArrayViewMut1<f32>,
        cycle_start_idx: usize,
        cycle_stop_idx: usize,
        mass_tolerance: f32,
        mz: f32,
    ) {
        let delta_mz = mz * mass_tolerance * 1e-6;
        let lower_mz = mz - delta_mz;
        let upper_mz = mz + delta_mz;

        for mz_idx in mz_index.mz_range_indices(lower_mz, upper_mz) {
            let actual_mz = mz_index.mz[mz_idx];

            // Direct slice access using optimized indexing
            let start = self.slice_starts[mz_idx] as usize;
            let stop = self.slice_starts[mz_idx + 1] as usize;

            let cycle_indices = &self.cycle_indices[start..stop];
            let intensities = &self.intensities[start..stop];

            // Binary search for start position
            let start_pos = cycle_indices
                .binary_search(&(cycle_start_idx as u16))
                .unwrap_or_else(|idx| idx);

            // Process cycles within range
            for i in start_pos..cycle_indices.len() {
                let cycle_idx = cycle_indices[i] as usize;

                if cycle_idx >= cycle_stop_idx {
                    break;
                }

                let relative_idx = cycle_idx - cycle_start_idx;
                let intensity = intensities[i];

                // Always accumulate intensity (even if zero)
                dense_xic[relative_idx] += intensity;

                // Update m/z weighted average only for non-zero intensities
                if intensity > 0.0 {
                    let prev_total_intensity = dense_xic[relative_idx] - intensity;

                    if prev_total_intensity == 0.0 {
                        // First non-zero intensity at this position
                        dense_mz[relative_idx] = actual_mz;
                    } else {
                        // Update running weighted average
                        dense_mz[relative_idx] = (dense_mz[relative_idx] * prev_total_intensity
                            + actual_mz * intensity)
                            / dense_xic[relative_idx];
                    }
                }
            }
        }
    }
}
