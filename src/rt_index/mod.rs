use numpy::ndarray::Array1;

use crate::dia_data::AlphaRawView;

const MIN_CYCLES: f32 = 10.0;

pub struct RTIndex {
    pub rt: Array1<f32>,
    delta_t: f32,
}

impl Default for RTIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl RTIndex {
    /// Calculates the mean delta_t (average cycle time) from retention times.
    pub fn calculate_mean_delta_t(rt: &Array1<f32>) -> f32 {
        if rt.len() > 1 {
            (rt[rt.len() - 1] - rt[0]) / (rt.len() - 1) as f32
        } else {
            1.0 // fallback if only one point
        }
    }

    /// Creates a new empty RTIndex.
    pub fn new() -> Self {
        Self {
            rt: Array1::from_vec(Vec::new()),
            delta_t: 1.0, // default fallback
        }
    }

    /// Returns the number of retention time points in the index
    pub fn len(&self) -> usize {
        self.rt.len()
    }

    /// Returns true if the index contains no retention time points
    pub fn is_empty(&self) -> bool {
        self.rt.is_empty()
    }

    /// Creates an RTIndex from an AlphaRawView.
    ///
    /// Extracts retention times for MS1 scans (where delta_scan_idx is 0).
    ///
    /// # Arguments
    ///
    /// * `alpha_raw_view` - Source view containing spectrum data
    pub fn from_alpha_raw(alpha_raw_view: &AlphaRawView) -> Self {
        // Estimate capacity using the last element of spectrum_cycle_idx
        // This represents the total number of cycles, which is the number of MS1 scans

        let estimated_capacity = alpha_raw_view.spectrum_cycle_idx
            [alpha_raw_view.spectrum_cycle_idx.len() - 1] as usize
            + 1;
        let mut rt = Vec::with_capacity(estimated_capacity);

        for i in 0..alpha_raw_view.spectrum_delta_scan_idx.len() {
            if alpha_raw_view.spectrum_delta_scan_idx[i] == 0 {
                rt.push(alpha_raw_view.spectrum_rt[i]);
            }
        }

        let rt_array = Array1::from_vec(rt);
        let delta_t = Self::calculate_mean_delta_t(&rt_array);

        Self {
            rt: rt_array,
            delta_t,
        }
    }

    /// Finds the index range within the retention time array that falls within a tolerance window.
    /// Ensures the range is at least MIN_CYCLES*delta_t + num_cycles_padding*delta_t wide.
    ///
    /// # Arguments
    ///
    /// * `precursor_rt` - Target retention time
    /// * `rt_tolerance` - Window size around the target (precursor_rt +- rt_tolerance)
    /// * `num_cycles_padding` - Additional padding size in number of cycles for the convolution kernel
    ///
    /// # Returns
    ///
    /// A tuple of (lower_idx, upper_idx) representing the range boundaries
    pub fn get_cycle_idx_limits(
        &self,
        precursor_rt: f32,
        rt_tolerance: f32,
        num_cycles_padding: usize,
    ) -> (usize, usize) {
        if self.rt.is_empty() {
            return (0, 0);
        }

        // Calculate minimum required tolerance: MIN_CYCLES*delta_t + num_cycles_padding*delta_t
        let min_tolerance = (MIN_CYCLES + num_cycles_padding as f32) * self.delta_t;
        let effective_tolerance = rt_tolerance.max(min_tolerance);

        let lower_rt = precursor_rt - effective_tolerance;
        let upper_rt = precursor_rt + effective_tolerance;

        // Check if completely below the range
        if upper_rt < self.rt[0] {
            return (0, 0);
        }

        // Check if completely above the range
        if lower_rt > self.rt[self.rt.len() - 1] {
            return (self.rt.len(), self.rt.len());
        }

        // Use slice directly for binary search to avoid allocation
        let rt_slice = self.rt.as_slice().unwrap();

        // Lower bound search - only if needed
        let lower_idx = if lower_rt <= self.rt[0] {
            0
        } else {
            rt_slice
                .binary_search_by(|&x| x.partial_cmp(&lower_rt).unwrap())
                .unwrap_or_else(|x| x)
        };

        // Upper bound search - only if needed
        let upper_idx = if upper_rt >= self.rt[self.rt.len() - 1] {
            self.rt.len()
        } else {
            rt_slice
                .binary_search_by(|&x| x.partial_cmp(&upper_rt).unwrap())
                .unwrap_or_else(|x| x)
        };

        (lower_idx, upper_idx)
    }
}

#[cfg(test)]
mod tests;
