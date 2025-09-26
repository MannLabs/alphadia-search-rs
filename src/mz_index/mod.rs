use numpy::ndarray::Array1;
use once_cell::sync::Lazy;

pub const RESOLUTION_PPM: f32 = 1.0;
pub const MZ_START: f32 = 150.0;
pub const MZ_END: f32 = 2000.0;

pub fn ppm_index(resolution_ppm: f32, mz_start: f32, mz_end: f32) -> Array1<f32> {
    let mz_start_safe = mz_start.max(50.0);

    // Estimate final size based on geometric series: n ≈ ln(mz_end/mz_start) / ln(1 + resolution_ppm/1e6)
    let growth_factor = 1.0 + (resolution_ppm / 1e6);
    let estimated_size = ((mz_end / mz_start_safe).ln() / growth_factor.ln()) as usize + 1;

    let mut index: Vec<f32> = Vec::with_capacity(estimated_size);
    index.push(mz_start_safe);
    let mut current_mz = mz_start_safe;

    while current_mz < mz_end {
        current_mz += current_mz * (resolution_ppm / 1e6);
        index.push(current_mz);
    }

    Array1::from_vec(index)
}

static GLOBAL_MZ_INDEX: Lazy<MZIndex> = Lazy::new(MZIndex::new);

#[derive(Clone)]
pub struct MZIndex {
    pub mz: Array1<f32>,
}

impl Default for MZIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl MZIndex {
    pub fn new() -> Self {
        Self {
            mz: ppm_index(RESOLUTION_PPM, MZ_START, MZ_END),
        }
    }

    pub fn global() -> &'static MZIndex {
        &GLOBAL_MZ_INDEX
    }

    pub fn len(&self) -> usize {
        self.mz.len()
    }

    pub fn find_closest_index(&self, mz: f32) -> usize {
        let mut left = 0;
        let mut right = self.mz.len();

        while left < right {
            let mid = left + (right - left) / 2;

            if self.mz[mid] == mz {
                return mid;
            }

            if self.mz[mid] < mz {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        // After the loop, left is the insertion point
        // We need to check which of the adjacent indices is closer
        if left == 0 {
            return 0;
        }
        if left == self.mz.len() {
            return self.mz.len() - 1;
        }

        let left_diff = (self.mz[left] - mz).abs();
        let right_diff = (self.mz[left - 1] - mz).abs();

        if left_diff < right_diff {
            left
        } else {
            left - 1
        }
    }

    /// Returns an iterator over the indices of m/z values in the range [lower_mz, upper_mz]
    ///
    /// This finds the first index with m/z >= lower_mz and iterates until the last index with m/z <= upper_mz
    pub fn mz_range_indices(
        &self,
        lower_mz: f32,
        upper_mz: f32,
    ) -> impl Iterator<Item = usize> + '_ {
        // Find the first index where mz >= lower_mz
        let mut start_idx = 0;
        let mut right = self.mz.len();

        while start_idx < right {
            let mid = start_idx + (right - start_idx) / 2;

            if self.mz[mid] < lower_mz {
                start_idx = mid + 1;
            } else {
                right = mid;
            }
        }

        // Find the end index by counting up from start_idx
        let end_idx = if start_idx < self.mz.len() {
            let mut idx = start_idx;
            while idx < self.mz.len() && self.mz[idx] <= upper_mz {
                idx += 1;
            }
            idx
        } else {
            start_idx
        };

        // Return an iterator over the range of indices
        start_idx..end_idx
    }
}

#[cfg(test)]
mod tests;
