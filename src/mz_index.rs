use numpy::ndarray::Array1;

pub const RESOLUTION_PPM: f32 = 1.0;
pub const MZ_START: f32 = 150.0;
pub const MZ_END: f32 = 2000.0;

pub fn ppm_index(resolution_ppm: f32, mz_start: f32, mz_end: f32) -> Array1<f32> {
    let mz_start_safe = mz_start.max(50.0);
    
    let mut index: Vec<f32> = Vec::from([mz_start_safe]);
    let mut current_mz = mz_start_safe;

    while current_mz < mz_end {
        current_mz += current_mz * (resolution_ppm / 1e6);
        index.push(current_mz);
    }

    Array1::from_vec(index)
}

pub struct MZIndex {
    pub mz: Array1<f32>,
}

impl MZIndex {
    pub fn new() -> Self {
        Self {
            mz: ppm_index(RESOLUTION_PPM, MZ_START, MZ_END),
        }
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
    pub fn mz_range_indices(&self, lower_mz: f32, upper_mz: f32) -> impl Iterator<Item = usize> + '_ {
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
mod tests {
    use super::*;

    #[test]
    fn test_ppm_index_len() {
        let result = ppm_index(1.0, 100.0, 2000.0);
        assert_eq!(result.len(), 2995974);
    }

    mod find_closest_index_tests {
        use super::*;

        #[test]
        fn exact_match_and_edges() {
            let mz_index = MZIndex::new();
            
            // Exact match
            let mz = mz_index.mz[100];
            assert_eq!(mz_index.find_closest_index(mz), 100);
            
            // First element
            assert_eq!(mz_index.find_closest_index(mz_index.mz[0]), 0);
            
            // Last element
            assert_eq!(mz_index.find_closest_index(mz_index.mz[mz_index.len() - 1]), 
                      mz_index.len() - 1);
        }

        #[test]
        fn out_of_range_values() {
            let mz_index = MZIndex::new();
            
            // Below range
            assert_eq!(mz_index.find_closest_index(0.0), 0);
            
            // Above range
            assert_eq!(mz_index.find_closest_index(3000.0), mz_index.len() - 1);
        }

        #[test]
        fn between_values() {
            let mz_index = MZIndex::new();
            let mz_between = (mz_index.mz[100] + mz_index.mz[101]) / 2.0;
            let closest = mz_index.find_closest_index(mz_between);
            assert!(closest == 100 || closest == 101);
        }
    }

    mod mz_range_indices_tests {
        use super::*;

        #[test]
        fn standard_ranges() {
            let mz_index = MZIndex::new();
            
            // Standard range
            let start = 100;
            let end = 105;
            let indices: Vec<usize> = mz_index.mz_range_indices(
                mz_index.mz[start], mz_index.mz[end]
            ).collect();
            // Should include the end value now
            assert_eq!(indices, (start..=end).collect::<Vec<_>>());
            
            // Single element
            let idx = 100;
            let indices: Vec<usize> = mz_index.mz_range_indices(
                mz_index.mz[idx], mz_index.mz[idx]
            ).collect();
            assert_eq!(indices, vec![idx]);
        }

        #[test]
        fn edge_cases() {
            let mz_index = MZIndex::new();
            
            // Below range
            let below_min = mz_index.mz[0] - 10.0;
            let indices: Vec<usize> = mz_index.mz_range_indices(below_min, below_min).collect();
            assert!(indices.is_empty());
            
            // Above range
            let max_mz = mz_index.mz[mz_index.len() - 1] + 10.0;
            let indices: Vec<usize> = mz_index.mz_range_indices(max_mz, max_mz + 10.0).collect();
            assert!(indices.is_empty());
            
            // Full range
            let below_min = mz_index.mz[0] - 10.0;
            let above_max = mz_index.mz[mz_index.len() - 1] + 10.0;
            let indices: Vec<usize> = mz_index.mz_range_indices(below_min, above_max).collect();
            assert_eq!(indices, (0..mz_index.len()).collect::<Vec<_>>());
        }

        #[test]
        fn partial_ranges() {
            let mz_index = MZIndex::new();
            
            // Lower out of bounds, upper in bounds
            let below_min = mz_index.mz[0] - 10.0;
            let target_idx = 50;
            let indices: Vec<usize> = mz_index.mz_range_indices(
                below_min, mz_index.mz[target_idx]
            ).collect();
            assert_eq!(indices, (0..=target_idx).collect::<Vec<_>>());
            
            // Lower in bounds, upper out of bounds
            let start_idx = mz_index.len() - 50;
            let above_max = mz_index.mz[mz_index.len() - 1] + 10.0;
            let indices: Vec<usize> = mz_index.mz_range_indices(
                mz_index.mz[start_idx], above_max
            ).collect();
            assert_eq!(indices, (start_idx..mz_index.len()).collect::<Vec<_>>());
        }
    }
} 