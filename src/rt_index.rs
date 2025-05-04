use numpy::ndarray::Array1;

use crate::AlphaRawView;

pub struct RTIndex {
    pub rt: Array1<f32>,
}

impl RTIndex {
    /// Creates a new empty RTIndex.
    pub fn new() -> Self {
        Self {
            rt: Array1::from_vec(Vec::new()),
        }
    }

    /// Creates an RTIndex from an AlphaRawView.
    ///
    /// Extracts retention times for MS1 scans (where delta_scan_idx is 0).
    ///
    /// # Arguments
    ///
    /// * `alpha_raw_view` - Source view containing spectrum data
    pub fn from_alpha_raw(alpha_raw_view: &AlphaRawView) -> Self {
        let mut rt = Vec::new();

        for i in 0..alpha_raw_view.spectrum_delta_scan_idx.len() {
            if alpha_raw_view.spectrum_delta_scan_idx[i] == 0 {
                rt.push(alpha_raw_view.spectrum_rt[i]);
            }
        }

        Self {
            rt: Array1::from_vec(rt),
        }
    }

    /// Finds the index range within the retention time array that falls within a tolerance window.
    ///
    /// # Arguments
    ///
    /// * `precursor_rt` - Target retention time
    /// * `rt_tolerance` - Window size around the target (precursor_rt +- rt_tolerance)
    ///
    /// # Returns
    ///
    /// A tuple of (lower_idx, upper_idx) representing the range boundaries
    pub fn get_cycle_idx_limits(&self, precursor_rt: f32, rt_tolerance: f32) -> (usize, usize) {
        if self.rt.is_empty() {
            return (0, 0);
        }
        
        let lower_rt = precursor_rt - rt_tolerance;
        let upper_rt = precursor_rt + rt_tolerance;
        
        // Check if completely below the range
        if upper_rt < self.rt[0] {
            return (0, 0);
        }
        
        // Check if completely above the range
        if lower_rt > self.rt[self.rt.len() - 1] {
            return (self.rt.len(), self.rt.len());
        }
        
        // Convert to Vec to use binary_search method
        let rt_vec: Vec<f32> = self.rt.to_vec();
        
        // Lower bound search - only if needed
        let lower_idx = if lower_rt <= self.rt[0] {
            0
        } else {
            rt_vec.binary_search_by(|&x| x.partial_cmp(&lower_rt).unwrap()).unwrap_or_else(|x| x)
        };
        
        // Upper bound search - only if needed
        let upper_idx = if upper_rt >= self.rt[self.rt.len() - 1] {
            self.rt.len()
        } else {
            rt_vec.binary_search_by(|&x| x.partial_cmp(&upper_rt).unwrap()).unwrap_or_else(|x| x)
        };
        
        (lower_idx, upper_idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Creates a test RTIndex with predefined values [10.0, 20.0, 30.0, 40.0, 50.0]
    fn create_test_index() -> RTIndex {
        let rt_values = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        RTIndex {
            rt: Array1::from_vec(rt_values),
        }
    }

    #[test]
    fn test_normal_case_within_range() {
        let rt_index = create_test_index();
        let (lower, upper) = rt_index.get_cycle_idx_limits(30.0, 5.0);
        assert_eq!(lower, 2);
        assert_eq!(upper, 3);
    }

    #[test]
    fn test_below_range() {
        let rt_index = create_test_index();
        let (lower, upper) = rt_index.get_cycle_idx_limits(5.0, 2.0);
        assert_eq!(lower, 0);
        assert_eq!(upper, 0);
    }

    #[test]
    fn test_above_range() {
        let rt_index = create_test_index();
        let (lower, upper) = rt_index.get_cycle_idx_limits(60.0, 2.0);
        assert_eq!(lower, 5);
        assert_eq!(upper, 5);
    }

    #[test]
    fn test_partially_below_range() {
        let rt_index = create_test_index();
        let (lower, upper) = rt_index.get_cycle_idx_limits(12.0, 5.0);
        assert_eq!(lower, 0);
        assert_eq!(upper, 1);
    }

    #[test]
    fn test_partially_above_range() {
        let rt_index = create_test_index();
        let (lower, upper) = rt_index.get_cycle_idx_limits(48.0, 5.0);
        assert_eq!(lower, 4);
        assert_eq!(upper, 5);
    }

    #[test]
    fn test_spanning_entire_range() {
        let rt_index = create_test_index();
        let (lower, upper) = rt_index.get_cycle_idx_limits(30.0, 30.0);
        assert_eq!(lower, 0);
        assert_eq!(upper, 5);
    }

    #[test]
    fn test_empty_index() {
        let empty_rt_index = RTIndex::new();
        let (lower, upper) = empty_rt_index.get_cycle_idx_limits(30.0, 5.0);
        assert_eq!(lower, 0);
        assert_eq!(upper, 0);
    }
} 