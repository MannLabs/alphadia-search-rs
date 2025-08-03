use crate::precursor::Precursor;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

#[pyclass]
pub struct SpecLibFlat {
    /// Precursor indices, MUST be sorted in ascending order for binary search to work correctly
    precursor_idx: Vec<usize>,
    /// Precursor m/z values, sorted according to precursor_idx order
    precursor_mz: Vec<f32>,
    /// Precursor retention times, sorted according to precursor_idx order
    precursor_rt: Vec<f32>,
    /// Start indices into fragment arrays for each precursor, sorted according to precursor_idx order
    precursor_start_idx: Vec<usize>,
    /// Stop indices into fragment arrays for each precursor, sorted according to precursor_idx order
    precursor_stop_idx: Vec<usize>,
    /// Fragment m/z values, expected to be sorted in ascending order within each precursor upon creation
    fragment_mz: Vec<f32>,
    /// Fragment intensity values in original library order (NOT sorted, maintains original order within each precursor)
    fragment_intensity: Vec<f32>,
}

#[pymethods]
impl SpecLibFlat {
    #[new]
    fn new() -> Self {
        Self {
            precursor_idx: Vec::new(),
            precursor_mz: Vec::new(),
            precursor_rt: Vec::new(),
            precursor_start_idx: Vec::new(),
            precursor_stop_idx: Vec::new(),
            fragment_mz: Vec::new(),
            fragment_intensity: Vec::new(),
        }
    }

    #[staticmethod]
    fn from_arrays(
        precursor_idx: PyReadonlyArray1<'_, usize>,
        precursor_mz: PyReadonlyArray1<'_, f32>,
        precursor_rt: PyReadonlyArray1<'_, f32>,
        precursor_start_idx: PyReadonlyArray1<'_, usize>,
        precursor_stop_idx: PyReadonlyArray1<'_, usize>,
        fragment_mz: PyReadonlyArray1<'_, f32>,
        fragment_intensity: PyReadonlyArray1<'_, f32>,
    ) -> Self {
        // Convert arrays to vectors
        let precursor_idx_vec = precursor_idx.as_array().to_vec();
        let precursor_mz_vec = precursor_mz.as_array().to_vec();
        let precursor_rt_vec = precursor_rt.as_array().to_vec();
        let precursor_start_idx_vec = precursor_start_idx.as_array().to_vec();
        let precursor_stop_idx_vec = precursor_stop_idx.as_array().to_vec();
        let fragment_mz_vec = fragment_mz.as_array().to_vec();
        let fragment_intensity_vec = fragment_intensity.as_array().to_vec();

        // Create indices for sorting
        let mut indices: Vec<usize> = (0..precursor_idx_vec.len()).collect();

        // Sort indices by precursor_idx values
        indices.sort_by_key(|&i| precursor_idx_vec[i]);

        // Reorder all precursor arrays according to sorted indices
        let sorted_precursor_idx: Vec<usize> =
            indices.iter().map(|&i| precursor_idx_vec[i]).collect();
        let sorted_precursor_mz: Vec<f32> = indices.iter().map(|&i| precursor_mz_vec[i]).collect();
        let sorted_precursor_rt: Vec<f32> = indices.iter().map(|&i| precursor_rt_vec[i]).collect();
        let sorted_precursor_start_idx: Vec<usize> = indices
            .iter()
            .map(|&i| precursor_start_idx_vec[i])
            .collect();
        let sorted_precursor_stop_idx: Vec<usize> =
            indices.iter().map(|&i| precursor_stop_idx_vec[i]).collect();

        Self {
            precursor_idx: sorted_precursor_idx,
            precursor_mz: sorted_precursor_mz,
            precursor_rt: sorted_precursor_rt,
            precursor_start_idx: sorted_precursor_start_idx,
            precursor_stop_idx: sorted_precursor_stop_idx,
            fragment_mz: fragment_mz_vec,
            fragment_intensity: fragment_intensity_vec,
        }
    }

    #[getter]
    pub fn num_precursors(&self) -> usize {
        self.precursor_mz.len()
    }

    #[getter]
    pub fn num_fragments(&self) -> usize {
        self.fragment_mz.len()
    }
}

/// Apply fragment filtering and return filtered fragment vectors
pub fn filter_fragments(
    fragment_mz: &[f32],
    fragment_intensity: &[f32],
    non_zero: bool,
    top_k_fragments: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut fragment_triplets: Vec<(f32, f32, usize)> = fragment_mz
        .iter()
        .zip(fragment_intensity.iter())
        .enumerate()
        .map(|(idx, (&mz, &intensity))| (mz, intensity, idx))
        .collect();

    // Filter non-zero intensities if requested
    if non_zero {
        fragment_triplets.retain(|(_, intensity, _)| *intensity > 0.0);
    }

    // Use partial sorting for top-k selection - much faster than full sort
    let k = top_k_fragments.min(fragment_triplets.len());
    if k < fragment_triplets.len() {
        // Partial sort: only sort the k-th element and everything before it
        fragment_triplets.select_nth_unstable_by(k, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        fragment_triplets.truncate(k);
    }
    // Sort by original index to maintain original m/z ordering
    fragment_triplets.sort_by_key(|(_, _, idx)| *idx);

    let (fragment_mz, fragment_intensity): (Vec<f32>, Vec<f32>) = fragment_triplets
        .into_iter()
        .map(|(mz, intensity, _)| (mz, intensity))
        .unzip();

    (fragment_mz, fragment_intensity)
}

// Regular Rust implementation (not exposed to Python)
impl SpecLibFlat {
    pub fn get_precursor(&self, index: usize) -> Precursor {
        let precursor_idx = self.precursor_idx[index];
        let precursor_mz = self.precursor_mz[index];
        let precursor_rt = self.precursor_rt[index];
        let start_idx = self.precursor_start_idx[index];
        let stop_idx = self.precursor_stop_idx[index];

        let fragment_mz = self.fragment_mz[start_idx..stop_idx].to_vec();
        let fragment_intensity = self.fragment_intensity[start_idx..stop_idx].to_vec();

        Precursor {
            idx: precursor_idx,
            mz: precursor_mz,
            rt: precursor_rt,
            fragment_mz,
            fragment_intensity,
        }
    }

    pub fn get_precursor_filtered(
        &self,
        index: usize,
        non_zero: bool,
        top_k_fragments: usize,
    ) -> Precursor {
        let precursor_idx = self.precursor_idx[index];
        let precursor_mz = self.precursor_mz[index];
        let precursor_rt = self.precursor_rt[index];
        let start_idx = self.precursor_start_idx[index];
        let stop_idx = self.precursor_stop_idx[index];

        let raw_fragment_mz = &self.fragment_mz[start_idx..stop_idx];
        let raw_fragment_intensity = &self.fragment_intensity[start_idx..stop_idx];

        let (fragment_mz, fragment_intensity) = filter_fragments(
            raw_fragment_mz,
            raw_fragment_intensity,
            non_zero,
            top_k_fragments,
        );

        Precursor {
            idx: precursor_idx,
            mz: precursor_mz,
            rt: precursor_rt,
            fragment_mz,
            fragment_intensity,
        }
    }

    pub fn get_precursor_by_idx(&self, precursor_idx: usize) -> Option<Precursor> {
        // Use binary search since precursor_idx is now sorted
        match self.precursor_idx.binary_search(&precursor_idx) {
            Ok(array_index) => Some(self.get_precursor(array_index)),
            Err(_) => None,
        }
    }

    pub fn get_precursor_by_idx_filtered(
        &self,
        precursor_idx: usize,
        non_zero: bool,
        top_k_fragments: usize,
    ) -> Option<Precursor> {
        // Use binary search since precursor_idx is now sorted
        match self.precursor_idx.binary_search(&precursor_idx) {
            Ok(array_index) => {
                Some(self.get_precursor_filtered(array_index, non_zero, top_k_fragments))
            }
            Err(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_fragments_no_filtering() {
        let fragment_mz = vec![200.0, 300.0, 400.0, 500.0, 600.0];
        let fragment_intensity = vec![0.0, 10.0, 5.0, 20.0, 0.0];

        let (filtered_mz, filtered_intensity) =
            filter_fragments(&fragment_mz, &fragment_intensity, false, usize::MAX);

        assert_eq!(filtered_mz, vec![200.0, 300.0, 400.0, 500.0, 600.0]);
        assert_eq!(filtered_intensity, vec![0.0, 10.0, 5.0, 20.0, 0.0]);
    }

    #[test]
    fn test_filter_fragments_non_zero_only() {
        let fragment_mz = vec![200.0, 300.0, 400.0, 500.0, 600.0];
        let fragment_intensity = vec![0.0, 10.0, 5.0, 20.0, 0.0];

        let (filtered_mz, filtered_intensity) =
            filter_fragments(&fragment_mz, &fragment_intensity, true, usize::MAX);

        assert_eq!(filtered_mz, vec![300.0, 400.0, 500.0]);
        assert_eq!(filtered_intensity, vec![10.0, 5.0, 20.0]);
        assert!(filtered_intensity.iter().all(|&i| i > 0.0));
    }

    #[test]
    fn test_filter_fragments_top_k_selection() {
        let fragment_mz = vec![200.0, 300.0, 400.0, 500.0, 600.0];
        let fragment_intensity = vec![0.0, 10.0, 5.0, 20.0, 0.0];

        let (filtered_mz, filtered_intensity) =
            filter_fragments(&fragment_mz, &fragment_intensity, false, 2);

        // Top 2: intensity 20.0 (mz 500.0) and 10.0 (mz 300.0), in original order
        assert_eq!(filtered_mz, vec![300.0, 500.0]);
        assert_eq!(filtered_intensity, vec![10.0, 20.0]);
    }

    #[test]
    fn test_filter_fragments_combined_filtering() {
        let fragment_mz = vec![200.0, 300.0, 400.0, 500.0, 600.0];
        let fragment_intensity = vec![0.0, 10.0, 5.0, 20.0, 0.0];

        let (filtered_mz, filtered_intensity) =
            filter_fragments(&fragment_mz, &fragment_intensity, true, 2);

        // Non-zero: [300.0->10.0, 400.0->5.0, 500.0->20.0], top 2: [300.0->10.0, 500.0->20.0]
        assert_eq!(filtered_mz, vec![300.0, 500.0]);
        assert_eq!(filtered_intensity, vec![10.0, 20.0]);
        assert!(filtered_intensity.iter().all(|&i| i > 0.0));
    }

    #[test]
    fn test_filter_fragments_order_preservation() {
        let fragment_mz = vec![600.0, 200.0, 800.0, 100.0, 400.0];
        let fragment_intensity = vec![15.0, 25.0, 5.0, 30.0, 20.0];

        let (filtered_mz, filtered_intensity) =
            filter_fragments(&fragment_mz, &fragment_intensity, false, 3);

        // Top 3: 100.0->30.0, 200.0->25.0, 400.0->20.0 in original index order
        assert_eq!(filtered_mz, vec![200.0, 100.0, 400.0]);
        assert_eq!(filtered_intensity, vec![25.0, 30.0, 20.0]);

        // Verify top-k correctness by checking intensities are the highest 3
        let mut sorted_intensity = filtered_intensity.clone();
        sorted_intensity.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert_eq!(sorted_intensity, vec![30.0, 25.0, 20.0]);
    }

    #[test]
    fn test_filter_fragments_top_k_larger_than_available() {
        let fragment_mz = vec![300.0, 400.0];
        let fragment_intensity = vec![10.0, 5.0];

        let (filtered_mz, filtered_intensity) =
            filter_fragments(&fragment_mz, &fragment_intensity, false, 5);

        assert_eq!(filtered_mz, vec![300.0, 400.0]);
        assert_eq!(filtered_intensity, vec![10.0, 5.0]);
    }

    #[test]
    fn test_filter_fragments_all_zero_intensities() {
        let fragment_mz = vec![300.0, 400.0];
        let fragment_intensity = vec![0.0, 0.0];

        let (filtered_mz, filtered_intensity) =
            filter_fragments(&fragment_mz, &fragment_intensity, true, usize::MAX);

        assert_eq!(filtered_mz, Vec::<f32>::new());
        assert_eq!(filtered_intensity, Vec::<f32>::new());
    }

    #[test]
    fn test_filter_fragments_empty_input() {
        let fragment_mz: Vec<f32> = vec![];
        let fragment_intensity: Vec<f32> = vec![];

        let (filtered_mz, filtered_intensity) =
            filter_fragments(&fragment_mz, &fragment_intensity, false, 5);

        assert_eq!(filtered_mz, Vec::<f32>::new());
        assert_eq!(filtered_intensity, Vec::<f32>::new());
    }

    #[test]
    fn test_filter_fragments_single_fragment() {
        let fragment_mz = vec![500.0];
        let fragment_intensity = vec![10.0];

        let (filtered_mz, filtered_intensity) =
            filter_fragments(&fragment_mz, &fragment_intensity, true, 1);

        assert_eq!(filtered_mz, vec![500.0]);
        assert_eq!(filtered_intensity, vec![10.0]);
    }

    #[test]
    fn test_filter_fragments_top_k_zero() {
        let fragment_mz = vec![200.0, 300.0, 400.0];
        let fragment_intensity = vec![10.0, 20.0, 15.0];

        let (filtered_mz, filtered_intensity) =
            filter_fragments(&fragment_mz, &fragment_intensity, false, 0);

        assert_eq!(filtered_mz, Vec::<f32>::new());
        assert_eq!(filtered_intensity, Vec::<f32>::new());
    }

    #[test]
    fn test_filter_fragments_identical_intensities() {
        let fragment_mz = vec![200.0, 300.0, 400.0, 500.0];
        let fragment_intensity = vec![10.0, 10.0, 10.0, 10.0];

        let (filtered_mz, filtered_intensity) =
            filter_fragments(&fragment_mz, &fragment_intensity, false, 2);

        // When intensities are equal, ordering is implementation-dependent
        // but we should get exactly 2 fragments and preserve the input values
        assert_eq!(filtered_mz.len(), 2);
        assert_eq!(filtered_intensity.len(), 2);
        assert!(filtered_intensity.iter().all(|&i| i == 10.0));
        // Verify all returned m/z values are from the original set
        for mz in &filtered_mz {
            assert!(fragment_mz.contains(mz));
        }
    }
}
