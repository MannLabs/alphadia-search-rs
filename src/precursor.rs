pub struct Precursor {
    pub idx: usize,
    pub mz: f32,
    pub rt: f32,
    pub fragment_mz: Vec<f32>,
    pub fragment_intensity: Vec<f32>,
}

impl Precursor {
    pub fn get_fragments_filtered(
        &self,
        non_zero: bool,
        top_k_fragments: Option<usize>,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut fragment_triplets: Vec<(f32, f32, usize)> = self
            .fragment_mz
            .iter()
            .zip(self.fragment_intensity.iter())
            .enumerate()
            .map(|(idx, (&mz, &intensity))| (mz, intensity, idx))
            .collect();

        // Filter non-zero intensities if requested
        if non_zero {
            fragment_triplets.retain(|(_, intensity, _)| *intensity > 0.0);
        }

        // Use partial sorting for top-k selection - much faster than full sort
        if let Some(k) = top_k_fragments {
            let k = k.min(fragment_triplets.len());
            if k < fragment_triplets.len() {
                // Partial sort: only sort the k-th element and everything before it
                fragment_triplets.select_nth_unstable_by(k, |a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                fragment_triplets.truncate(k);
            }
            // Sort by original index to maintain original m/z ordering
            fragment_triplets.sort_by_key(|(_, _, idx)| *idx);
        }

        let (fragment_mz, fragment_intensity): (Vec<f32>, Vec<f32>) = fragment_triplets
            .into_iter()
            .map(|(mz, intensity, _)| (mz, intensity))
            .unzip();

        (fragment_mz, fragment_intensity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_precursor() -> Precursor {
        Precursor {
            idx: 0,
            mz: 500.0,
            rt: 100.0,
            fragment_mz: vec![200.0, 300.0, 400.0, 500.0, 600.0],
            fragment_intensity: vec![0.0, 10.0, 5.0, 20.0, 0.0],
        }
    }

    #[test]
    fn test_no_filtering() {
        let precursor = test_precursor();
        let (mz, intensity) = precursor.get_fragments_filtered(false, None);

        assert_eq!(mz, vec![200.0, 300.0, 400.0, 500.0, 600.0]);
        assert_eq!(intensity, vec![0.0, 10.0, 5.0, 20.0, 0.0]);
    }

    #[test]
    fn test_non_zero_filtering() {
        let precursor = test_precursor();
        let (mz, intensity) = precursor.get_fragments_filtered(true, None);

        assert_eq!(mz, vec![300.0, 400.0, 500.0]);
        assert_eq!(intensity, vec![10.0, 5.0, 20.0]);
        assert!(intensity.iter().all(|&i| i > 0.0));
    }

    #[test]
    fn test_top_k_selection() {
        let precursor = test_precursor();
        let (mz, intensity) = precursor.get_fragments_filtered(false, Some(2));

        // Top 2: intensity 20.0 (mz 500.0) and 10.0 (mz 300.0), in original order
        assert_eq!(mz, vec![300.0, 500.0]);
        assert_eq!(intensity, vec![10.0, 20.0]);
        assert!(mz.windows(2).all(|w| w[0] <= w[1])); // Maintains order
    }

    #[test]
    fn test_combined_filtering() {
        let precursor = test_precursor();
        let (mz, intensity) = precursor.get_fragments_filtered(true, Some(2));

        // Non-zero: [300.0->10.0, 400.0->5.0, 500.0->20.0], top 2: [300.0->10.0, 500.0->20.0]
        assert_eq!(mz, vec![300.0, 500.0]);
        assert_eq!(intensity, vec![10.0, 20.0]);
        assert!(intensity.iter().all(|&i| i > 0.0));
        assert!(mz.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn test_ordering_preservation() {
        let precursor = Precursor {
            idx: 0,
            mz: 500.0,
            rt: 100.0,
            fragment_mz: vec![600.0, 200.0, 800.0, 100.0, 400.0],
            fragment_intensity: vec![15.0, 25.0, 5.0, 30.0, 20.0],
        };

        let (mz, intensity) = precursor.get_fragments_filtered(false, Some(3));

        // Top 3: 100.0->30.0, 200.0->25.0, 400.0->20.0 in original index order
        assert_eq!(mz, vec![200.0, 100.0, 400.0]);
        assert_eq!(intensity, vec![25.0, 30.0, 20.0]);

        // Verify top-k correctness
        let mut sorted_intensity = intensity.clone();
        sorted_intensity.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert_eq!(sorted_intensity, vec![30.0, 25.0, 20.0]);
    }

    #[test]
    fn test_top_k_larger_than_available() {
        let small_precursor = Precursor {
            idx: 0,
            mz: 500.0,
            rt: 100.0,
            fragment_mz: vec![300.0, 400.0],
            fragment_intensity: vec![10.0, 5.0],
        };

        let (mz, intensity) = small_precursor.get_fragments_filtered(false, Some(5));
        assert_eq!(mz, vec![300.0, 400.0]);
        assert_eq!(intensity, vec![10.0, 5.0]);
    }

    #[test]
    fn test_all_zero_intensities_filtered() {
        let zero_precursor = Precursor {
            idx: 0,
            mz: 500.0,
            rt: 100.0,
            fragment_mz: vec![300.0, 400.0],
            fragment_intensity: vec![0.0, 0.0],
        };

        let (mz, intensity) = zero_precursor.get_fragments_filtered(true, None);
        assert_eq!(mz, Vec::<f32>::new());
        assert_eq!(intensity, Vec::<f32>::new());
    }
}
