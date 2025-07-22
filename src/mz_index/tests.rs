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
        assert_eq!(
            mz_index.find_closest_index(mz_index.mz[mz_index.len() - 1]),
            mz_index.len() - 1
        );
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
        let indices: Vec<usize> = mz_index
            .mz_range_indices(mz_index.mz[start], mz_index.mz[end])
            .collect();
        // Should include the end value now
        assert_eq!(indices, (start..=end).collect::<Vec<_>>());

        // Single element
        let idx = 100;
        let indices: Vec<usize> = mz_index
            .mz_range_indices(mz_index.mz[idx], mz_index.mz[idx])
            .collect();
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
        let indices: Vec<usize> = mz_index
            .mz_range_indices(below_min, mz_index.mz[target_idx])
            .collect();
        assert_eq!(indices, (0..=target_idx).collect::<Vec<_>>());

        // Lower in bounds, upper out of bounds
        let start_idx = mz_index.len() - 50;
        let above_max = mz_index.mz[mz_index.len() - 1] + 10.0;
        let indices: Vec<usize> = mz_index
            .mz_range_indices(mz_index.mz[start_idx], above_max)
            .collect();
        assert_eq!(indices, (start_idx..mz_index.len()).collect::<Vec<_>>());
    }
}
