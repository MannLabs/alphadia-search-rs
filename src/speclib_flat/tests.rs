use super::*;

#[test]
fn test_filter_fragments_no_filtering() {
    let fragment_mz = vec![100.0, 200.0, 300.0, 400.0, 500.0];
    let fragment_intensity = vec![0.0, 15.0, 10.0, 20.0, 5.0];

    let (mz, intensity) = filter_fragments(&fragment_mz, &fragment_intensity, false, usize::MAX);

    assert_eq!(mz, fragment_mz);
    assert_eq!(intensity, fragment_intensity);
}

#[test]
fn test_filter_fragments_non_zero_only() {
    let fragment_mz = vec![100.0, 200.0, 300.0, 400.0, 500.0];
    let fragment_intensity = vec![0.0, 15.0, 10.0, 0.0, 5.0];

    let (mz, intensity) = filter_fragments(&fragment_mz, &fragment_intensity, true, usize::MAX);

    assert_eq!(mz, vec![200.0, 300.0, 500.0]);
    assert_eq!(intensity, vec![15.0, 10.0, 5.0]);
    assert!(intensity.iter().all(|&i| i > 0.0));
}

#[test]
fn test_filter_fragments_top_k_selection() {
    let fragment_mz = vec![100.0, 200.0, 300.0, 400.0, 500.0];
    let fragment_intensity = vec![5.0, 15.0, 10.0, 20.0, 8.0];

    let (mz, intensity) = filter_fragments(&fragment_mz, &fragment_intensity, false, 3);

    // Top 3: 20.0, 15.0, 10.0 in original index order
    assert_eq!(mz, vec![200.0, 300.0, 400.0]);
    assert_eq!(intensity, vec![15.0, 10.0, 20.0]);
}

#[test]
fn test_filter_fragments_combined_non_zero_and_top_k() {
    let fragment_mz = vec![100.0, 200.0, 300.0, 400.0, 500.0];
    let fragment_intensity = vec![0.0, 15.0, 10.0, 0.0, 20.0];

    let (mz, intensity) = filter_fragments(&fragment_mz, &fragment_intensity, true, 2);

    // Non-zero: [15.0, 10.0, 20.0], top 2: [20.0, 15.0] in original order
    assert_eq!(mz, vec![200.0, 500.0]);
    assert_eq!(intensity, vec![15.0, 20.0]);
    assert!(intensity.iter().all(|&i| i > 0.0));
}

#[test]
fn test_filter_fragments_order_preservation() {
    let fragment_mz = vec![600.0, 200.0, 800.0, 100.0, 400.0];
    let fragment_intensity = vec![15.0, 25.0, 5.0, 30.0, 20.0];

    let (mz, intensity) = filter_fragments(&fragment_mz, &fragment_intensity, false, 3);

    // Top 3: 30.0, 25.0, 20.0 in original index order: 200.0, 100.0, 400.0
    assert_eq!(mz, vec![200.0, 100.0, 400.0]);
    assert_eq!(intensity, vec![25.0, 30.0, 20.0]);

    // Verify original ordering is preserved
    for i in 1..mz.len() {
        let idx1 = fragment_mz.iter().position(|&x| x == mz[i - 1]).unwrap();
        let idx2 = fragment_mz.iter().position(|&x| x == mz[i]).unwrap();
        assert!(idx1 < idx2);
    }
}

#[test]
fn test_filter_fragments_identical_intensities() {
    let fragment_mz = vec![100.0, 200.0, 300.0, 400.0, 500.0];
    let fragment_intensity = vec![10.0, 15.0, 10.0, 10.0, 10.0];

    let (mz, intensity) = filter_fragments(&fragment_mz, &fragment_intensity, false, 3);

    // Should get highest intensity (15.0) and 2 of the 10.0s in original order
    assert_eq!(mz.len(), 3);
    assert_eq!(intensity.len(), 3);
    assert!(intensity.contains(&15.0));

    // Count the 10.0s (should be 2)
    let count_tens = intensity.iter().filter(|&&x| x == 10.0).count();
    assert_eq!(count_tens, 2);

    // Verify original ordering
    for i in 1..mz.len() {
        let idx1 = fragment_mz.iter().position(|&x| x == mz[i - 1]).unwrap();
        let idx2 = fragment_mz.iter().position(|&x| x == mz[i]).unwrap();
        assert!(idx1 < idx2);
    }
}

#[test]
fn test_filter_fragments_empty_input() {
    let (mz, intensity) = filter_fragments(&[], &[], false, 5);

    assert!(mz.is_empty());
    assert!(intensity.is_empty());
}

#[test]
fn test_filter_fragments_single_fragment() {
    // Non-zero single fragment
    let (mz, intensity) = filter_fragments(&[500.0], &[10.0], true, 1);
    assert_eq!(mz, vec![500.0]);
    assert_eq!(intensity, vec![10.0]);

    // Zero single fragment with non-zero filter
    let (mz, intensity) = filter_fragments(&[500.0], &[0.0], true, 1);
    assert!(mz.is_empty());
    assert!(intensity.is_empty());
}

#[test]
fn test_filter_fragments_all_zero_intensities() {
    let fragment_mz = vec![100.0, 200.0, 300.0];
    let fragment_intensity = vec![0.0, 0.0, 0.0];

    let (mz, intensity) = filter_fragments(&fragment_mz, &fragment_intensity, true, usize::MAX);

    assert!(mz.is_empty());
    assert!(intensity.is_empty());
}

#[test]
fn test_filter_fragments_top_k_zero() {
    let fragment_mz = vec![100.0, 200.0, 300.0];
    let fragment_intensity = vec![10.0, 20.0, 15.0];

    let (mz, intensity) = filter_fragments(&fragment_mz, &fragment_intensity, false, 0);

    assert!(mz.is_empty());
    assert!(intensity.is_empty());
}

#[test]
fn test_filter_fragments_top_k_larger_than_available() {
    let fragment_mz = vec![100.0, 200.0];
    let fragment_intensity = vec![10.0, 20.0];

    let (mz, intensity) = filter_fragments(&fragment_mz, &fragment_intensity, false, 10);

    assert_eq!(mz, fragment_mz);
    assert_eq!(intensity, fragment_intensity);
}
