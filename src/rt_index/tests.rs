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