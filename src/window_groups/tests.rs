use super::*;

#[test]
fn test_without_overlap_every_window_is_one_group_with_exclusive_upper_limit() {
    // given
    let groups = WindowGroups::new(&[[400.0, 425.0], [425.0, 450.0]]);

    // when / then
    assert_eq!(groups.group_of(400.0), groups.group_of(424.9));
    assert_eq!(groups.group_of(425.0), groups.group_of(430.0));
    assert_ne!(groups.group_of(410.0), groups.group_of(430.0));
    assert_eq!(groups.group_of(450.0), 0);
}

#[test]
fn test_overlap_is_a_group_of_its_own() {
    // given
    let groups = WindowGroups::new(&[[400.0, 425.0], [424.0, 450.0]]);

    // when
    let [below, overlap, above] = [410.0, 424.5, 430.0].map(|mz| groups.group_of(mz));

    // then
    assert!(below != overlap && overlap != above && below != above);
    assert_eq!(groups.group_of(425.0), above);
}

#[test]
fn test_nested_window_leaves_one_group_on_both_sides() {
    // given
    let groups = WindowGroups::new(&[[400.0, 450.0], [420.0, 430.0]]);

    // when / then
    assert_eq!(groups.group_of(410.0), groups.group_of(440.0));
    assert_ne!(groups.group_of(410.0), groups.group_of(425.0));
}

#[test]
fn test_outside_all_windows_and_invalid_windows_are_group_0() {
    // given: an MS1 scan without isolation, a NaN limit and two valid windows
    let groups = WindowGroups::new(&[
        [-1.0, -1.0],
        [f32::NAN, 500.0],
        [400.0, 425.0],
        [500.0, 525.0],
    ]);

    // when / then
    for mz in [-1.0, 100.0, 450.0, 600.0, f32::NAN] {
        assert_eq!(groups.group_of(mz), 0, "{mz}");
    }
    assert_ne!(groups.group_of(410.0), 0);
    assert_eq!(WindowGroups::new(&[]).group_of(500.0), 0);
}
