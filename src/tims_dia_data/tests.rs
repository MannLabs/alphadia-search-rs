use super::*;
use crate::dense_xic_observation::DenseXICObservation;
use crate::traits::{DIADataTrait, MobilityFilter};

const FRAGMENT_MZ: f32 = 600.0;

/// One window group over 4 scans (1/K0 1.3, 1.2, 1.1, 1.0). Scans 0-2 isolate 400-500,
/// scan 3 isolates 900-1000. Every scan has one peak at `FRAGMENT_MZ` in cycle 1,
/// with intensity 10^scan so each scan's contribution is visible in the sum.
fn tims_data() -> TimsDIAData {
    let mz_index = MZIndex::global();
    let mut obs = TimsQuadrupoleObservation::new(vec![
        Some([400.0, 500.0]),
        Some([400.0, 500.0]),
        Some([400.0, 500.0]),
        Some([900.0, 1000.0]),
    ]);
    let bin = mz_index.find_closest_index(FRAGMENT_MZ);
    obs.slice_starts = (0..=mz_index.len())
        .map(|i| if i <= bin { 0 } else { 4 })
        .collect();
    obs.cycle_indices = vec![1; 4];
    obs.scan_indices = vec![0, 1, 2, 3];
    obs.intensities = vec![1.0, 10.0, 100.0, 1000.0];

    TimsDIAData {
        rt_index: RTIndex::from_rt(Array1::from_vec(vec![0.0, 1.0, 2.0])),
        quadrupole_observations: vec![obs],
        scan_mobility: vec![1.3, 1.2, 1.1, 1.0],
        rt_values: Array1::zeros(0),
        cycle: Array4::zeros((0, 0, 0, 0)),
    }
}

fn xic(data: &TimsDIAData, precursor_mz: f32, mobility: MobilityFilter) -> Vec<f32> {
    DenseXICObservation::new(data, precursor_mz, mobility, 0, 3, 10.0, &[FRAGMENT_MZ])
        .dense_xic
        .row(0)
        .to_vec()
}

#[test]
fn test_without_mobility_all_isolating_scans_count() {
    let data = tims_data();
    assert_eq!(
        xic(&data, 450.0, MobilityFilter::ANY),
        vec![0.0, 111.0, 0.0]
    );
}

#[test]
fn test_mobility_filter_keeps_only_scans_in_range() {
    let data = tims_data();
    let mobility = MobilityFilter {
        mobility: 1.2,
        tolerance: 0.05,
    };
    assert_eq!(xic(&data, 450.0, mobility), vec![0.0, 10.0, 0.0]);
}

#[test]
fn test_scans_not_isolating_the_precursor_are_ignored() {
    let data = tims_data();
    let mobility = MobilityFilter {
        mobility: 1.0,
        tolerance: 0.05,
    };
    // scan 3 is in mobility range but isolates 900-1000, not 450
    assert_eq!(xic(&data, 450.0, mobility), vec![0.0, 0.0, 0.0]);
    assert_eq!(xic(&data, 950.0, mobility), vec![0.0, 1000.0, 0.0]);
}

#[test]
fn test_precursor_outside_all_windows_has_no_observation() {
    let data = tims_data();
    assert!(data
        .select_observations(700.0, MobilityFilter::ANY)
        .is_empty());
    let selected = data.select_observations(450.0, MobilityFilter::ANY);
    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0].0, 0);
}

#[test]
fn test_invalid_mobility_does_not_restrict_scans() {
    let data = tims_data();
    for mobility in [0.0, f32::INFINITY, -1.0] {
        let filter = MobilityFilter {
            mobility,
            tolerance: 0.05,
        };
        assert_eq!(xic(&data, 450.0, filter), vec![0.0, 111.0, 0.0]);
    }
}

mod fill_group {
    use super::super::builder::fill_group;
    use super::*;

    const NUM_BINS: usize = 3;

    #[test]
    fn test_peaks_on_the_same_bin_cycle_and_scan_are_summed() {
        // bin 0, cycle 1 has two peaks on scan 0 (from two frames) and one on scan 1;
        // bin 2 has one peak on scan 3
        let staged = vec![
            vec![(0, 1, 0, 1.0), (0, 1, 1, 2.0)],
            vec![(0, 1, 0, 3.0), (2, 0, 3, 0.5)],
        ];
        let mut obs = TimsQuadrupoleObservation::new(vec![Some([400.0, 500.0]); 4]);
        fill_group(&mut obs, staged, NUM_BINS).unwrap();
        assert_eq!(obs.slice_starts, vec![0, 2, 2, 3]);
        assert_eq!(obs.cycle_indices, vec![1, 1, 0]);
        assert_eq!(obs.scan_indices, vec![0, 1, 3]);
        assert_eq!(obs.intensities, vec![4.0, 2.0, 0.5]);
    }
}

#[test]
fn test_zero_mobility_tolerance_keeps_the_matching_scan() {
    let data = tims_data();
    let mobility = MobilityFilter {
        mobility: 1.2,
        tolerance: 0.0,
    };
    assert_eq!(xic(&data, 450.0, mobility), vec![0.0, 10.0, 0.0]);
}

#[test]
fn test_mobility_outside_the_scan_range_has_no_observation() {
    let data = tims_data();
    let mobility = MobilityFilter {
        mobility: 2.0,
        tolerance: 0.05,
    };
    assert!(data.select_observations(450.0, mobility).is_empty());
}

mod layout {
    use super::super::layout::{assign_cycles, frame_rt_seconds, window_per_scan, FrameMeta};
    use timsrust::core::{IsolationWindow, Mz, QuadrupoleSettings};

    fn ms1(frame_index: usize, rt_seconds: f32) -> FrameMeta {
        FrameMeta {
            frame_index,
            is_ms1: true,
            settings_id: 0,
            rt_seconds,
        }
    }

    fn ms2(frame_index: usize, settings_id: usize, rt_seconds: f32) -> FrameMeta {
        FrameMeta {
            frame_index,
            is_ms1: false,
            settings_id,
            rt_seconds,
        }
    }

    #[test]
    fn test_frames_are_assigned_to_cycles_and_groups() {
        // an MS2 frame before the first MS1 frame, then two cycles; group ids 7 and 3
        // become groups 0 and 1 in order of first appearance, and cycle 1 lacks group 3
        let frames = vec![
            ms2(1, 3, 0.5),
            ms1(2, 1.0),
            ms2(3, 7, 1.1),
            ms2(4, 3, 1.2),
            ms1(5, 2.0),
            ms2(6, 7, 2.1),
        ];
        let cycles = assign_cycles(frames).unwrap();
        assert_eq!(cycles.cycle_rt_seconds, vec![1.0, 2.0]);
        assert_eq!(cycles.settings_ids, vec![7, 3]);
        let placed: Vec<_> = cycles
            .ms2_frames
            .iter()
            .map(|f| (f.frame_index, f.cycle, f.group))
            .collect();
        assert_eq!(placed, vec![(3, 0, 0), (4, 0, 1), (6, 1, 0)]);
        // the missing frame slot keeps its cycle's MS1 RT
        assert_eq!(
            frame_rt_seconds(&cycles),
            vec![1.0, 1.1, 1.2, 2.0, 2.1, 2.0]
        );
    }

    #[test]
    fn test_runs_without_cycles_or_windows_are_rejected() {
        assert!(assign_cycles(vec![ms1(1, 1.0), ms1(2, 2.0)]).is_err());
        assert!(assign_cycles(vec![ms2(1, 1, 1.0)]).is_err());
    }

    #[test]
    fn test_more_cycles_than_u16_is_an_error() {
        let mut frames: Vec<_> = (0..=u16::MAX as usize + 1)
            .map(|i| ms1(i, i as f32))
            .collect();
        frames.push(ms2(frames.len(), 1, 0.0));
        assert!(assign_cycles(frames).is_err());
    }

    #[test]
    fn test_scan_ends_are_exclusive_and_clamped() {
        let window =
            |lo: f64, hi: f64| IsolationWindow::new_from_bounds(Mz::from(lo), Mz::from(hi), 0.0);
        let settings = QuadrupoleSettings {
            index: 1,
            scan_starts: vec![1, 3],
            scan_ends: vec![3, 9],
            isolation_windows: vec![window(800.0, 825.0), window(600.0, 625.0)],
        };
        let windows = window_per_scan(&settings, 5);
        assert_eq!(
            windows,
            vec![
                None,
                Some([800.0, 825.0]),
                Some([800.0, 825.0]),
                Some([600.0, 625.0]),
                Some([600.0, 625.0]),
            ]
        );
    }

    #[test]
    fn test_settings_without_windows_isolate_nothing() {
        let windows = window_per_scan(&QuadrupoleSettings::default(), 3);
        assert_eq!(windows, vec![None; 3]);
    }
}
