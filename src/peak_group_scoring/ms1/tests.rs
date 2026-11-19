use crate::peak_group_scoring::feature_block::FeatureBlock;
use crate::peak_group_scoring::ms1::accuracy::set_accuracy_features;
use crate::peak_group_scoring::ms1::fine::set_fine_features;
use crate::peak_group_scoring::ms1::*;
use numpy::ndarray::{arr2, Array2};

mod apex {
    use super::*;

    #[test]
    fn isotope_mz_starts_one_isotope_below_and_uses_charge_spacing() {
        let mz = isotope_mz(500.0, 2, 3);
        assert_eq!(mz.len(), 4);
        assert!((mz[0] - (500.0 - C13_MASS_DELTA / 2.0)).abs() < 1e-4);
        assert!((mz[1] - 500.0).abs() < 1e-4);
        assert!((mz[3] - (500.0 + C13_MASS_DELTA)).abs() < 1e-4);
    }

    #[test]
    fn coeluting_envelope_at_the_apex_scores_well() {
        let abundance = [1.0, 0.6, 0.2];
        let profile = [0.0, 0.0, 1.0, 3.0, 1.0, 0.0, 0.0];
        let xic = arr2(&[
            [5.0, 5.0, 0.0, 0.0, 0.0, 5.0, 5.0],
            [1.0, 1.0, 10.0, 30.0, 10.0, 1.0, 1.0],
            [0.0, 0.0, 6.0, 18.0, 6.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 6.0, 2.0, 0.0, 0.0],
        ]);
        let mz = xic.mapv(|x| if x > 0.0 { 500.001 } else { 0.0 });

        let (f, iso_m1_err) = calculate_ms1_apex_features(&xic, &mz, 500.0, &abundance, &profile);

        assert!(f.ms1_iso_cosine > 0.99);
        assert_eq!(f.ms1_n_isotopes, 3.0);
        assert!((f.ms1_log_m0_apex - 51.0f32.log10()).abs() < 1e-5);
        assert!(iso_m1_err.abs() < 0.05);
        assert!((f.ms1_ppm_signed - 2.0).abs() < 0.1);
    }

    #[test]
    fn too_intense_m1_gives_a_positive_isotope_error() {
        let profile = [1.0, 3.0, 1.0];
        let xic = arr2(&[[0.0; 3], [10.0, 30.0, 10.0], [20.0, 60.0, 20.0]]);
        let mz = Array2::<f32>::zeros((3, 3));

        let (_, iso_m1_err) = calculate_ms1_apex_features(&xic, &mz, 500.0, &[1.0, 0.5], &profile);

        // observed M+1/M0 about 2, library 0.5: log2(2) - log2(0.5) = 2
        assert!((iso_m1_err - 2.0).abs() < 0.05);
    }

    #[test]
    fn signal_away_from_the_apex_does_not_count() {
        let profile = [0.0, 1.0, 3.0, 1.0, 0.0, 0.0, 0.0];
        let xic = arr2(&[
            [0.0; 7],
            [0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 40.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 24.0],
        ]);
        let mz = Array2::<f32>::zeros((3, 7));

        let (f, iso_m1_err) = calculate_ms1_apex_features(&xic, &mz, 500.0, &[1.0, 0.6], &profile);

        assert_eq!(f.ms1_n_isotopes, 0.0);
        assert_eq!(f.ms1_log_m0_apex, 0.0);
        assert_eq!(iso_m1_err, 0.0);
        assert_eq!(f.ms1_ppm_signed, 0.0);
    }

    #[test]
    fn no_isotopes_mismatched_or_empty_traces_give_zero() {
        let xic = Array2::<f32>::zeros((1, 3));
        let mz = Array2::<f32>::zeros((1, 3));
        let zero = (Ms1ApexFeatures::default(), 0.0);
        assert_eq!(
            calculate_ms1_apex_features(&xic, &mz, 500.0, &[], &[1.0, 2.0, 1.0]),
            zero
        );
        assert_eq!(
            calculate_ms1_apex_features(&xic, &mz, 500.0, &[1.0, 0.5], &[1.0, 2.0, 1.0]),
            zero
        );
        let empty = Array2::<f32>::zeros((3, 5));
        let empty_mz = Array2::<f32>::zeros((3, 5));
        assert_eq!(
            calculate_ms1_apex_features(&empty, &empty_mz, 500.0, &[1.0, 0.5], &[0.0; 5]),
            zero
        );
    }

    #[test]
    fn apex_feature_names_are_the_calibration_columns() {
        assert_eq!(
            Ms1ApexFeatures::NAMES,
            &[
                "ms1_iso_cosine",
                "ms1_n_isotopes",
                "ms1_log_m0_apex",
                "ms1_ppm_signed"
            ]
        );
    }

    #[test]
    fn missing_values_are_all_zero() {
        assert!(Ms1Features::missing().values().iter().all(|&v| v == 0.0));
        assert_eq!(Ms1Features::NAMES.len(), 10);
    }
}

mod profile {
    use super::*;
    use crate::peak_group_scoring::ms1::profile::set_profile_features;

    fn profile_features(
        xic: &Array2<f32>,
        abundance: &[f32],
        profile: &[f32],
        apex: usize,
    ) -> Ms1Features {
        let mut features = Ms1Features::missing();
        set_profile_features(&mut features, xic, abundance, profile, apex);
        features
    }

    /// M-1, M0..M+3 traces of a co-eluting envelope with library abundances 1, 0.6, 0.2, 0.05
    fn coeluting() -> (Array2<f32>, [f32; 4], [f32; 7]) {
        let profile = [0.0, 0.0, 1.0, 3.0, 1.0, 0.0, 0.0];
        let xic = arr2(&[
            [0.0; 7],
            [0.0, 0.0, 10.0, 30.0, 10.0, 0.0, 0.0],
            [0.0, 0.0, 6.0, 18.0, 6.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 6.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 1.5, 0.5, 0.0, 0.0],
        ]);
        (xic, [1.0, 0.6, 0.2, 0.05], profile)
    }

    #[test]
    fn coeluting_envelope_scores_well() {
        let (xic, abundance, profile) = coeluting();

        let f = profile_features(&xic, &abundance, &profile, 3);

        assert_eq!(f.ms1_signal_at_apex, 1.0);
        assert!(f.ms1_isotope_correlation > 0.999);
        assert!(f.ms1_isotope_cosine > 0.999);
        assert_eq!(f.ms1_m_minus_1_frac, 0.0);
        // M0 = 10 x profile, profile maximum 3
        assert!((f.ms1_log_intensity - 31.0f32.log10()).abs() < 1e-5);
    }

    #[test]
    fn only_m0_to_m2_enter_the_cosine() {
        let (mut xic, abundance, profile) = coeluting();
        // an M+3 far off the library abundance changes nothing
        xic.row_mut(4).assign(&numpy::ndarray::arr1(&[
            0.0, 0.0, 50.0, 150.0, 50.0, 0.0, 0.0,
        ]));

        let f = profile_features(&xic, &abundance, &profile, 3);

        assert!(f.ms1_isotope_cosine > 0.999);
    }

    #[test]
    fn signal_off_the_profile_does_not_count() {
        let profile = [0.0, 1.0, 3.0, 1.0, 0.0, 0.0, 0.0];
        let xic = arr2(&[
            [0.0; 7],
            [0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 40.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 24.0],
        ]);

        let f = profile_features(&xic, &[1.0, 0.6], &profile, 2);

        assert_eq!(f.ms1_signal_at_apex, 0.0);
        assert!(f.ms1_isotope_correlation < 0.0);
        assert_eq!(f.ms1_isotope_cosine, 0.0);
        assert_eq!(f.ms1_log_intensity, 0.0);
    }

    #[test]
    fn another_precursors_isotope_raises_the_m_minus_1_frac() {
        let profile = [1.0, 3.0, 1.0];
        let xic = arr2(&[[30.0, 90.0, 30.0], [10.0, 30.0, 10.0], [6.0, 18.0, 6.0]]);

        let f = profile_features(&xic, &[1.0, 0.6], &profile, 1);

        assert!((f.ms1_m_minus_1_frac - 0.75).abs() < 1e-5);
    }

    #[test]
    fn no_isotopes_mismatched_or_empty_traces_give_zero() {
        let xic = Array2::<f32>::zeros((1, 3));
        let profile = [1.0, 2.0, 1.0];
        let zero = Ms1Features::missing();
        assert_eq!(profile_features(&xic, &[], &profile, 1), zero);
        assert_eq!(profile_features(&xic, &[1.0, 0.5], &profile, 1), zero);
        let empty = Array2::<f32>::zeros((3, 5));
        assert_eq!(profile_features(&empty, &[1.0, 0.5], &[0.0; 5], 0), zero);
    }
}

mod accuracy {
    use super::*;

    fn accuracy_features(scan_xics: &[Array2<f32>], profile: &[f32]) -> Ms1Features {
        let mut features = Ms1Features::missing();
        set_accuracy_features(&mut features, scan_xics, profile);
        features
    }

    #[test]
    fn accurate_shares_count_apex_scans_with_signal() {
        let profile = [0.0, 1.0, 3.0, 1.0, 0.0];
        let scan_a = arr2(&[[0.0, 5.0, 5.0, 0.0, 9.0], [0.0, 0.0, 2.0, 0.0, 0.0]]);
        let scan_b = arr2(&[[0.0, 0.0, 5.0, 5.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]);

        let f = accuracy_features(&[scan_a, scan_b], &profile);

        // apex cycle 2, apex window cycles 1..=3: M0 in 4 of 6 points, M+1 in 1 of 6; cycle 4 is outside
        assert!((f.ms1_m0_accurate_share - 4.0 / 6.0).abs() < 1e-6);
        assert!((f.ms1_m1_accurate_share - 1.0 / 6.0).abs() < 1e-6);
    }

    #[test]
    fn accurate_shares_without_scans_are_zero() {
        assert_eq!(accuracy_features(&[], &[1.0, 2.0]), Ms1Features::missing());
    }
}

mod fine {
    use super::*;

    fn gaussian(t: f32, centre: f32, sigma: f32) -> f32 {
        (-0.5 * ((t - centre) / sigma).powi(2)).exp()
    }

    fn fine_features(
        m0: &Array2<f32>,
        m1: &Array2<f32>,
        positions: &[f32],
        cycle_rt: &[f32],
        profile: &[f32],
        fwhm: f32,
    ) -> Ms1Features {
        let mut features = Ms1Features::missing();
        set_fine_features(
            &mut features,
            m0,
            m1,
            positions,
            cycle_rt,
            0.5,
            profile,
            fwhm,
        );
        features
    }

    /// 3 MS1 scans per 1.5 s cycle at positions 0, 1/3, 2/3; the precursor's MS2 window at position 0.5.
    fn traces(centre: f32, sigma: f32, cycle_rt: &[f32]) -> (Array2<f32>, Vec<f32>, Vec<f32>) {
        let positions = vec![0.0, 1.0 / 3.0, 2.0 / 3.0];
        let m0 = Array2::from_shape_fn((3, cycle_rt.len()), |(j, c)| {
            100.0 * gaussian(cycle_rt[c] + positions[j] * 1.5, centre, sigma)
        });
        let profile = cycle_rt
            .iter()
            .map(|&t| gaussian(t + 0.75, centre, sigma))
            .collect();
        (m0, positions, profile)
    }

    #[test]
    fn coeluting_ms1_peak_scores_well() {
        let cycle_rt: Vec<f32> = (0..7).map(|c| 100.0 + 1.5 * c as f32).collect();
        let (m0, positions, profile) = traces(104.6, 1.3, &cycle_rt);
        let m1 = m0.mapv(|v| 0.6 * v);

        let f = fine_features(&m0, &m1, &positions, &cycle_rt, &profile, 3.0);

        assert!(f.ms1_fine_shape > 0.95);
        assert!(f.ms1_fine_iso_corr > 0.99);
    }

    #[test]
    fn ms1_peak_off_the_fragment_apex_does_not_fit_the_shape() {
        let cycle_rt: Vec<f32> = (0..7).map(|c| 100.0 + 1.5 * c as f32).collect();
        let (_, positions, profile) = traces(103.0, 1.0, &cycle_rt);
        let (m0, _, _) = traces(109.0, 1.0, &cycle_rt);

        let f = fine_features(&m0, &m0, &positions, &cycle_rt, &profile, 2.4);

        assert!(f.ms1_fine_shape < 0.2);
    }

    #[test]
    fn no_signal_or_mismatched_shapes_give_zero() {
        let cycle_rt: Vec<f32> = (0..5).map(|c| 1.5 * c as f32).collect();
        let empty = Array2::<f32>::zeros((3, 5));
        let profile = [0.0, 1.0, 3.0, 1.0, 0.0];
        let f = fine_features(&empty, &empty, &[0.0, 0.3, 0.6], &cycle_rt, &profile, 2.0);
        assert_eq!(f, Ms1Features::missing());

        let wrong = Array2::<f32>::zeros((2, 5));
        assert_eq!(
            fine_features(&wrong, &wrong, &[0.0, 0.3, 0.6], &cycle_rt, &profile, 2.0),
            Ms1Features::missing()
        );
    }
}
