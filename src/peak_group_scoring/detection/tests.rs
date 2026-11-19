mod flanks {
    use crate::peak_group_scoring::detection::*;
    use numpy::ndarray::Array2;

    /// 12 fragments over 19 cycles with the apex in column 9.
    fn empty() -> Array2<f32> {
        Array2::zeros((12, 2 * FLANK_MAX + 1))
    }

    #[test]
    fn window_is_clamped_to_the_run() {
        assert_eq!(window(3, 100), (0, 13));
        assert_eq!(window(50, 100), (41, 60));
        assert_eq!(window(95, 100), (86, 100));
    }

    #[test]
    fn flank_counts_skip_the_peak_region() {
        let mut xic = empty();
        xic[[0, 9]] = 5.0; // apex
        xic[[0, 11]] = 5.0; // too close
        xic[[0, 13]] = 5.0; // flank (distance 4)
        xic[[0, 0]] = 5.0; // flank (distance 9)
        let (hits, scans) = flank_counts(&xic, 9);
        assert_eq!(scans, 12);
        assert_eq!(hits[0], 2);
        assert_eq!(hits[1], 0);
    }

    #[test]
    fn background_rates_shrink_flank_shares_towards_the_run_rate() {
        let mut xic = empty();
        for c in [0, 1, 2, 3, 4, 5, 13, 14, 15, 16, 17, 18] {
            xic[[0, c]] = 5.0;
        }
        let rates = background_rates(&xic, 9, 0.1);
        // 12 of 12 flank scans hit: (12 + 2 x 0.1) / 14; no flank hit: 2 x 0.1 / 14
        assert!((rates[0] - 12.2 / 14.0).abs() < 1e-5);
        assert!((rates[1] - 0.2 / 14.0).abs() < 1e-5);
    }

    #[test]
    fn missing_values_are_all_zero() {
        use crate::peak_group_scoring::feature_block::FeatureBlock;
        assert!(DetectionFeatures::missing()
            .values()
            .iter()
            .all(|&v| v == 0.0));
        assert_eq!(DetectionFeatures::NAMES.len(), 3);
    }
}

mod faint {
    use crate::peak_group_scoring::detection::faint::*;
    use numpy::ndarray::Array2;

    const CURVE: DetectionCurve = DetectionCurve {
        limit: 8.0,
        width: 1.5,
        scatter: 1.0,
    };

    fn model() -> FaintModel {
        // background hits around 2^9 +- 1.3
        let values: Vec<f64> = (0..2000)
            .map(|i| 9.0 + 1.3 * ((i as f64 / 2000.0) * 6.0 - 3.0))
            .collect();
        FaintModel {
            curve: CURVE,
            background: BackgroundDensity::from_log2(&values),
        }
    }

    fn library() -> Vec<f64> {
        // library intensities 1 .. 0.1 over 12 fragments
        (0..12).map(|f| (1.0 - f as f64 * 0.08).log2()).collect()
    }

    fn fragments(hit: &[(usize, f64)]) -> Vec<ApexFragment> {
        library()
            .into_iter()
            .enumerate()
            .map(|(f, log2_relative)| ApexFragment {
                log2_relative,
                log2_intensity: hit.iter().find(|(i, _)| *i == f).map(|(_, y)| *y),
                background_rate: 0.1,
            })
            .collect()
    }

    #[test]
    fn ln_ndtr_matches_reference_values() {
        // scipy.special.log_ndtr
        for (x, expected) in [
            (0.0, -std::f64::consts::LN_2),
            (1.5, -0.069_143_455_612_234),
            (-2.0, -3.783_184_333_682_031_7),
            (-10.0, -53.231_285_150_512_48),
            (5.0, -2.866_516_129_637_6e-7),
        ] {
            let got = ln_ndtr(x);
            assert!(
                (got - expected).abs() < 1e-5 * expected.abs().max(1.0),
                "{x}: {got} vs {expected}"
            );
        }
    }

    #[test]
    fn background_density_integrates_to_one_and_has_a_floor() {
        let m = model();
        let integral: f64 = (0..DENSITY_NUM_BINS)
            .map(|i| {
                m.background
                    .at(DENSITY_MIN_LOG2 + (i as f64 + 0.5) * DENSITY_BIN_LOG2)
                    * DENSITY_BIN_LOG2
            })
            .sum();
        assert!((integral - 1.0).abs() < 0.05);
        assert_eq!(m.background.at(30.0), DENSITY_FLOOR);
        assert!((m.background.default_curve().limit - 8.0).abs() < 0.1);
    }

    #[test]
    fn library_consistent_faint_peptide_beats_a_random_match() {
        // abundance 2^10.5: the six strongest fragments are seen at their predicted intensities
        let consistent: Vec<(usize, f64)> = (0..6).map(|f| (f, 10.5 + library()[f])).collect();
        // as many hits, on weak fragments, at background intensities
        let random: Vec<(usize, f64)> = (6..12).map(|f| (f, 9.0)).collect();
        let c = faint_features(&fragments(&consistent), &model());
        let r = faint_features(&fragments(&random), &model());
        // the log-likelihood ratio is the sum of the hit and the miss terms
        assert!(c.det_faint_hits + c.det_faint_miss > r.det_faint_hits + r.det_faint_miss + 5.0);
        assert!(c.det_faint_hits > 0.0);
        assert!(c.det_faint_miss <= 0.0);
    }

    #[test]
    fn intense_hits_make_missing_strong_fragments_costly() {
        // the same four hit channels, once at the limit and once far above it, with fragments 0-3 missing
        let faint: Vec<(usize, f64)> = (4..8).map(|f| (f, 8.5)).collect();
        let bright: Vec<(usize, f64)> = (4..8).map(|f| (f, 14.0)).collect();
        let f = faint_features(&fragments(&faint), &model());
        let b = faint_features(&fragments(&bright), &model());
        assert!(b.det_faint_miss < f.det_faint_miss);
    }

    #[test]
    fn no_hits_give_zero_and_empty_inputs_missing() {
        // without hits the best abundance is the lowest, where the misses cost almost nothing
        let none = faint_features(&fragments(&[]), &model());
        let llr = none.det_faint_hits + none.det_faint_miss;
        assert!(llr <= 0.0 && llr > -0.5);
        assert_eq!(faint_features(&[], &model()), FaintFeatures::missing());
        let xic = Array2::<f32>::zeros((3, 5));
        assert!(apex_fragments(&xic, 9, &[1.0, 0.5, 0.2], &[0.1, 0.1, 0.1]).is_none());
        assert!(apex_fragments(&xic, 2, &[1.0, 0.5], &[0.1, 0.1]).is_none());
    }

    #[test]
    fn fit_recovers_the_curve_of_simulated_candidates() {
        // deterministic pseudo-random candidates drawn from CURVE with abundances 2^7 .. 2^13
        let mut state: u64 = 20_261_109;
        let mut uniform = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        let normal = |u: &mut dyn FnMut() -> f64| {
            let (u1, u2) = (u(), u());
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };
        let mut candidates = Vec::new();
        for i in 0..1500 {
            let a = 7.0 + 6.0 * (i as f64 / 1500.0);
            let c: Vec<ApexFragment> = library()
                .into_iter()
                .map(|log2_relative| {
                    let y = a + log2_relative + CURVE.scatter * normal(&mut uniform);
                    let seen = (CURVE.width * normal(&mut uniform) + CURVE.limit) < y;
                    ApexFragment {
                        log2_relative,
                        log2_intensity: seen.then_some(y),
                        background_rate: 0.1,
                    }
                })
                .collect();
            candidates.push(c);
        }
        let fitted = fit_curve(&candidates, &model().background);
        assert!((fitted.limit - CURVE.limit).abs() <= 0.5, "{fitted:?}");
        assert!((fitted.width - CURVE.width).abs() <= 0.5, "{fitted:?}");
        assert!((fitted.scatter - CURVE.scatter).abs() <= 0.25, "{fitted:?}");
    }

    #[test]
    fn too_few_hits_fall_back_to_the_default_curve() {
        let fitted = fit_curve(&[fragments(&[(0, 10.0)])], &model().background);
        assert_eq!(fitted, model().background.default_curve());
    }
}

mod shadow {
    use crate::peak_group_scoring::detection::shadow::*;
    use crate::peak_group_scoring::ms1::C13_MASS_DELTA;
    use numpy::ndarray::Array2;

    const N: usize = 4;
    const CYCLES: usize = 7;
    const PEAK: [f32; CYCLES] = [0.0, 1.0, 4.0, 10.0, 4.0, 1.0, 0.0];

    fn fragments() -> Array2<f32> {
        let mut xic = Array2::zeros((N, CYCLES));
        for f in 0..N {
            for c in 0..CYCLES {
                xic[[f, c]] = PEAK[c] * 100.0;
            }
        }
        xic
    }

    fn library() -> Vec<f32> {
        vec![1.0, 0.8, 0.6, 0.4]
    }

    fn mz() -> Vec<f32> {
        vec![500.0, 700.0, 900.0, 1100.0]
    }

    fn score(shadow: &Array2<f32>) -> f32 {
        shadow_lib_share(&fragments(), shadow, &PEAK, &[0.99; N], &library(), &mz())
    }

    #[test]
    fn shadow_mz_lists_1plus_then_2plus_spacings() {
        let out = shadow_mz(&[500.0, 800.0]);
        assert_eq!(out.len(), 4);
        assert!((out[0] - (500.0 - C13_MASS_DELTA)).abs() < 1e-4);
        assert!((out[1] - (800.0 - C13_MASS_DELTA)).abs() < 1e-4);
        assert!((out[2] - (500.0 - C13_MASS_DELTA / 2.0)).abs() < 1e-4);
    }

    #[test]
    fn clean_fragments_are_not_shadowed() {
        assert_eq!(score(&Array2::zeros((2 * N, CYCLES))), 0.0);
    }

    #[test]
    fn a_coeluting_intense_m_minus_1_shadows_the_fragment() {
        let mut shadow = Array2::zeros((2 * N, CYCLES));
        // fragment 0 (500 m/z, library share 0.357): a 1+ ion at M-1 three times as intense
        for c in 0..CYCLES {
            shadow[[0, c]] = PEAK[c] * 300.0;
        }
        assert!((score(&shadow) - 1.0 / 2.8).abs() < 1e-5);
    }

    #[test]
    fn a_weak_or_non_coeluting_shadow_does_not_count() {
        let mut weak = Array2::zeros((2 * N, CYCLES));
        let mut off_peak = Array2::zeros((2 * N, CYCLES));
        for c in 0..CYCLES {
            // M0/M1 of a 500 Da ion is ~3.6, so 0.5 x that needs a shadow of > 1.8 x the fragment
            weak[[0, c]] = PEAK[c] * 100.0;
            off_peak[[0, c]] = PEAK[(c + 3) % CYCLES] * 1000.0;
        }
        assert_eq!(score(&weak), 0.0);
        assert_eq!(score(&off_peak), 0.0);
    }

    #[test]
    fn the_stronger_shadow_charge_is_kept() {
        let mut shadow = Array2::zeros((2 * N, CYCLES));
        for c in 0..CYCLES {
            // fragment 3 (1100 m/z): a weak 1+ shadow and an intense 2+ shadow (a 2+ ion of ~2200 Da: M0/M1 ~0.8)
            shadow[[3, c]] = PEAK[c] * 10.0;
            shadow[[N + 3, c]] = PEAK[c] * 100.0;
        }
        assert!((score(&shadow) - 0.4 / 2.8).abs() < 1e-5);
    }

    #[test]
    fn incoherent_fragments_are_never_shadowed() {
        let mut shadow = Array2::zeros((2 * N, CYCLES));
        for c in 0..CYCLES {
            shadow[[0, c]] = PEAK[c] * 1000.0;
        }
        let share = shadow_lib_share(
            &fragments(),
            &shadow,
            &PEAK,
            &[0.1, 0.99, 0.99, 0.99],
            &library(),
            &mz(),
        );
        assert_eq!(share, 0.0);
    }

    #[test]
    fn mismatched_inputs_give_zero() {
        assert_eq!(score(&Array2::zeros((N, CYCLES))), 0.0);
        assert_eq!(
            shadow_lib_share(
                &fragments(),
                &Array2::zeros((2 * N, CYCLES)),
                &PEAK[..3],
                &[0.99; N],
                &library(),
                &mz(),
            ),
            0.0
        );
    }
}
