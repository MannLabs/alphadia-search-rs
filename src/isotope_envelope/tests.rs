use super::*;
use crate::dense_xic_observation::DenseXICMZObservation;
use crate::dia_data::{AlphaRawView, DIAData};
use crate::dia_data_builder::DIADataBuilder;
use numpy::ndarray::{Array4, ArrayView1, ArrayView4};

const ISOTOPES: [f32; 4] = [0.5, 0.3, 0.15, 0.05];
/// A heavy peptide: M1 is the most abundant isotope
const HEAVY_ISOTOPES: [f32; 4] = [0.2, 0.35, 0.3, 0.15];
/// M0 and M1 almost equal: a window holding M1.. beats the M0 window for short fragments only
const MIXED_ISOTOPES: [f32; 4] = [0.4, 0.35, 0.18, 0.07];
const FRAGMENT_MZ: f32 = 400.0;
const LOWER_WINDOW: [f32; 2] = [498.0, 500.0];
const UPPER_WINDOW: [f32; 2] = [500.0, 502.0];
/// M0 of a 2+ precursor here lies in the lower window, M1.. in the upper one
const SPLIT_PRECURSOR_MZ: f32 = 499.7;
const GAIN_TOLERANCE: f32 = 0.05;
/// M1 of the split precursor sits 0.2 Th from the window edge, so a few per cent of it leak into the lower
/// window. The rectangle model ignores the leak, which is a sizeable share of a heavy precursor's small M0.
const HEAVY_GAIN_TOLERANCE: f32 = 0.15;

/// Two adjacent 2 Th windows, [498, 500] and [500, 502], in one cycle, recording the given (m/z, intensity) peaks.
fn two_window_data(lower: &[(f32, f32)], upper: &[(f32, f32)]) -> DIAData {
    let delta_scan_idx = [0_i64, 1];
    let isolation_lower = [LOWER_WINDOW[0], UPPER_WINDOW[0]];
    let isolation_upper = [LOWER_WINDOW[1], UPPER_WINDOW[1]];
    let peak_start = [0_i64, lower.len() as i64];
    let peak_stop = [lower.len() as i64, (lower.len() + upper.len()) as i64];
    let cycle_idx = [0_i64, 0];
    let rt = [10.0_f32, 10.1];
    let peak_mz: Vec<f32> = lower.iter().chain(upper).map(|p| p.0).collect();
    let peak_intensity: Vec<f32> = lower.iter().chain(upper).map(|p| p.1).collect();
    let cycle = [1.0_f32];
    let view = AlphaRawView {
        spectrum_delta_scan_idx: ArrayView1::from(&delta_scan_idx),
        isolation_lower_mz: ArrayView1::from(&isolation_lower),
        isolation_upper_mz: ArrayView1::from(&isolation_upper),
        spectrum_peak_start_idx: ArrayView1::from(&peak_start),
        spectrum_peak_stop_idx: ArrayView1::from(&peak_stop),
        spectrum_cycle_idx: ArrayView1::from(&cycle_idx),
        spectrum_rt: ArrayView1::from(&rt),
        peak_mz: ArrayView1::from(&peak_mz),
        peak_intensity: ArrayView1::from(&peak_intensity),
        cycle: ArrayView4::from_shape([1, 1, 1, 1], &cycle).unwrap(),
    };
    DIADataBuilder::from_alpha_raw(&view)
}

fn empty_data() -> DIAData {
    two_window_data(&[], &[])
}

fn precursor(mz: f32, charge: u8, isotope_intensity: Vec<f32>) -> Precursor {
    Precursor {
        precursor_idx: 0,
        mz,
        mz_library: mz,
        rt: 10.0,
        rt_library: 10.0,
        naa: 9,
        charge,
        isotope_intensity,
        fragment_mz: vec![FRAGMENT_MZ],
        fragment_mz_library: vec![FRAGMENT_MZ],
        fragment_intensity: vec![1.0],
        fragment_cardinality: vec![1],
        fragment_charge: vec![1],
        fragment_loss_type: vec![0],
        fragment_number: vec![3],
        fragment_position: vec![3],
        fragment_type: vec![121],
    }
}

/// The plan of a 2+ precursor at `SPLIT_PRECURSOR_MZ` with singly charged fragments of equal library intensity.
fn split_plan(isotopes: &[f32], fragment_mz: &[f32]) -> ExtractionPlan {
    ExtractionPlan::isotope_envelope(
        &empty_data(),
        SPLIT_PRECURSOR_MZ,
        2,
        isotopes,
        fragment_mz,
        &vec![1; fragment_mz.len()],
        &vec![1.0; fragment_mz.len()],
    )
}

fn mass_share(fragment_mz: f32) -> f32 {
    (fragment_mz - PROTON_MASS) / ((SPLIT_PRECURSOR_MZ - PROTON_MASS) * 2.0)
}

/// Expected signal of fragment isotope `j` in a window holding precursor isotopes `from..to` only (rectangle
/// model), relative to the monoisotopic fragment of the full envelope.
fn rectangle_gain(isotopes: &[f32], from: usize, to: usize, j: usize, share: f32) -> f32 {
    let reference: f32 = isotopes
        .iter()
        .enumerate()
        .map(|(k, &abundance)| abundance * binomial(k, 0, share))
        .sum();
    isotopes[..to]
        .iter()
        .enumerate()
        .skip(from)
        .map(|(k, &abundance)| abundance * binomial(k, j, share))
        .sum::<f32>()
        / reference
}

/// Gain of fragment isotope `j` in the upper window (M1..) relative to the monoisotopic channel of the lower
/// window (M0), rectangle model.
fn upper_relative_gain(isotopes: &[f32], j: usize, fragment_mz: f32) -> f32 {
    let share = mass_share(fragment_mz);
    rectangle_gain(isotopes, 1, isotopes.len(), j, share) / rectangle_gain(isotopes, 0, 1, 0, share)
}

fn assert_close(actual: f32, expected: f32, tolerance: f32) {
    assert!(
        (actual / expected - 1.0).abs() < tolerance,
        "{actual} vs {expected}"
    );
}

#[test]
fn test_transmission_is_one_inside_half_at_edges_zero_outside() {
    assert!((transmission(501.0, 500.0, 502.0) - 1.0).abs() < 1e-3);
    assert!((transmission(500.0, 500.0, 502.0) - 0.5).abs() < 1e-3);
    assert!((transmission(502.0, 500.0, 502.0) - 0.5).abs() < 1e-3);
    assert!(transmission(498.5, 500.0, 502.0) < 1e-3);
    assert_eq!(transmission(450.0, 500.0, 502.0), 0.0);
}

#[test]
fn test_binomial_sums_to_one_over_fragment_isotopes() {
    for k in 0..5 {
        let total: f32 = (0..=k).map(|j| binomial(k, j, 0.3)).sum();
        assert!((total - 1.0).abs() < 1e-6);
    }
    assert_eq!(binomial(1, 2, 0.3), 0.0);
    assert!((binomial(2, 1, 0.3) - 2.0 * 0.3 * 0.7).abs() < 1e-6);
}

#[test]
fn test_monoisotopic_plan_uses_windows_containing_precursor() {
    let dia_data = empty_data();

    let plan = ExtractionPlan::monoisotopic(&dia_data, 499.0, &[300.0, 400.0]);

    assert_eq!(plan.primary.observation_indices(), vec![0]);
    assert_eq!(plan.primary.channels.len(), 2);
    assert!(plan.primary.channels.iter().all(|c| c.mz_shift == 0.0));
    assert_eq!(plan.primary.fragment_gain, vec![1.0, 1.0]);
    assert!(plan.secondary.is_none());
}

#[test]
fn test_unusable_isotopes_fall_back_to_monoisotopic_plan() {
    let dia_data = empty_data();
    for (charge, isotopes) in [
        (0, Vec::new()),
        (2, vec![f32::NAN, 0.3]),
        (2, vec![0.0, 0.0]),
    ] {
        let plan = ExtractionPlan::new(&dia_data, &precursor(499.8, charge, isotopes));

        // M0 at 499.8 is in the lower window only
        assert_eq!(plan.primary.observation_indices(), vec![0]);
        assert_eq!(plan.primary.fragment_gain, vec![1.0]);
        assert!(plan.secondary.is_none());
    }
}

#[test]
fn test_envelope_inside_one_window_is_extracted_there_at_gain_one() {
    // Given: M0..M3 of a 2+ precursor at 498.6..500.1 lie almost fully in [498, 500]
    let plan = ExtractionPlan::new(&empty_data(), &precursor(498.6, 2, ISOTOPES.to_vec()));

    assert_eq!(plan.primary.observation_indices(), vec![0]);
    assert_eq!(plan.primary.channels[0].mz_shift, 0.0);
    assert_eq!(plan.primary.fragment_gain, vec![1.0]);
    // the upper window expects less than SECONDARY_MIN_SIGNAL_RATIO of the signal
    assert!(plan.secondary.is_none());
}

#[test]
fn test_own_window_reads_the_monoisotopic_fragment_only() {
    // Given: a heavy precursor whose envelope lies inside the lower window (M0..M3 at 498.6..500.1), and a
    // long fragment (share ~0.9) whose first isotope would be stronger than its monoisotope there
    let plan = ExtractionPlan::isotope_envelope(
        &empty_data(),
        498.6,
        2,
        &HEAVY_ISOTOPES,
        &[900.0],
        &[1],
        &[1.0],
    );

    // Then: it keeps the monoisotopic channel of that window at the library intensity
    assert_eq!(plan.primary.observation_indices(), vec![0]);
    assert_eq!(plan.primary.channels[0].mz_shift, 0.0);
    assert_eq!(plan.primary.fragment_gain, vec![1.0]);
}

#[test]
fn test_fragments_stay_in_the_window_of_the_precursor_mz_while_its_channel_is_strongest() {
    // Given: M0 (499.7) of a light 2+ precursor in the lower window, M1..M3 in the upper one, a short
    // fragment (share ~0.2) and a long one (~0.6)
    let plan = split_plan(&ISOTOPES, &[200.0, 600.0]);

    // Then: both read their monoisotopic channel in the lower window at the library intensity
    assert_eq!(plan.primary.observation_indices(), vec![0]);
    assert!(plan.primary.channels.iter().all(|c| c.mz_shift == 0.0));
    assert_eq!(plan.primary.fragment_gain, vec![1.0, 1.0]);

    // and the upper window is their secondary channel: the short fragment at its monoisotope, the long
    // one at its first isotope, with gains relative to the lower window
    let secondary = plan.secondary.as_ref().expect("secondary channels");
    assert_eq!(secondary.observation_indices(), vec![1]);
    let shifts: Vec<f32> = secondary.channels.iter().map(|c| c.mz_shift).collect();
    assert_eq!(shifts, vec![0.0, ISOTOPE_SPACING]);
    assert_close(
        secondary.fragment_gain[0],
        upper_relative_gain(&ISOTOPES, 0, 200.0),
        GAIN_TOLERANCE,
    );
    assert_close(
        secondary.fragment_gain[1],
        upper_relative_gain(&ISOTOPES, 1, 600.0),
        GAIN_TOLERANCE,
    );
}

#[test]
fn test_each_fragment_reads_its_strongest_channel() {
    // Given: the heavy precursor's M0 in the lower window, M1..M3 in the upper one
    let plan = split_plan(&HEAVY_ISOTOPES, &[200.0, 600.0]);

    // Then: both fragments move to the upper window, the short one at its monoisotope, the long one at its
    // first isotope, with library intensities scaled up relative to the lower window
    assert_eq!(plan.primary.observation_indices(), vec![1]);
    let shifts: Vec<f32> = plan.primary.channels.iter().map(|c| c.mz_shift).collect();
    assert_eq!(shifts, vec![0.0, ISOTOPE_SPACING]);
    assert!((plan.primary.channels[1].mz - (600.0 + ISOTOPE_SPACING)).abs() < 1e-4);
    assert_close(
        plan.primary.fragment_gain[0],
        upper_relative_gain(&HEAVY_ISOTOPES, 0, 200.0),
        HEAVY_GAIN_TOLERANCE,
    );
    assert_close(
        plan.primary.fragment_gain[1],
        upper_relative_gain(&HEAVY_ISOTOPES, 1, 600.0),
        HEAVY_GAIN_TOLERANCE,
    );
    assert!(plan.primary.fragment_gain.iter().all(|&g| g > 1.5));

    // and the lower window's monoisotopic channels are secondary, at the library intensity
    let secondary = plan.secondary.as_ref().expect("secondary channels");
    assert_eq!(secondary.observation_indices(), vec![0]);
    assert_eq!(secondary.fragment_gain, vec![1.0, 1.0]);
}

#[test]
fn test_fragments_of_one_precursor_can_read_different_windows() {
    // Given: M0 and M1 almost equal; a short fragment (share 0.15) gains from M1..M3 in the upper window,
    // a long one (0.9) spreads their signal over its isotopes and keeps the lower window
    let plan = split_plan(&MIXED_ISOTOPES, &[150.0, 900.0]);

    assert_eq!(plan.primary.channels.len(), 2);
    assert_eq!(plan.primary.channels[0].observation_idx, 1);
    assert_eq!(plan.primary.channels[1].observation_idx, 0);
    assert!(plan.primary.channels.iter().all(|c| c.mz_shift == 0.0));
    assert!(plan.primary.fragment_gain[0] > 1.0);
    assert_eq!(plan.primary.fragment_gain[1], 1.0);

    // and each secondary channel is in the other window
    let secondary = plan.secondary.as_ref().expect("secondary channels");
    assert_eq!(secondary.channels[0].observation_idx, 0);
    assert_eq!(secondary.channels[1].observation_idx, 1);
}

#[test]
fn test_expected_intensity_scales_library_by_gain() {
    let channels = ChannelSet {
        channels: Vec::new(),
        fragment_gain: vec![1.0, 0.5, 0.0],
    };

    assert_eq!(
        channels.expected_intensity(&[2.0, 2.0, 2.0]),
        vec![2.0, 1.0, 0.0]
    );
}

#[test]
fn test_dense_mz_reports_monoisotopic_mz_for_isotope_channels() {
    // Given: the long fragment's first isotope recorded in the upper window, its strongest channel
    let dia_data = two_window_data(&[], &[(600.0 + ISOTOPE_SPACING, 500.0)]);
    let plan = ExtractionPlan::isotope_envelope(
        &dia_data,
        SPLIT_PRECURSOR_MZ,
        2,
        &HEAVY_ISOTOPES,
        &[600.0],
        &[1],
        &[1.0],
    );

    // When
    let dense = DenseXICMZObservation::new(&dia_data, &plan.primary, 0, 1, 20.0);

    // Then: its signal is found and its m/z is reported at the monoisotopic fragment
    assert_eq!(dense.dense_xic[[0, 0]], 500.0);
    let mz_error_ppm = (dense.dense_mz[[0, 0]] - 600.0) / 600.0 * 1e6;
    assert!(mz_error_ppm.abs() < 5.0, "{mz_error_ppm} ppm");
}

#[test]
fn test_primary_window_mz_is_the_window_expecting_most_fragment_signal() {
    let windows = vec![[-1.0, -1.0], LOWER_WINDOW, UPPER_WINDOW];
    let fragments = [200.0, 600.0];
    let charges = [1, 1];
    let intensities = [1.0, 1.0];
    let window_mz = |mz: f32, charge: u8, isotopes: &[f32]| {
        primary_window_mz(
            mz,
            charge,
            isotopes,
            &fragments,
            &charges,
            &intensities,
            &windows,
        )
    };

    assert_eq!(window_mz(SPLIT_PRECURSOR_MZ, 2, &HEAVY_ISOTOPES), 501.0);
    assert_eq!(window_mz(SPLIT_PRECURSOR_MZ, 2, &ISOTOPES), 499.0);
    assert_eq!(window_mz(498.6, 2, &ISOTOPES), 499.0);
    assert_eq!(
        window_mz(SPLIT_PRECURSOR_MZ, 0, &HEAVY_ISOTOPES),
        SPLIT_PRECURSOR_MZ
    );
    assert_eq!(
        window_mz(SPLIT_PRECURSOR_MZ, 2, &[0.0, 0.0]),
        SPLIT_PRECURSOR_MZ
    );
    assert_eq!(window_mz(300.0, 2, &ISOTOPES), 300.0);
}

#[test]
fn test_cycle_windows_span_all_scans() {
    let mut cycle = Array4::<f32>::zeros((1, 2, 2, 2));
    cycle[[0, 0, 0, 0]] = -1.0;
    cycle[[0, 0, 0, 1]] = -1.0;
    cycle[[0, 0, 1, 0]] = -1.0;
    cycle[[0, 0, 1, 1]] = -1.0;
    cycle[[0, 1, 0, 0]] = 500.0;
    cycle[[0, 1, 0, 1]] = 501.0;
    cycle[[0, 1, 1, 0]] = 500.5;
    cycle[[0, 1, 1, 1]] = 502.0;

    assert_eq!(
        cycle_windows(cycle.view()),
        vec![[-1.0, -1.0], [500.0, 502.0]]
    );
}
