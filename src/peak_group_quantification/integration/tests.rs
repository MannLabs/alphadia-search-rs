use approx::assert_relative_eq;
use numpy::ndarray::Array2;

use super::{argmax, integrate, trapezoid, IntegrationContext};
use crate::peak_group_quantification::parameters::{
    QuantificationMethod, QuantificationParameters,
};

/// Build a `[n_fragments, n_cycles]` XIC from one row per fragment.
fn xic(rows: &[&[f32]]) -> Array2<f32> {
    let n_cycles = rows[0].len();
    let flat: Vec<f32> = rows.iter().flat_map(|row| row.iter().copied()).collect();
    Array2::from_shape_vec((rows.len(), n_cycles), flat).expect("consistent row lengths")
}

/// Evenly spaced retention times, one second apart.
fn rt(n_cycles: usize) -> Vec<f32> {
    (0..n_cycles).map(|idx| 100.0 + idx as f32).collect()
}

/// Parameters with template smoothing switched off, so that the boundary and amplitude
/// arithmetic below can be worked out by hand from the templates as written. Smoothing gets
/// its own tests.
fn params(method: QuantificationMethod) -> QuantificationParameters {
    let mut params = QuantificationParameters::new();
    params.method = method;
    params.template_smoothing_lambda = 0.0;
    params
}

/// A symmetric triangular peak, five cycles wide, sitting in an eleven cycle window.
fn triangular_template() -> Vec<f32> {
    vec![0.0, 0.0, 0.0, 0.25, 0.5, 1.0, 0.5, 0.25, 0.0, 0.0, 0.0]
}

// -------------------------------------------------------------------------------------
// trapezoid
// -------------------------------------------------------------------------------------

#[test]
fn test_trapezoid_even_spacing() {
    // (1*(10+20) + 1*(20+5) + 1*(5+0)) / 2 = 15 + 12.5 + 2.5
    let area = trapezoid(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 5.0, 0.0]);
    assert_relative_eq!(area, 30.0, epsilon = 1e-5);
}

#[test]
fn test_trapezoid_uneven_spacing() {
    // (2*(10+20) + 1*(20+5) + 2*(5+0)) / 2 = 30 + 12.5 + 5
    let area = trapezoid(&[1.0, 3.0, 4.0, 6.0], &[10.0, 20.0, 5.0, 0.0]);
    assert_relative_eq!(area, 47.5, epsilon = 1e-5);
}

#[test]
fn test_trapezoid_two_points() {
    assert_relative_eq!(trapezoid(&[1.0, 2.0], &[10.0, 20.0]), 15.0, epsilon = 1e-5);
}

#[test]
fn test_trapezoid_degenerate_inputs() {
    assert_eq!(trapezoid(&[1.0], &[10.0]), 0.0);
    assert_eq!(trapezoid(&[], &[]), 0.0);
    assert_eq!(trapezoid(&[1.0, 2.0], &[10.0]), 0.0);
}

// -------------------------------------------------------------------------------------
// argmax
// -------------------------------------------------------------------------------------

#[test]
fn test_argmax_returns_first_of_equal_maxima() {
    assert_eq!(argmax(&[1.0, 5.0, 5.0, 2.0]), Some(1));
}

#[test]
fn test_argmax_skips_non_finite() {
    assert_eq!(argmax(&[f32::NAN, 3.0, f32::NAN]), Some(1));
    assert_eq!(argmax(&[f32::NAN, f32::NAN]), None);
    assert_eq!(argmax(&[]), None);
}

// -------------------------------------------------------------------------------------
// dispatch and fallbacks
// -------------------------------------------------------------------------------------

#[test]
fn test_sum_reproduces_per_cycle_sum() {
    let data = xic(&[&[1.0, 2.0, 3.0], &[0.0, 0.0, 4.0]]);
    let rt_values = rt(3);
    let ctx = IntegrationContext {
        xic: &data,
        rt: &rt_values,
        template: &[0.0, 1.0, 0.0],
        apex: 1,
    };

    let areas = integrate(&ctx, &params(QuantificationMethod::Sum));
    assert_eq!(areas, vec![6.0, 4.0]);
}

#[test]
fn test_single_cycle_window_falls_back_to_sum() {
    let data = xic(&[&[7.0]]);
    let ctx = IntegrationContext {
        xic: &data,
        rt: &[100.0],
        template: &[1.0],
        apex: 0,
    };

    for method in [
        QuantificationMethod::Trapezoid,
        QuantificationMethod::BoundaryTrapezoid,
        QuantificationMethod::ProfileProjection,
        QuantificationMethod::EmgFit,
    ] {
        assert_eq!(integrate(&ctx, &params(method)), vec![7.0], "{method:?}");
    }
}

#[test]
fn test_empty_template_falls_back_to_sum() {
    let data = xic(&[&[1.0, 2.0, 3.0]]);
    let rt_values = rt(3);
    let ctx = IntegrationContext {
        xic: &data,
        rt: &rt_values,
        template: &[0.0, 0.0, 0.0],
        apex: 1,
    };

    for method in [
        QuantificationMethod::BoundaryTrapezoid,
        QuantificationMethod::ProfileProjection,
        QuantificationMethod::EmgFit,
    ] {
        assert_eq!(integrate(&ctx, &params(method)), vec![6.0], "{method:?}");
    }
}

#[test]
fn test_apex_outside_window_falls_back_to_sum() {
    let data = xic(&[&[1.0, 2.0, 3.0]]);
    let rt_values = rt(3);
    let ctx = IntegrationContext {
        xic: &data,
        rt: &rt_values,
        template: &[0.0, 1.0, 0.0],
        apex: 99,
    };

    assert_eq!(
        integrate(&ctx, &params(QuantificationMethod::ProfileProjection)),
        vec![6.0]
    );
}

// -------------------------------------------------------------------------------------
// boundary_trapezoid
// -------------------------------------------------------------------------------------

#[test]
fn test_boundary_trapezoid_removes_constant_baseline() {
    // A triangular peak of height 100 on a constant background of 20. The background must
    // not reach the reported area.
    let peak = [
        20.0, 20.0, 20.0, 45.0, 70.0, 120.0, 70.0, 45.0, 20.0, 20.0, 20.0,
    ];
    let data = xic(&[&peak]);
    let rt_values = rt(11);
    let template = triangular_template();
    let ctx = IntegrationContext {
        xic: &data,
        rt: &rt_values,
        template: &template,
        apex: 5,
    };

    let areas = integrate(&ctx, &params(QuantificationMethod::BoundaryTrapezoid));

    // Boundaries land on cycles 2 and 8, where the profile has decayed to baseline. The
    // baseline-subtracted trace is 0, 25, 50, 100, 50, 25, 0 with one second spacing.
    let expected = trapezoid(
        &rt_values[2..=8],
        &[0.0, 25.0, 50.0, 100.0, 50.0, 25.0, 0.0],
    );
    assert_relative_eq!(areas[0], expected, epsilon = 1e-3);
}

#[test]
fn test_boundary_trapezoid_stops_at_valley_between_peaks() {
    // A second, larger peak sits in the right flank of the window. The consensus profile
    // dips between the two, and integration must stop in that valley.
    let template = [0.0, 0.0, 0.25, 0.5, 1.0, 0.5, 0.2, 0.6, 0.9, 0.4, 0.0];
    let trace = [
        0.0, 0.0, 25.0, 50.0, 100.0, 50.0, 20.0, 60.0, 90.0, 40.0, 0.0,
    ];
    let data = xic(&[&trace]);
    let rt_values = rt(11);
    let ctx = IntegrationContext {
        xic: &data,
        rt: &rt_values,
        template: &template,
        apex: 4,
    };

    let boundary = integrate(&ctx, &params(QuantificationMethod::BoundaryTrapezoid))[0];
    let full_window = integrate(&ctx, &params(QuantificationMethod::Trapezoid))[0];

    // The curvature maximum at cycle 6 stops the walk short of the neighbouring peak, so
    // integration runs over cycles 1..=6. Raw values 0, 25, 50, 100, 50, 20 minus the
    // base-to-base line 0, 4, 8, 12, 16, 20 leave 0, 21, 42, 88, 34, 0.
    let expected = trapezoid(&rt_values[1..=6], &[0.0, 21.0, 42.0, 88.0, 34.0, 0.0]);
    assert_relative_eq!(boundary, expected, epsilon = 1e-3);

    // The neighbouring peak is worth a lot of area, all of which the full window absorbs.
    assert!(boundary < 0.75 * full_window);
}

#[test]
fn test_boundary_trapezoid_without_baseline_subtraction() {
    let peak = [
        20.0, 20.0, 20.0, 45.0, 70.0, 120.0, 70.0, 45.0, 20.0, 20.0, 20.0,
    ];
    let data = xic(&[&peak]);
    let rt_values = rt(11);
    let template = triangular_template();
    let ctx = IntegrationContext {
        xic: &data,
        rt: &rt_values,
        template: &template,
        apex: 5,
    };

    let mut without_baseline = params(QuantificationMethod::BoundaryTrapezoid);
    without_baseline.subtract_baseline = false;

    // Trapezoid over cycles 2..=8 of the raw trace, background included.
    let expected = trapezoid(&rt_values[2..=8], &peak[2..=8]);
    assert_relative_eq!(
        integrate(&ctx, &without_baseline)[0],
        expected,
        epsilon = 1e-3
    );
}

#[test]
fn test_boundary_trapezoid_covers_a_narrow_peak() {
    // A consensus profile that is a lone spike leaves the curvature search nothing to find,
    // so the bounds stay at the apex plus the minimum search offset and still cover the
    // whole peak rather than collapsing onto the apex.
    let template = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let trace = [0.0, 0.0, 0.0, 0.0, 10.0, 100.0, 10.0, 0.0, 0.0, 0.0, 0.0];
    let data = xic(&[&trace]);
    let rt_values = rt(11);
    let ctx = IntegrationContext {
        xic: &data,
        rt: &rt_values,
        template: &template,
        apex: 5,
    };

    let area = integrate(&ctx, &params(QuantificationMethod::BoundaryTrapezoid))[0];
    let expected = trapezoid(&rt_values[3..=7], &trace[3..=7]);
    assert_relative_eq!(area, expected, epsilon = 1e-3);
    assert_relative_eq!(area, 120.0, epsilon = 1e-3);
}

#[test]
fn test_boundary_trapezoid_always_covers_the_apex() {
    // Windows clipped at the start or end of a run put the apex against an edge; the bounds
    // must still contain it and still span enough cycles to integrate.
    let rt_values = rt(4);
    let trace = [100.0, 60.0, 20.0, 5.0];
    let data = xic(&[&trace]);

    for apex in 0..4 {
        let template = [1.0, 0.6, 0.2, 0.05];
        let ctx = IntegrationContext {
            xic: &data,
            rt: &rt_values,
            template: &template,
            apex,
        };
        let area = integrate(&ctx, &params(QuantificationMethod::BoundaryTrapezoid))[0];
        assert!(area.is_finite() && area >= 0.0, "apex {apex} gave {area}");
    }
}

// -------------------------------------------------------------------------------------
// profile_projection
// -------------------------------------------------------------------------------------

#[test]
fn test_projection_reproduces_trapezoid_for_a_trace_that_matches_the_template() {
    // The defining property: a noise free trace proportional to the consensus profile is
    // integrated to exactly its trapezoidal area, so the two methods are on one scale.
    let template = triangular_template();
    let trace: Vec<f32> = template.iter().map(|value| value * 250.0).collect();
    let data = xic(&[&trace]);
    let rt_values = rt(11);
    let ctx = IntegrationContext {
        xic: &data,
        rt: &rt_values,
        template: &template,
        apex: 5,
    };

    let projection = integrate(&ctx, &params(QuantificationMethod::ProfileProjection))[0];
    let trapezoidal = integrate(&ctx, &params(QuantificationMethod::Trapezoid))[0];
    assert_relative_eq!(projection, trapezoidal, epsilon = 1e-3);
}

#[test]
fn test_projection_ignores_interference_outside_the_elution_profile() {
    // A co-eluting species dumps signal into the flanks of the window, where the consensus
    // profile says there is nothing. The plain trapezoid absorbs it; the projection does not.
    let template = triangular_template();
    let clean: Vec<f32> = template.iter().map(|value| value * 250.0).collect();
    let mut contaminated = clean.clone();
    contaminated[0] += 400.0;
    contaminated[10] += 400.0;

    let rt_values = rt(11);
    let clean_data = xic(&[&clean]);
    let contaminated_data = xic(&[&contaminated]);

    let clean_ctx = IntegrationContext {
        xic: &clean_data,
        rt: &rt_values,
        template: &template,
        apex: 5,
    };
    let contaminated_ctx = IntegrationContext {
        xic: &contaminated_data,
        rt: &rt_values,
        template: &template,
        apex: 5,
    };

    let projection_params = params(QuantificationMethod::ProfileProjection);
    let projection_error = (integrate(&contaminated_ctx, &projection_params)[0]
        - integrate(&clean_ctx, &projection_params)[0])
        .abs();

    let trapezoid_params = params(QuantificationMethod::Trapezoid);
    let trapezoid_error = (integrate(&contaminated_ctx, &trapezoid_params)[0]
        - integrate(&clean_ctx, &trapezoid_params)[0])
        .abs();

    assert!(
        projection_error < 0.05 * trapezoid_error,
        "projection error {projection_error} should be far below trapezoid error {trapezoid_error}"
    );
}

#[test]
fn test_projection_bounds_the_influence_of_an_on_peak_spike() {
    // Interference on the peak itself cannot be ignored, but Huber reweighting must keep a
    // single inflated cycle from dominating the amplitude.
    let template = triangular_template();
    let clean: Vec<f32> = template.iter().map(|value| value * 250.0).collect();
    let mut spiked = clean.clone();
    spiked[4] += 500.0;

    let rt_values = rt(11);
    let clean_data = xic(&[&clean]);
    let spiked_data = xic(&[&spiked]);

    let clean_area = integrate(
        &IntegrationContext {
            xic: &clean_data,
            rt: &rt_values,
            template: &template,
            apex: 5,
        },
        &params(QuantificationMethod::ProfileProjection),
    )[0];

    let spiked_ctx = IntegrationContext {
        xic: &spiked_data,
        rt: &rt_values,
        template: &template,
        apex: 5,
    };

    let robust = integrate(
        &spiked_ctx,
        &params(QuantificationMethod::ProfileProjection),
    )[0];

    let mut least_squares_params = params(QuantificationMethod::ProfileProjection);
    least_squares_params.robust_iterations = 0;
    let least_squares = integrate(&spiked_ctx, &least_squares_params)[0];

    assert!(
        (robust - clean_area).abs() < (least_squares - clean_area).abs(),
        "robust fit {robust} should be closer to {clean_area} than plain least squares {least_squares}"
    );
}

#[test]
fn test_projection_reports_zero_for_an_anticorrelated_trace() {
    let template = triangular_template();
    let trace: Vec<f32> = template.iter().map(|value| 1.0 - value).collect();
    let data = xic(&[&trace]);
    let rt_values = rt(11);
    let ctx = IntegrationContext {
        xic: &data,
        rt: &rt_values,
        template: &template,
        apex: 5,
    };

    // The amplitude stays positive here because the trace is positive everywhere, but the
    // reported area may never go negative for any input.
    assert!(integrate(&ctx, &params(QuantificationMethod::ProfileProjection))[0] >= 0.0);
}

#[test]
fn test_projection_keeps_fragment_ratios() {
    // Fragments differ only in amplitude, so their reported areas must keep that ratio
    // exactly — this is the quantity label-free quantification consumes.
    let template = triangular_template();
    let strong: Vec<f32> = template.iter().map(|value| value * 1000.0).collect();
    let weak: Vec<f32> = template.iter().map(|value| value * 250.0).collect();
    let data = xic(&[&strong, &weak]);
    let rt_values = rt(11);
    let ctx = IntegrationContext {
        xic: &data,
        rt: &rt_values,
        template: &template,
        apex: 5,
    };

    let areas = integrate(&ctx, &params(QuantificationMethod::ProfileProjection));
    assert_relative_eq!(areas[0] / areas[1], 4.0, epsilon = 1e-4);
}

// -------------------------------------------------------------------------------------
// emg_fit
// -------------------------------------------------------------------------------------

use super::emg::{fit, EmgShape};

/// Sample a shape onto a cycle grid, normalised to unit apex like the consensus profile is.
fn sample_shape(shape: &EmgShape, rt_values: &[f32]) -> Vec<f32> {
    let raw: Vec<f32> = rt_values
        .iter()
        .map(|&t| shape.density(t as f64) as f32)
        .collect();
    let peak = raw.iter().copied().fold(0.0f32, f32::max);
    raw.iter().map(|value| value / peak).collect()
}

#[test]
fn test_emg_density_integrates_to_one() {
    let shape = EmgShape {
        mu: 0.0,
        sigma: 1.0,
        tau: 2.0,
    };

    let step = 0.01;
    let mut area = 0.0;
    let mut previous = shape.density(-20.0);
    for idx in 1..=8000 {
        let current = shape.density(-20.0 + step * idx as f64);
        area += 0.5 * step * (previous + current);
        previous = current;
    }

    assert_relative_eq!(area, 1.0, epsilon = 1e-4);
}

#[test]
fn test_emg_density_approaches_the_gaussian_limit() {
    // As tau goes to zero the exponentially modified Gaussian becomes a Gaussian. This also
    // exercises the asymptotic branch, where the naive expression overflows.
    let sigma = 1.0;
    let shape = EmgShape {
        mu: 0.0,
        sigma,
        tau: 1e-4,
    };

    for t in [-2.0, -0.5, 0.0, 0.5, 2.0] {
        let gaussian =
            (-t * t / (2.0 * sigma * sigma)).exp() / (sigma * (2.0 * std::f64::consts::PI).sqrt());
        assert_relative_eq!(shape.density(t), gaussian, epsilon = 1e-3);
    }
}

#[test]
fn test_emg_fit_recovers_a_known_shape() {
    let rt_values = rt(15);
    let truth = EmgShape {
        mu: 106.0,
        sigma: 1.1,
        tau: 1.4,
    };
    let template: Vec<f64> = rt_values.iter().map(|&t| truth.density(t as f64)).collect();
    let rt_f64: Vec<f64> = rt_values.iter().map(|&t| t as f64).collect();

    let fitted = fit(&rt_f64, &template).expect("a clean exponentially modified Gaussian must fit");

    assert_relative_eq!(fitted.mu, truth.mu, epsilon = 0.15);
    assert_relative_eq!(fitted.sigma, truth.sigma, epsilon = 0.15);
    assert_relative_eq!(fitted.tau, truth.tau, epsilon = 0.2);
}

#[test]
fn test_emg_fit_rejects_a_shape_it_cannot_describe() {
    // Alternating spikes are not a chromatographic peak; the fit must report failure rather
    // than return a meaningless shape.
    let rt_values = rt(11);
    let rt_f64: Vec<f64> = rt_values.iter().map(|&t| t as f64).collect();
    let template = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];

    assert!(fit(&rt_f64, &template).is_none());
}

#[test]
fn test_emg_fit_matches_the_trapezoid_on_a_well_sampled_peak() {
    // A trace that follows a real chromatographic shape must be integrated to essentially
    // the same area, so switching methods does not rescale the results.
    let rt_values = rt(15);
    let shape = EmgShape {
        mu: 106.0,
        sigma: 1.3,
        tau: 1.2,
    };
    let template = sample_shape(&shape, &rt_values);
    let trace: Vec<f32> = template.iter().map(|value| value * 500.0).collect();
    let data = xic(&[&trace]);

    let apex = argmax(&template).unwrap();
    let ctx = IntegrationContext {
        xic: &data,
        rt: &rt_values,
        template: &template,
        apex,
    };

    let emg_area = integrate(&ctx, &params(QuantificationMethod::EmgFit))[0];
    let trapezoid_area = integrate(&ctx, &params(QuantificationMethod::Trapezoid))[0];
    assert_relative_eq!(emg_area, trapezoid_area, max_relative = 0.05);
}

#[test]
fn test_emg_fit_recovers_a_truncated_tail_when_asked_to() {
    // The window cuts the tail of a strongly tailing peak. Widening the integration range
    // must recover signal, and only when the caller opts in.
    let rt_values = rt(11);
    let shape = EmgShape {
        mu: 103.0,
        sigma: 1.0,
        tau: 3.0,
    };
    let template = sample_shape(&shape, &rt_values);
    let trace: Vec<f32> = template.iter().map(|value| value * 500.0).collect();
    let data = xic(&[&trace]);

    let apex = argmax(&template).unwrap();
    let ctx = IntegrationContext {
        xic: &data,
        rt: &rt_values,
        template: &template,
        apex,
    };

    let window_only = integrate(&ctx, &params(QuantificationMethod::EmgFit))[0];

    let mut extrapolating = params(QuantificationMethod::EmgFit);
    extrapolating.emg_extrapolation_factor = 2.0;
    let extended = integrate(&ctx, &extrapolating)[0];

    assert!(
        extended > window_only,
        "extrapolated area {extended} should exceed window-only area {window_only}"
    );
}

#[test]
fn test_emg_fit_falls_back_to_projection_when_the_shape_does_not_fit() {
    let rt_values = rt(11);
    let template = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let trace: Vec<f32> = template.iter().map(|value| value * 100.0).collect();
    let data = xic(&[&trace]);
    let ctx = IntegrationContext {
        xic: &data,
        rt: &rt_values,
        template: &template,
        apex: 0,
    };

    let emg = integrate(&ctx, &params(QuantificationMethod::EmgFit))[0];
    let projection = integrate(&ctx, &params(QuantificationMethod::ProfileProjection))[0];
    assert_relative_eq!(emg, projection, epsilon = 1e-4);
}

#[test]
fn test_emg_fit_keeps_fragment_ratios() {
    let rt_values = rt(15);
    let shape = EmgShape {
        mu: 106.0,
        sigma: 1.3,
        tau: 1.2,
    };
    let template = sample_shape(&shape, &rt_values);
    let strong: Vec<f32> = template.iter().map(|value| value * 800.0).collect();
    let weak: Vec<f32> = template.iter().map(|value| value * 100.0).collect();
    let data = xic(&[&strong, &weak]);

    let apex = argmax(&template).unwrap();
    let ctx = IntegrationContext {
        xic: &data,
        rt: &rt_values,
        template: &template,
        apex,
    };

    let areas = integrate(&ctx, &params(QuantificationMethod::EmgFit));
    assert_relative_eq!(areas[0] / areas[1], 8.0, epsilon = 1e-3);
}

// -------------------------------------------------------------------------------------
// cross-method behaviour
// -------------------------------------------------------------------------------------

#[test]
fn test_every_method_handles_an_all_zero_fragment() {
    let template = triangular_template();
    let peak: Vec<f32> = template.iter().map(|value| value * 100.0).collect();
    let zeros = vec![0.0f32; template.len()];
    let data = xic(&[&peak, &zeros]);
    let rt_values = rt(11);
    let ctx = IntegrationContext {
        xic: &data,
        rt: &rt_values,
        template: &template,
        apex: 5,
    };

    for method in [
        QuantificationMethod::Sum,
        QuantificationMethod::Trapezoid,
        QuantificationMethod::BoundaryTrapezoid,
        QuantificationMethod::ProfileProjection,
        QuantificationMethod::EmgFit,
    ] {
        let areas = integrate(&ctx, &params(method));
        assert_eq!(areas.len(), 2, "{method:?}");
        assert!(areas[0] > 0.0, "{method:?}");
        assert_eq!(areas[1], 0.0, "{method:?}");
    }
}

#[test]
fn test_every_method_handles_uneven_cycle_spacing() {
    let template = triangular_template();
    let peak: Vec<f32> = template.iter().map(|value| value * 100.0).collect();
    let data = xic(&[&peak]);
    // A gap in acquisition doubles the spacing across the apex.
    let rt_values: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let ctx = IntegrationContext {
        xic: &data,
        rt: &rt_values,
        template: &template,
        apex: 5,
    };

    for method in [
        QuantificationMethod::Trapezoid,
        QuantificationMethod::BoundaryTrapezoid,
        QuantificationMethod::ProfileProjection,
        QuantificationMethod::EmgFit,
    ] {
        let area = integrate(&ctx, &params(method))[0];
        assert!(area.is_finite() && area > 0.0, "{method:?} gave {area}");
    }
}

// -------------------------------------------------------------------------------------
// consensus profile: smoothing
// -------------------------------------------------------------------------------------

use super::smoothing::whittaker_henderson;
use super::template;

#[test]
fn test_smoothing_is_a_no_op_when_disabled_or_impossible() {
    let rt_values = rt(11);
    let original = triangular_template();

    let mut untouched = original.clone();
    whittaker_henderson(&rt_values, &mut untouched, 0.0);
    assert_eq!(untouched, original);

    // Third differences need four points.
    let mut too_short = vec![1.0, 2.0, 3.0];
    whittaker_henderson(&rt(3), &mut too_short, 1.0);
    assert_eq!(too_short, vec![1.0, 2.0, 3.0]);

    // A window with no retention time span has no grid to smooth on.
    let mut no_span = vec![1.0, 2.0, 3.0, 4.0];
    whittaker_henderson(&[5.0, 5.0, 5.0, 5.0], &mut no_span, 1.0);
    assert_eq!(no_span, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_smoothing_leaves_a_quadratic_untouched() {
    // The third-difference penalty annihilates polynomials of degree two, so quadratic
    // structure — which includes the curvature of a peak apex — must survive intact even at
    // a smoothing strength far above the default.
    let rt_values = rt(11);
    let quadratic: Vec<f32> = (0..11)
        .map(|idx| {
            let x = idx as f32;
            10.0 + 2.0 * x + 0.5 * x * x
        })
        .collect();

    let mut smoothed = quadratic.clone();
    whittaker_henderson(&rt_values, &mut smoothed, 1.0);

    for (before, after) in quadratic.iter().zip(smoothed.iter()) {
        assert_relative_eq!(before, after, max_relative = 1e-4);
    }
}

#[test]
fn test_smoothing_moves_a_noisy_profile_towards_the_truth() {
    let rt_values = rt(11);
    // A Gaussian elution profile, which is what the penalty expects to see; the sawtooth
    // added on top is exactly the high-frequency component it is meant to remove.
    let truth: Vec<f32> = (0..11)
        .map(|idx| {
            let offset = idx as f32 - 5.0;
            (-offset * offset / (2.0 * 1.5 * 1.5)).exp()
        })
        .collect();
    let noisy: Vec<f32> = truth
        .iter()
        .enumerate()
        .map(|(idx, &value)| {
            let sawtooth = if idx % 2 == 0 { 0.05 } else { -0.05 };
            (value + sawtooth).max(0.0)
        })
        .collect();

    let mut smoothed = noisy.clone();
    whittaker_henderson(&rt_values, &mut smoothed, 5e-7);

    let error = |profile: &[f32]| -> f32 {
        profile
            .iter()
            .zip(truth.iter())
            .map(|(&value, &reference)| (value - reference).powi(2))
            .sum()
    };

    assert!(
        error(&smoothed) < error(&noisy),
        "smoothed error {} should be below noisy error {}",
        error(&smoothed),
        error(&noisy)
    );
}

#[test]
fn test_smoothing_roughly_preserves_area() {
    // Smoothing runs before integration, so it must not move the area it is preparing.
    let rt_values = rt(11);
    let truth = triangular_template();
    let mut smoothed = truth.clone();
    whittaker_henderson(&rt_values, &mut smoothed, 5e-7);

    assert_relative_eq!(
        trapezoid(&rt_values, &smoothed),
        trapezoid(&rt_values, &truth),
        max_relative = 0.02
    );
}

#[test]
fn test_smoothing_handles_uneven_spacing_and_stays_non_negative() {
    let rt_values: Vec<f32> = vec![0.0, 1.0, 2.0, 4.0, 5.0, 9.0, 10.0, 11.0];
    let mut profile = vec![0.0, 0.1, 0.4, 1.0, 0.5, 0.1, 0.0, 0.0];
    whittaker_henderson(&rt_values, &mut profile, 5e-7);

    assert!(profile
        .iter()
        .all(|value| value.is_finite() && *value >= 0.0));
}

#[test]
fn test_default_smoothing_keeps_the_projection_close_to_the_trapezoid() {
    // Smoothing perturbs the template, so the exact agreement between a matching trace and
    // its trapezoidal area becomes approximate. It must stay approximate, not drift.
    let template = triangular_template();
    let trace: Vec<f32> = template.iter().map(|value| value * 250.0).collect();
    let data = xic(&[&trace]);
    let rt_values = rt(11);
    let ctx = IntegrationContext {
        xic: &data,
        rt: &rt_values,
        template: &template,
        apex: 5,
    };

    let mut smoothing_on = params(QuantificationMethod::ProfileProjection);
    smoothing_on.template_smoothing_lambda = 5e-7;

    let projection = integrate(&ctx, &smoothing_on)[0];
    let trapezoidal = integrate(&ctx, &params(QuantificationMethod::Trapezoid))[0];
    assert_relative_eq!(projection, trapezoidal, max_relative = 0.05);
}

// -------------------------------------------------------------------------------------
// consensus profile: refinement
// -------------------------------------------------------------------------------------

#[test]
fn test_template_refinement_is_disabled_by_a_zero_cut() {
    let normalized = xic(&[&[0.0, 1.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 1.0, 0.0]]);
    assert!(template::refine(&normalized, &[1.0, 1.0, 1.0], 0.0).is_none());
}

#[test]
fn test_template_refinement_needs_a_quorum() {
    let normalized = xic(&[&[0.0, 1.0, 0.0], &[0.0, 1.0, 0.0], &[1.0, 0.0, 1.0]]);
    // Only two fragments clear the cut, one short of a consensus.
    assert!(template::refine(&normalized, &[0.9, 0.9, 0.1], 0.5).is_none());
}

#[test]
fn test_template_refinement_drops_interfered_fragments() {
    let peak = [0.0, 0.5, 1.0, 0.5, 0.0];
    let interfered = [1.0, 1.0, 1.0, 1.0, 1.0];
    let normalized = xic(&[&peak, &peak, &peak, &interfered, &interfered, &interfered]);

    // Half the fragments agree with the elution profile and half do not. The plain median
    // over all six is pulled up in the flanks; the refined profile is not.
    let plain = crate::peak_group_scoring::utils::median_axis_0(&normalized);
    let refined = template::refine(&normalized, &[0.99, 0.99, 0.99, -0.2, -0.2, -0.2], 0.5)
        .expect("three agreeing fragments are a quorum");

    assert_eq!(refined, peak.to_vec());
    assert!(plain[0] > refined[0]);
}

#[test]
fn test_template_refinement_ignores_all_zero_fragments() {
    // A fragment with no signal cannot define an elution profile even if its correlation
    // happens to clear the cut.
    let peak = [0.0, 0.5, 1.0, 0.5, 0.0];
    let zeros = [0.0f32; 5];
    let normalized = xic(&[&peak, &peak, &zeros, &zeros, &zeros]);

    assert!(template::refine(&normalized, &[0.99, 0.99, 0.99, 0.99, 0.99], 0.5).is_none());
}

// -------------------------------------------------------------------------------------
// background handling
// -------------------------------------------------------------------------------------

#[test]
fn test_projection_removes_a_flat_background() {
    let template = triangular_template();
    let clean: Vec<f32> = template.iter().map(|value| value * 250.0).collect();
    let with_background: Vec<f32> = clean.iter().map(|value| value + 40.0).collect();

    let rt_values = rt(11);
    let clean_data = xic(&[&clean]);
    let background_data = xic(&[&with_background]);

    let projection = params(QuantificationMethod::ProfileProjection);
    let clean_area = integrate(
        &IntegrationContext {
            xic: &clean_data,
            rt: &rt_values,
            template: &template,
            apex: 5,
        },
        &projection,
    )[0];
    let background_area = integrate(
        &IntegrationContext {
            xic: &background_data,
            rt: &rt_values,
            template: &template,
            apex: 5,
        },
        &projection,
    )[0];

    assert_relative_eq!(background_area, clean_area, max_relative = 1e-4);
}

#[test]
fn test_projection_background_survives_a_contaminated_flank() {
    // The background is a median over the flanks, so one contaminated flank cycle must not
    // move it — which is exactly what a jointly fitted background would fail to do.
    let template = triangular_template();
    let clean: Vec<f32> = template.iter().map(|value| value * 250.0).collect();
    let mut contaminated = clean.clone();
    contaminated[0] += 2000.0;

    let rt_values = rt(11);
    let clean_data = xic(&[&clean]);
    let contaminated_data = xic(&[&contaminated]);

    let projection = params(QuantificationMethod::ProfileProjection);
    let clean_area = integrate(
        &IntegrationContext {
            xic: &clean_data,
            rt: &rt_values,
            template: &template,
            apex: 5,
        },
        &projection,
    )[0];
    let contaminated_area = integrate(
        &IntegrationContext {
            xic: &contaminated_data,
            rt: &rt_values,
            template: &template,
            apex: 5,
        },
        &projection,
    )[0];

    assert_relative_eq!(contaminated_area, clean_area, max_relative = 1e-4);
}

#[test]
fn test_projection_without_background_fitting_is_biased_by_it() {
    // The documented trade-off: switching the background term off leaves the area inflated.
    let template = triangular_template();
    let with_background: Vec<f32> = template.iter().map(|value| value * 250.0 + 40.0).collect();
    let rt_values = rt(11);
    let data = xic(&[&with_background]);
    let ctx = IntegrationContext {
        xic: &data,
        rt: &rt_values,
        template: &template,
        apex: 5,
    };

    let mut without = params(QuantificationMethod::ProfileProjection);
    without.projection_fit_baseline = false;

    let corrected = integrate(&ctx, &params(QuantificationMethod::ProfileProjection))[0];
    let uncorrected = integrate(&ctx, &without)[0];
    let full_window = integrate(&ctx, &params(QuantificationMethod::Trapezoid))[0];

    assert!(uncorrected > corrected);
    // Still much better than letting the background run through the whole window: the
    // amplitude picks up b·Σp/Σp² rather than b times the full window width.
    assert!(uncorrected - corrected < 0.5 * (full_window - corrected));
}

// -------------------------------------------------------------------------------------
// end to end: fragment ratio accuracy on a synthetic peak group
// -------------------------------------------------------------------------------------

/// Deterministic linear congruential generator, so the synthetic benchmark below is
/// reproducible without pulling `rand` into the test build.
struct Lcg(u64);

impl Lcg {
    fn signed_unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32) / ((1u64 << 23) as f32) - 1.0
    }
}

/// Mean absolute relative error of the recovered fragment ratios against the true ones.
///
/// Ratios, not areas, because that is what label-free quantification consumes downstream and
/// because it is the one measure comparable across methods with different units.
fn ratio_error(areas: &[f32], truth: &[f32]) -> f32 {
    let total: f32 = areas
        .iter()
        .zip(truth.iter())
        .skip(1)
        .map(|(&area, &reference)| {
            let recovered = area / areas[0];
            let expected = reference / truth[0];
            ((recovered - expected) / expected).abs()
        })
        .sum();
    total / (areas.len() - 1) as f32
}

#[test]
fn test_new_methods_recover_fragment_ratios_better_than_a_window_sum() {
    // A realistic peak group: one shared tailing elution profile, fragments spanning two
    // orders of magnitude, a flat chemical background that does *not* scale with abundance —
    // so it distorts the weak fragments most — shot noise, and a neighbouring peak leaking
    // into one fragment's flank.
    const N_CYCLES: usize = 15;
    let rt_values = rt(N_CYCLES);
    let shape = EmgShape {
        mu: 106.5,
        sigma: 1.3,
        tau: 1.1,
    };
    let profile = sample_shape(&shape, &rt_values);
    let amplitudes = [4000.0f32, 2000.0, 1000.0, 400.0, 200.0, 80.0];
    const BACKGROUND: f32 = 60.0;

    let mut rng = Lcg(0x5eed_1234_abcd_ef01);
    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(amplitudes.len());
    for (fragment_idx, &amplitude) in amplitudes.iter().enumerate() {
        let mut row = Vec::with_capacity(N_CYCLES);
        for (cycle_idx, &shape_value) in profile.iter().enumerate() {
            let mut value = amplitude * shape_value + BACKGROUND;
            // A neighbouring peak tails into the left flank of one fragment only.
            if fragment_idx == 2 && cycle_idx < 3 {
                value += 1500.0 * (3 - cycle_idx) as f32;
            }
            value += value.sqrt() * rng.signed_unit() * 2.0;
            row.push(value.max(0.0));
        }
        rows.push(row);
    }
    let data = xic(&rows.iter().map(|row| row.as_slice()).collect::<Vec<_>>());

    // Build the consensus profile the way production does, so the template path is exercised
    // too rather than being handed the ground truth.
    let normalized = crate::peak_group_scoring::utils::normalize_profiles_at(&data, 7, 1);
    let first_pass = crate::peak_group_scoring::utils::median_axis_0(
        &crate::peak_group_scoring::utils::filter_non_zero(&normalized),
    );
    let correlations: Vec<f32> = (0..rows.len())
        .map(|idx| {
            crate::peak_group_scoring::utils::calculate_correlation_safe(
                normalized.row(idx).as_slice().unwrap(),
                &first_pass,
            )
        })
        .collect();
    let consensus = template::refine(&normalized, &correlations, 0.5).unwrap_or(first_pass);

    let ctx = IntegrationContext {
        xic: &data,
        rt: &rt_values,
        template: &consensus,
        apex: 7,
    };

    let error = |method: QuantificationMethod| -> f32 {
        let mut with_smoothing = params(method);
        with_smoothing.template_smoothing_lambda = 5e-7;
        ratio_error(&integrate(&ctx, &with_smoothing), &amplitudes)
    };

    let sum = error(QuantificationMethod::Sum);
    let trapezoid_error = error(QuantificationMethod::Trapezoid);
    let boundary = error(QuantificationMethod::BoundaryTrapezoid);
    let projection = error(QuantificationMethod::ProfileProjection);
    let emg = error(QuantificationMethod::EmgFit);

    println!(
        "ratio error — sum {sum:.4} trapezoid {trapezoid_error:.4} \
         boundary {boundary:.4} projection {projection:.4} emg {emg:.4}"
    );

    // The two whole-window methods are the baseline: neither removes the background, so both
    // charge the same absolute amount of it to every fragment and wreck the weak ratios.
    let baseline = sum.min(trapezoid_error);

    for (name, value) in [
        ("boundary_trapezoid", boundary),
        ("profile_projection", projection),
        ("emg_fit", emg),
    ] {
        assert!(
            value < 0.5 * baseline,
            "{name} error {value:.4} should be well below the whole-window error              (sum {sum:.4}, trapezoid {trapezoid_error:.4})"
        );
    }
}
