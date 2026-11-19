use rayon::prelude::*;

use super::dropout::{Dropout, CURVE_ROUNDS};
use super::noise::{FitParameters, NoiseModel, VarianceCurve};

const DEFAULT_MAX_SWEEPS: usize = 60;
const DEFAULT_MIN_IONS: usize = 2;
const DEFAULT_TOLERANCE: f64 = 1e-4;

/// `tau` and `lambda` are derived from the data unless set.
pub struct FitConfig {
    pub tau: Option<f64>,
    pub lambda: Option<f64>,
    pub max_sweeps: usize,
    pub min_ions: usize,
    pub tolerance: f64,
    /// Missing fragments of a quantified run enter the fit as censored observations.
    pub censored: bool,
    /// gamma of the measurement variance factor `exp(gamma * (1 - sqrt(C)))`; 0 turns it off.
    pub quality_exponent: f64,
}

impl FitConfig {
    pub const DEFAULT: Self = Self {
        tau: None,
        lambda: None,
        max_sweeps: DEFAULT_MAX_SWEEPS,
        min_ions: DEFAULT_MIN_IONS,
        tolerance: DEFAULT_TOLERANCE,
        censored: false,
        quality_exponent: 0.0,
    };
}

/// Run quantities, run-major `(n_runs, n_precursors)`, NaN where too few fragments were
/// observed, and the dropout curves when censoring.
pub struct FitResult {
    pub quantity: Vec<f64>,
    pub parameters: FitParameters,
    pub dropout: Option<Dropout>,
}

pub fn validate_shapes(
    intensity_shape: (usize, usize),
    n_intensity_library: usize,
    precursor: &[usize],
    n_precursors: usize,
) -> Result<(), String> {
    let (_, n_ions) = intensity_shape;
    if n_intensity_library != n_ions || precursor.len() != n_ions {
        return Err(format!(
            "intensity has {n_ions} ions, intensity_library {n_intensity_library} and precursor {}",
            precursor.len()
        ));
    }
    if let Some(&index) = precursor.iter().find(|&&index| index >= n_precursors) {
        return Err(format!(
            "precursor index {index} out of range for {n_precursors} precursors"
        ));
    }
    Ok(())
}

pub fn validate_correlation(correlation: &[f32]) -> Result<(), String> {
    match correlation.iter().find(|c| !(0.0..=1.0).contains(*c)) {
        Some(c) => Err(format!("correlation must lie in [0, 1], found {c}")),
        None => Ok(()),
    }
}

/// `correlation` is run-major like `intensity`.
pub fn fit(
    intensity: &[f32],
    intensity_library: &[f32],
    precursor: &[usize],
    correlation: &[f32],
    n_runs: usize,
    n_precursors: usize,
    config: &FitConfig,
) -> FitResult {
    let groups = build_groups(
        intensity,
        intensity_library,
        precursor,
        n_runs,
        n_precursors,
    );
    let initial: Vec<Vec<f64>> = groups.par_iter().map(initial_quantities).collect();
    let noise = NoiseModel::estimate(&groups, &initial, config);
    let variances: Vec<Vec<f64>> = groups
        .par_iter()
        .zip(&initial)
        .map(|(group, quantity)| {
            measurement_variances(group, quantity, &noise.variance, correlation, config)
        })
        .collect();
    let fit = Fit {
        groups: &groups,
        initial: &initial,
        variances: &variances,
        noise: &noise,
        config,
    };

    let mut states: Vec<State> = (0..groups.len())
        .into_par_iter()
        .map(|index| fit.group(index, None))
        .collect();
    let mut dropout = None;
    if config.censored {
        for _ in 0..CURVE_ROUNDS {
            let curves = Dropout::fit(&groups, &states, n_runs);
            states = fit.refit_censored(&curves, states);
            dropout = Some(curves);
        }
    }
    FitResult {
        quantity: run_major(&groups, &states, n_runs, n_precursors, config.min_ions),
        parameters: noise.parameters,
        dropout,
    }
}

/// One precursor: its fragments with a library intensity (their indices in the input), their
/// library shares and their measurements as `log2(I / s)`, ion-major, NaN where missing.
pub struct Group {
    ions: Vec<usize>,
    values: Vec<f64>,
    pub share: Vec<f64>,
    pub log_share: Vec<f64>,
    pub n_ions: usize,
    pub n_runs: usize,
    n_total_ions: usize,
}

impl Group {
    pub fn index(&self, ion: usize, run: usize) -> usize {
        ion * self.n_runs + run
    }

    pub fn value(&self, ion: usize, run: usize) -> f64 {
        self.values[self.index(ion, run)]
    }

    pub fn observed(&self, ion: usize, run: usize) -> bool {
        self.value(ion, run).is_finite()
    }

    pub fn n_observed(&self, run: usize) -> usize {
        (0..self.n_ions)
            .filter(|&ion| self.observed(ion, run))
            .count()
    }

    pub fn deviation(&self, ion: usize, run: usize, quantity: &[f64]) -> Option<f64> {
        let y = self.value(ion, run);
        let q = quantity[run];
        (y.is_finite() && q.is_finite()).then_some(y - q)
    }

    fn correlation(&self, correlation: &[f32], ion: usize, run: usize) -> f64 {
        correlation[run * self.n_total_ions + self.ions[ion]] as f64
    }

    /// A missing measurement in a run with an observed one: only these carry censoring
    /// information.
    fn has_censored(&self) -> bool {
        (0..self.n_runs).any(|run| {
            let n_observed = self.n_observed(run);
            n_observed > 0 && n_observed < self.n_ions
        })
    }
}

pub fn build_groups(
    intensity: &[f32],
    intensity_library: &[f32],
    precursor: &[usize],
    n_runs: usize,
    n_precursors: usize,
) -> Vec<Group> {
    let n_total_ions = intensity_library.len();
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); n_precursors];
    for (ion, &index) in precursor.iter().enumerate() {
        let l = intensity_library[ion] as f64;
        if l.is_finite() && l > 0.0 {
            members[index].push(ion);
        }
    }
    members
        .into_par_iter()
        .map(|ions| {
            let intensity_library_sum: f64 =
                ions.iter().map(|&ion| intensity_library[ion] as f64).sum();
            let share: Vec<f64> = ions
                .iter()
                .map(|&ion| intensity_library[ion] as f64 / intensity_library_sum)
                .collect();
            let mut values = Vec::with_capacity(ions.len() * n_runs);
            for (&ion, &s) in ions.iter().zip(&share) {
                for run in 0..n_runs {
                    let i = intensity[run * n_total_ions + ion] as f64;
                    values.push(if i > 0.0 && i.is_finite() {
                        (i / s).log2()
                    } else {
                        f64::NAN
                    });
                }
            }
            Group {
                n_ions: ions.len(),
                log_share: share.iter().map(|s| s.log2()).collect(),
                ions,
                values,
                share,
                n_runs,
                n_total_ions,
            }
        })
        .collect()
}

/// Step 1 of the procedure: the library-share weighted mean of the observed fragments, without
/// fragment offsets; NaN for a run without any.
pub fn initial_quantities(group: &Group) -> Vec<f64> {
    (0..group.n_runs)
        .map(|run| {
            let mut numerator = 0.0;
            let mut denominator = 0.0;
            for ion in 0..group.n_ions {
                if group.observed(ion, run) {
                    numerator += group.share[ion] * group.value(ion, run);
                    denominator += group.share[ion];
                }
            }
            if denominator > 0.0 {
                numerator / denominator
            } else {
                f64::NAN
            }
        })
        .collect()
}

/// The variance of every measurement, fixed at the share-weighted first pass so that the
/// weights cannot chase the fit, times its quality factor.
fn measurement_variances(
    group: &Group,
    quantity: &[f64],
    curve: &VarianceCurve,
    correlation: &[f32],
    config: &FitConfig,
) -> Vec<f64> {
    let mut variance = Vec::with_capacity(group.n_ions * group.n_runs);
    for ion in 0..group.n_ions {
        for (run, &q) in quantity.iter().enumerate() {
            let quality =
                config.quality_exponent * (1.0 - group.correlation(correlation, ion, run).sqrt());
            variance.push(curve.variance(group.log_share[ion] + q) * quality.exp());
        }
    }
    variance
}

/// Fragment offsets and run quantities of one precursor.
pub struct State {
    pub quantity: Vec<f64>,
    pub offset: Vec<f64>,
}

/// Step 3 of the procedure, with what is shared by all precursors.
struct Fit<'a> {
    groups: &'a [Group],
    initial: &'a [Vec<f64>],
    variances: &'a [Vec<f64>],
    noise: &'a NoiseModel,
    config: &'a FitConfig,
}

impl Fit<'_> {
    /// Alternates weights, fragment offsets and run quantities of one precursor from the
    /// share-weighted start until no quantity moves by `tolerance`, at most `max_sweeps`.
    fn group(&self, index: usize, dropout: Option<&Dropout>) -> State {
        let group = &self.groups[index];
        let FitParameters { sigma, tau, lambda } = self.noise.parameters;
        let mut sweep = Sweep {
            group,
            variance: &self.variances[index],
            dropout,
            loss: (sigma / tau).powi(2),
            // lambda counts observations of variance sigma ** 2
            prior_weight: lambda / (sigma * sigma),
            weights: vec![0.0; group.n_ions * group.n_runs],
            values: vec![0.0; group.n_ions * group.n_runs],
        };
        let mut state = State {
            quantity: self.initial[index].clone(),
            offset: vec![0.0; group.n_ions],
        };
        for _ in 0..self.config.max_sweeps {
            if sweep.run(&mut state) < self.config.tolerance {
                break;
            }
        }
        state
    }

    /// The fit with the censored pseudo-observations of `dropout`. Precursors without a
    /// missing measurement in a quantified run have none and keep their previous fit.
    fn refit_censored(&self, dropout: &Dropout, previous: Vec<State>) -> Vec<State> {
        previous
            .into_par_iter()
            .enumerate()
            .map(|(index, state)| {
                if self.groups[index].has_censored() {
                    self.group(index, Some(dropout))
                } else {
                    state
                }
            })
            .collect()
    }
}

/// The buffers and constants of the sweeps of one precursor.
struct Sweep<'a> {
    group: &'a Group,
    variance: &'a [f64],
    dropout: Option<&'a Dropout>,
    loss: f64,
    prior_weight: f64,
    weights: Vec<f64>,
    values: Vec<f64>,
}

impl Sweep<'_> {
    /// One sweep: weights, offsets, quantities. Returns the largest change of a quantity.
    fn run(&mut self, state: &mut State) -> f64 {
        self.update_weights(state);
        self.update_offsets(state);
        self.update_quantities(state)
    }

    /// Observed measurements get their inverse variance, downweighted by the Cauchy loss of
    /// their residual; missing ones of a quantified run a censored pseudo-observation of the
    /// run's dropout curve, if any.
    fn update_weights(&mut self, state: &State) {
        let group = self.group;
        for ion in 0..group.n_ions {
            for run in 0..group.n_runs {
                let index = group.index(ion, run);
                let q = state.quantity[run];
                let y = group.value(ion, run);
                (self.values[index], self.weights[index]) = if y.is_finite() {
                    let e = y - q - state.offset[ion];
                    (y, 1.0 / (self.variance[index] + self.loss * e * e))
                } else {
                    match self
                        .dropout
                        .and_then(|dropout| dropout.curves[run].as_ref())
                    {
                        Some(curve) if q.is_finite() => {
                            let log_share = group.log_share[ion];
                            curve.pseudo_observation(q + state.offset[ion] + log_share, log_share)
                        }
                        _ => (0.0, 0.0),
                    }
                };
            }
        }
    }

    fn update_offsets(&self, state: &mut State) {
        let group = self.group;
        for ion in 0..group.n_ions {
            let mut numerator = 0.0;
            let mut denominator = 0.0;
            for run in 0..group.n_runs {
                let index = group.index(ion, run);
                let w = self.weights[index];
                if w > 0.0 {
                    numerator += w * (self.values[index] - state.quantity[run]);
                    denominator += w;
                }
            }
            state.offset[ion] = numerator / (denominator + self.prior_weight);
        }
    }

    fn update_quantities(&self, state: &mut State) -> f64 {
        let group = self.group;
        let mut change: f64 = 0.0;
        for run in 0..group.n_runs {
            if !state.quantity[run].is_finite() {
                continue;
            }
            let mut numerator = 0.0;
            let mut denominator = 0.0;
            for ion in 0..group.n_ions {
                let index = group.index(ion, run);
                let w = self.weights[index];
                if w > 0.0 {
                    numerator += w * (self.values[index] - state.offset[ion]);
                    denominator += w;
                }
            }
            let quantity = numerator / denominator;
            change = change.max((quantity - state.quantity[run]).abs());
            state.quantity[run] = quantity;
        }
        change
    }
}

fn run_major(
    groups: &[Group],
    states: &[State],
    n_runs: usize,
    n_precursors: usize,
    min_ions: usize,
) -> Vec<f64> {
    let mut quantity = vec![f64::NAN; n_runs * n_precursors];
    for (index, (group, state)) in groups.iter().zip(states).enumerate() {
        for (run, &q) in state.quantity.iter().enumerate() {
            if group.n_observed(run) >= min_ions {
                quantity[run * n_precursors + index] = q;
            }
        }
    }
    quantity
}
