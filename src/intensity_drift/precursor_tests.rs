use super::precursor::{
    correct_precursor_drift, correct_precursor_drift_in_rounds, PrecursorDriftParameters,
    PrecursorDriftSummary, ROW_LEVEL_ROUNDS,
};

const N_PRECURSORS: usize = 30_000;
const GRADIENT_SECONDS: f64 = 1500.0;
const MASS_LOW: f64 = 800.0;
const MASS_HIGH: f64 = 4000.0;
const LEVEL_LOW_LOG2: f64 = 10.0;
const LEVEL_SPREAD_LOG2: f64 = 10.0;
/// Noise of every cell, in log2, as between replicate runs of a low-input sample.
const NOISE_SD_LOG2: f64 = 0.4;
/// Drift between the two conditions at the end of the gradient for the heaviest precursors.
const DRIFT_LOG2: f64 = 0.8;
/// Minimum share of the drift removed, averaged over `N_SEEDS` draws of the noise. With
/// `points_per_cell` 125 the half-sample mode of a cell scatters by ~0.12 log2 at this noise, so
/// a single draw removes 0.83 to 1.06 of the drift; the average measures the bias.
const MIN_REMOVED_SHARE: f64 = 0.9;
const N_SEEDS: u64 = 8;
/// Maximum root mean square difference between the applied correction and the drift, in log2.
const MAX_CORRECTION_ERROR_LOG2: f64 = 0.05;
/// Runs of the partial-reference tests: three per condition, as in a small study.
const N_PARTIAL_RUNS: usize = 6;
/// Quantified precursors of the near-blank run, fewer than `min_reference_per_run`.
const N_BLANK_RUN_PRECURSORS: usize = 50;
const MISSING_SHARE: f64 = 0.4;
/// Draws of the tests with missing cells; their shares scatter more than those of complete
/// tables.
const N_MISSING_SEEDS: u64 = 16;
/// Maximum shortfall of the mean removed share with missing cells against the complete tables of
/// the same draws. Over 48 draws the shortfall is 0.017 (standard error 0.008), 0.029 (0.010)
/// against complete tables with as many reference precursors per run: the half-sample mode
/// under-corrects a little on the deviations of incomplete rows, a cell mean does not (0.007).
const MAX_MISSING_SHORTFALL: f64 = 0.05;
/// Minimum gain of the alternating estimate over a single round with missing cells. A single
/// round measures the deviations of incomplete rows from biased row levels: 0.83 against 0.92
/// over 48 draws.
const MIN_ROUNDS_GAIN: f64 = 0.04;
/// The rounds of the alternating estimate never stop early.
const NEVER_STOP: f64 = -1.0;
/// Agreement of between-run log2 ratios of two estimates that must be equal.
const RATIO_TOLERANCE_LOG2: f64 = 1e-9;

/// SplitMix64 with Box-Muller normal draws.
struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform in `(0, 1)`.
    fn uniform(&mut self) -> f64 {
        ((self.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        (-2.0 * self.uniform().ln()).sqrt() * (std::f64::consts::TAU * self.uniform()).cos()
    }
}

/// Run-major precursor tables of `n_runs` runs. The even runs drift up along the RT, the odd
/// runs down, in proportion to the relative mass of the precursor: a drift confounded with the
/// condition that one factor per run cannot remove.
struct Tables {
    n_runs: usize,
    n_precursors: usize,
    intensity: Vec<f64>,
    rt: Vec<f64>,
    mass: Vec<f64>,
    /// Log2 drift of every cell.
    drift: Vec<f64>,
}

impl Tables {
    fn new(n_runs: usize, n_precursors: usize, seed: u64) -> Self {
        let mut random = Random(seed);
        let rt: Vec<f64> = (0..n_precursors)
            .map(|_| random.uniform() * GRADIENT_SECONDS)
            .collect();
        let mass: Vec<f64> = (0..n_precursors)
            .map(|_| MASS_LOW + random.uniform() * (MASS_HIGH - MASS_LOW))
            .collect();
        let level: Vec<f64> = (0..n_precursors)
            .map(|_| LEVEL_LOW_LOG2 + random.uniform() * LEVEL_SPREAD_LOG2)
            .collect();
        let mut intensity = Vec::with_capacity(n_runs * n_precursors);
        let mut drift = Vec::with_capacity(n_runs * n_precursors);
        for run in 0..n_runs {
            let sign = if run % 2 == 0 { 0.5 } else { -0.5 };
            for precursor in 0..n_precursors {
                let cell_drift = sign
                    * DRIFT_LOG2
                    * (rt[precursor] / GRADIENT_SECONDS)
                    * ((mass[precursor] - MASS_LOW) / (MASS_HIGH - MASS_LOW));
                drift.push(cell_drift);
                intensity
                    .push((level[precursor] + cell_drift + random.normal() * NOISE_SD_LOG2).exp2());
            }
        }
        Self {
            n_runs,
            n_precursors,
            intensity,
            rt: rt.repeat(n_runs),
            mass,
            drift,
        }
    }

    fn correct(&self) -> (Vec<f64>, PrecursorDriftSummary) {
        self.correct_with(&self.intensity, &self.rt)
    }

    fn correct_with(&self, intensity: &[f64], rt: &[f64]) -> (Vec<f64>, PrecursorDriftSummary) {
        let mut corrected = intensity.to_vec();
        let summary = correct_precursor_drift(
            &mut corrected,
            rt,
            &self.mass,
            self.n_runs,
            &PrecursorDriftParameters::DEFAULT,
        );
        (corrected, summary)
    }

    fn correct_in_rounds(&self, rounds: usize, tolerance: f64) -> Vec<f64> {
        let mut corrected = self.intensity.clone();
        correct_precursor_drift_in_rounds(
            &mut corrected,
            &self.rt,
            &self.mass,
            self.n_runs,
            &PrecursorDriftParameters::DEFAULT,
            rounds,
            tolerance,
        );
        corrected
    }

    /// Marks `share` of the cells, drawn at random, as missing.
    fn drop_at_random(&mut self, share: f64, seed: u64) {
        let mut random = Random(seed);
        for value in self.intensity.iter_mut() {
            if random.uniform() < share {
                *value = 0.0;
            }
        }
    }

    /// Log2 correction of every cell, NaN where the cell is missing.
    fn applied(&self, corrected: &[f64]) -> Vec<f64> {
        self.intensity
            .iter()
            .zip(corrected)
            .map(|(before, after)| (before / after).log2())
            .collect()
    }

    /// Applied correction and the drift visible against `runs`, over the quantified cells of
    /// `runs`, both centred per run. The visible drift of a cell is its drift minus the mean drift
    /// of its precursor over `runs`.
    fn applied_and_drift(&self, corrected: &[f64], runs: &[usize]) -> (Vec<f64>, Vec<f64>) {
        let n = self.n_precursors;
        let applied = self.applied(corrected);
        let mean_drift: Vec<f64> = (0..n)
            .map(|precursor| {
                runs.iter()
                    .map(|&run| self.drift[run * n + precursor])
                    .sum::<f64>()
                    / runs.len() as f64
            })
            .collect();
        let (mut all_applied, mut all_drift) = (Vec::new(), Vec::new());
        for &run in runs {
            let cells: Vec<usize> = (run * n..(run + 1) * n)
                .filter(|&cell| applied[cell].is_finite())
                .collect();
            let run_applied: Vec<f64> = cells.iter().map(|&cell| applied[cell]).collect();
            let run_drift: Vec<f64> = cells
                .iter()
                .map(|&cell| self.drift[cell] - mean_drift[cell % n])
                .collect();
            all_applied.extend(centred(&run_applied));
            all_drift.extend(centred(&run_drift));
        }
        (all_applied, all_drift)
    }

    /// Share of the drift that is removed in `runs`: the least squares coefficient of the applied
    /// correction on the visible drift, over the quantified cells.
    fn removed_share(&self, corrected: &[f64], runs: &[usize]) -> f64 {
        let (applied, truth) = self.applied_and_drift(corrected, runs);
        let along: f64 = applied.iter().zip(&truth).map(|(a, t)| a * t).sum();
        along / truth.iter().map(|t| t * t).sum::<f64>()
    }

    /// Root mean square difference between the applied correction and the visible drift in
    /// `runs`.
    fn correction_error(&self, corrected: &[f64], runs: &[usize]) -> f64 {
        let (applied, truth) = self.applied_and_drift(corrected, runs);
        let squared: f64 = applied
            .iter()
            .zip(&truth)
            .map(|(a, t)| (a - t).powi(2))
            .sum();
        (squared / truth.len() as f64).sqrt()
    }

    fn all_runs(&self) -> Vec<usize> {
        (0..self.n_runs).collect()
    }

    fn mean_log2(&self, values: &[f64], run: usize) -> f64 {
        let run_values: Vec<f64> = values[run * self.n_precursors..(run + 1) * self.n_precursors]
            .iter()
            .filter(|value| **value > 0.0)
            .map(|value| value.log2())
            .collect();
        run_values.iter().sum::<f64>() / run_values.len() as f64
    }
}

fn centred(values: &[f64]) -> Vec<f64> {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|value| value - mean).collect()
}

/// Mean removed share over `N_SEEDS` draws of `n_runs` runs; asserts the correction error of
/// every draw.
fn mean_removed_share(n_runs: usize) -> f64 {
    let mut total = 0.0;
    for seed in 0..N_SEEDS {
        let tables = Tables::new(n_runs, N_PRECURSORS, seed);

        let (corrected, summary) = tables.correct();

        assert!(summary.corrected.iter().all(|&corrected| corrected));
        let runs = tables.all_runs();
        let error = tables.correction_error(&corrected, &runs);
        assert!(
            error < MAX_CORRECTION_ERROR_LOG2,
            "{n_runs} runs, seed {seed}: error {error:.3}"
        );
        total += tables.removed_share(&corrected, &runs);
    }
    total / N_SEEDS as f64
}

#[test]
fn test_removes_the_rt_and_mass_dependent_drift_of_three_and_five_runs() {
    for n_runs in [3, 5] {
        let share = mean_removed_share(n_runs);

        assert!(
            share >= MIN_REMOVED_SHARE,
            "{n_runs} runs: removed {share:.3}"
        );
    }
}

#[test]
fn test_corrects_two_runs() {
    let share = mean_removed_share(2);

    assert!(share >= MIN_REMOVED_SHARE, "removed {share:.3}");
}

#[test]
fn test_uses_all_strata_for_a_large_reference() {
    let tables = Tables::new(3, N_PRECURSORS, 0);

    let (_, summary) = tables.correct();

    assert_eq!(summary.n_strata, PrecursorDriftParameters::DEFAULT.n_strata);
}

#[test]
fn test_leaves_a_single_run_unchanged() {
    let tables = Tables::new(1, N_PRECURSORS, 0);

    let (corrected, summary) = tables.correct();

    assert_eq!(summary.corrected, vec![false]);
    assert_eq!(summary.max_abs_correction, vec![0.0]);
    assert_eq!(corrected, tables.intensity);
}

#[test]
fn test_leaves_runs_with_too_few_reference_precursors_unchanged() {
    let tables = Tables::new(
        3,
        PrecursorDriftParameters::DEFAULT.min_reference_per_run - 1,
        0,
    );

    let (corrected, summary) = tables.correct();

    assert!(summary.corrected.iter().all(|&corrected| !corrected));
    assert_eq!(corrected, tables.intensity);
}

#[test]
fn test_uses_fewer_strata_when_the_reference_is_small() {
    // 5000 references fill 10 RT bins of 125 precursors in 4 strata only
    let tables = Tables::new(3, 5000, 0);

    let (_, summary) = tables.correct();

    assert_eq!(summary.n_strata, 4);
}

#[test]
fn test_keeps_missing_cells_missing_and_the_level_of_every_run() {
    let mut tables = Tables::new(3, N_PRECURSORS, 0);
    for precursor in (0..N_PRECURSORS).step_by(5) {
        tables.intensity[precursor] = 0.0;
    }
    tables.intensity[1] = f64::NAN;

    let (corrected, _) = tables.correct();

    for precursor in (0..N_PRECURSORS).step_by(5) {
        assert_eq!(corrected[precursor], 0.0);
    }
    assert!(corrected[1].is_nan());
    for run in 0..tables.n_runs {
        let before = tables.mean_log2(&tables.intensity, run);
        let after = tables.mean_log2(&corrected, run);
        assert!(
            (before - after).abs() < 1e-9,
            "run {run}: {before} vs {after}"
        );
    }
}

#[test]
fn test_places_a_precursor_without_rt_at_its_mean_rt() {
    // the RT is the same in every run, so the mean RT over the other runs is the RT itself
    let tables = Tables::new(3, N_PRECURSORS, 0);
    let mut rt = tables.rt.clone();
    rt[N_PRECURSORS - 1] = f64::NAN;

    let (with_rt, _) = tables.correct();
    let (without_rt, _) = tables.correct_with(&tables.intensity, &rt);

    assert_eq!(with_rt, without_rt);
}

#[test]
fn test_complete_data_gives_the_between_run_ratios_of_a_single_round() {
    let tables = Tables::new(N_PARTIAL_RUNS, N_PRECURSORS, 0);

    let single_round = tables.correct_in_rounds(1, NEVER_STOP);
    let all_rounds = tables.correct_in_rounds(ROW_LEVEL_ROUNDS, NEVER_STOP);

    // a function of the precursor common to all runs leaves every between-run ratio unchanged
    let difference: Vec<f64> = tables
        .applied(&all_rounds)
        .iter()
        .zip(tables.applied(&single_round))
        .map(|(all, single)| all - single)
        .collect();
    for (cell, value) in difference.iter().enumerate() {
        let first_run = difference[cell % tables.n_precursors];
        assert!(
            (value - first_run).abs() < RATIO_TOLERANCE_LOG2,
            "cell {cell}: {value} vs {first_run}"
        );
    }
}

#[test]
fn test_leaves_a_near_blank_run_uncorrected_and_corrects_the_others() {
    let other_runs: Vec<usize> = (1..N_PARTIAL_RUNS).collect();
    let mut total = 0.0;
    for seed in 0..N_SEEDS {
        let mut tables = Tables::new(N_PARTIAL_RUNS, N_PRECURSORS, seed);
        tables.intensity[N_BLANK_RUN_PRECURSORS..N_PRECURSORS].fill(0.0);

        let (corrected, summary) = tables.correct();

        assert!(!summary.corrected[0]);
        assert_eq!(summary.max_abs_correction[0], 0.0);
        assert_eq!(corrected[..N_PRECURSORS], tables.intensity[..N_PRECURSORS]);
        assert!(summary.corrected[1..].iter().all(|&corrected| corrected));
        let error = tables.correction_error(&corrected, &other_runs);
        assert!(
            error < MAX_CORRECTION_ERROR_LOG2,
            "seed {seed}: error {error:.3}"
        );
        total += tables.removed_share(&corrected, &other_runs);
    }
    let share = total / N_SEEDS as f64;

    assert!(share >= MIN_REMOVED_SHARE, "removed {share:.3}");
}

#[test]
fn test_removes_the_drift_with_cells_missing_at_random_as_from_complete_data() {
    let mut complete_total = 0.0;
    let mut missing_total = 0.0;
    for seed in 0..N_MISSING_SEEDS {
        let complete = Tables::new(N_PARTIAL_RUNS, N_PRECURSORS, seed);
        let mut missing = Tables::new(N_PARTIAL_RUNS, N_PRECURSORS, seed);
        missing.drop_at_random(MISSING_SHARE, seed + N_MISSING_SEEDS);
        let runs = complete.all_runs();

        let (complete_corrected, _) = complete.correct();
        let (missing_corrected, summary) = missing.correct();

        assert!(summary.corrected.iter().all(|&corrected| corrected));
        let error = missing.correction_error(&missing_corrected, &runs);
        assert!(
            error < MAX_CORRECTION_ERROR_LOG2,
            "seed {seed}: error {error:.3}"
        );
        complete_total += complete.removed_share(&complete_corrected, &runs);
        missing_total += missing.removed_share(&missing_corrected, &runs);
    }
    let complete_share = complete_total / N_MISSING_SEEDS as f64;
    let missing_share = missing_total / N_MISSING_SEEDS as f64;

    assert!(
        missing_share >= MIN_REMOVED_SHARE,
        "removed {missing_share:.3}"
    );
    assert!(
        complete_share - missing_share < MAX_MISSING_SHORTFALL,
        "removed {missing_share:.3} with missing cells, {complete_share:.3} without"
    );
}

#[test]
fn test_the_rounds_remove_the_bias_of_rows_missing_in_some_runs() {
    let mut single_total = 0.0;
    let mut rounds_total = 0.0;
    for seed in 0..N_MISSING_SEEDS {
        let mut tables = Tables::new(N_PARTIAL_RUNS, N_PRECURSORS, seed);
        tables.drop_at_random(MISSING_SHARE, seed + N_MISSING_SEEDS);
        let runs = tables.all_runs();

        let single_round = tables.correct_in_rounds(1, NEVER_STOP);
        let (all_rounds, _) = tables.correct();

        single_total += tables.removed_share(&single_round, &runs);
        rounds_total += tables.removed_share(&all_rounds, &runs);
    }
    let gain = (rounds_total - single_total) / N_MISSING_SEEDS as f64;

    assert!(gain >= MIN_ROUNDS_GAIN, "gain {gain:.3}");
}
