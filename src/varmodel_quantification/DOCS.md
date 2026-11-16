# Problem

A precursor is observed through several fragment ions per run. Each fragment reflects the
same abundance with noise and partial compression: a weak fragment on an interfering signal
follows the interference rather than the peptide. An equal-vote estimator (the median of the
fragment ratios) therefore compresses the ratios of regulated precursors towards the
background. Fragment-level filtering cannot remove this, the estimator has to weigh the
evidence.

The spectral library predicts the relative intensity `L[f]` of every fragment of a precursor,
run-independent and blind to interference. This module estimates the run quantities of every
precursor from its fragment intensities anchored to that pattern.

# Assumptions

A precursor is observed through several fragment ions in several runs. A fragment
measurement is the intensity of one fragment in one run. The run quantity is the log2
estimate of the precursor abundance in a run, on the scale of the summed fragment
intensity of the precursor.

1. Proportionality. In the absence of interference, the intensity of every fragment is
   proportional to the precursor abundance, so fragment ratios are run-independent. Errors
   scale with intensity, so the model is additive on the log2 scale.

2. Library prior. The spectral library predicts the relative fragment intensities. Its
   prediction error is a run-independent offset per fragment, modelled as a random effect
   centred on zero whose variance is estimated from the data.

3. Intensity-dependent variance. The residual variance of a fragment measurement depends
   only on its expected intensity and is common to all precursors and runs. It decreases
   with intensity and is bounded at the detection floor and at high intensity. Residuals
   are independent given this variance.

4. Sparse interference. Interference affects a minority of fragment measurements and
   shifts them predominantly upwards. The residuals are therefore heavy-tailed, and the
   estimator must bound the influence of single measurements.

# Procedure

1. Initial estimate. Each fragment, scaled by its library share, estimates the precursor
   abundance. Their weighted average gives a first run quantity per precursor.

2. Noise model. The residuals of all precursors around this estimate show how noisy a
   fragment measurement is at a given intensity, how far fragments deviate from the
   library, and what counts as an outlier.

3. Robust fit. For each precursor, run quantities and fragment offsets are refined
   together. Fragment measurements are weighted by their expected precision, outliers
   lose influence, and fragment offsets are kept close to the library.

# Model

For precursor `g`, fragment `f` and run `r`, the fragment measurement `I[f, r]` follows

```text
log2(I[f, r] / s[f]) = q[g, r] + c[f] + e[f, r]
```

- `s[f] = L[f] / sum over the fragments of g of L` is the library share of the fragment, so
  that `2 ** q` is on the scale of the summed fragment intensity of the precursor.
- `q[g, r]` is the log2 run quantity of the precursor, the estimate.
- `c[f]` is the run-independent offset of the fragment from the library pattern (prediction
  error of the library). A random effect, shrunk towards 0.
- `e[f, r]` is the residual. Interference appears here as measurements far above the model in some
  runs.

Missing measurements carry no information and enter no sum. Fragments without a library intensity
(0 or not finite) are ignored.

# Input

| array       | shape              | content                                                  |
|-------------|--------------------|----------------------------------------------------------|
| `intensity` | `(n_runs, n_ions)` | linear fragment intensities, 0 marks a missing measurement |
| `intensity_library` | `(n_ions,)` | library intensity of every fragment                |
| `precursor` | `(n_ions,)`        | precursor (group) index of every fragment, `< n_precursors` |

The arrays are run-major and C-contiguous like those of the intensity drift correction. The
fragments of a precursor need not be adjacent.

# Output

A dict with

- `quantity` — `(n_runs, n_precursors)` log2 run quantities, NaN where the run of the
  precursor has fewer than 2 observed fragments (`min_ions`).
- `residual_scale`, `loss_scale`, `offset_shrinkage` — the scalars of the fit, for logging.

# Algorithm

Iteratively reweighted least squares with a Cauchy loss. The weight of measurement `(f, r)` is

```text
w[f, r] = 1 / (v(mu[f, r]) + (sigma / tau) ** 2 * e[f, r] ** 2)
```

the inverse variance `1 / v` of the measurement, downweighted by a Cauchy loss of scale
`tau * sqrt(v) / sigma`.

In every sweep, per precursor,

```text
c[f]    = sum_r w (y - q) / (sum_r w + lambda / sigma ** 2)
q[g, r] = sum_f w (y - c) / sum_f w
```

with `y = log2(I / s)`. The sweeps of a precursor stop when none of its quantities moves by
more than 1e-3 log2 (`tolerance`), at most 20 (`max_sweeps`).

`v(mu)` is the residual variance at the expected log2 intensity `mu[f, r] = log2(s[f]) + q[g, r]`
of the measurement, under the quantities of the first sweep. The variance falls with intensity where counting
noise dominates and levels off at both ends: at the detection floor, where faint measurements are as
noisy as the background, and at the top, where the fragment-to-fragment noise
dominates. A weight by library share alone would see neither end. `v` is
estimated from the residuals that give `sigma` (below): their squared scaled median absolute deviation (MAD) in up to 30
bins of equal count along `mu`, at most one bin per 50 residuals, made non-increasing in `mu` and
interpolated linearly in log variance between the bin medians. The loss scale keeps the
tuning constant of every measurement relative to its own standard deviation, and `lambda` counts
observations of variance `sigma ** 2`.

The loss scale and the shrinkage are derived from the data, not tuned:

| quantity       | value | derivation |
|----------------|-------|------------|
| `tau`          | `2.385 * sigma`, `sigma = 1.4826 * MAD` of the residuals of the first sweep (library-share weights, no loss) over all precursors, taken around the unshrunk fragment offsets and scaled by `sqrt(n_f / (n_f - 1))` | Cauchy tuning constant for 95 % efficiency at the normal distribution (Holland & Welsch 1977); the library offsets are not noise |
| `lambda`       | `sigma ** 2 / sigma_c ** 2`, `sigma_c ** 2 = var(unshrunk per-fragment mean deviations) - mean(sigma ** 2 / n_f)`, floored | random-effects (ridge) shrinkage of a fragment offset towards the library pattern |

`sigma` is floored at a small value so that data whose majority of residuals is exactly zero
still downweights its outliers.

# Censored missing fragments and quality weights

Two options extend the fit; with both off it is the fit above.

`missing_fragments="censored"`: the backend reports a fragment only when its XIC carries signal,
so a missing fragment of a quantified run (at least one observed fragment) says "below the
detection limit". Leaving it out reads every precursor near the limit too high in its low runs,
which compresses ratios. Each run gets a dropout curve fitted on its own data,

```text
P(missing | mu) = pi + (1 - pi) * Phi((L - mu) / s)
```

with `mu = q[g, r] + c[f] + log2(s[f])` the expected log2 fragment intensity. The fit maximises
the observed-data likelihood: observed measurements keep their robust weights, and a missing
measurement adds `log P(missing | mu)`, entered as an EM pseudo-observation with the exact score
of that term,

```text
y = mu - s * phi(z) / Phi(z) - log2(s[f]),   weight = rho / s ** 2,   z = (L - mu) / s
```

where `rho = (1 - pi) Phi(z) / (pi + (1 - pi) Phi(z))` is the posterior probability that the
measurement was censored. A run whose curve never censors (no missing fragments) gives its
missing measurements weight 0.

The curve of a run is fitted by maximum likelihood (Nelder-Mead on `(L, log s, logit pi)`) on the
measurements of the precursor runs with at least 4 observed fragments, at most `curve_max_cells`
of them drawn at random, starting from the 2 % quantile of the observed `mu`, `s = 1` and
`pi = max(missing share / 2, 0.001)`. Fit and curves alternate: the fit without censoring, then
twice a curve on the current fit and a fit with that curve.

`quality_exponent = gamma > 0`: the variance of measurement `(f, r)` is multiplied by
`exp(gamma * (1 - sqrt(C[f, r])))`, `C` the XIC correlation of the fragment in that run, so that
a value with a poor correlation in one run (interference, background) counts less in that run
only, instead of dropping the fragment in every run.

With either option the fit runs in f64 (the plain fit stores measurements in f32), sweeps all
precursors in lockstep until no run quantity of any precursor moves by more than 1e-4 log2, at
most 60 sweeps, and reports the dropout curves (`detection_limit`, `dropout_width`,
`dropout_plateau` per run) when censoring.
