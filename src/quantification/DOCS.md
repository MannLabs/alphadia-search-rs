# Problem

A precursor is observed through several fragment ions per run. Each fragment is a noisy,
partly compressed measurement of the same abundance: a weak fragment on an interfering signal
follows the interference rather than the peptide. An equal-vote estimator (the median of the
fragment ratios) therefore compresses the ratios of regulated precursors towards the
background. Fragment-level filtering cannot remove this, the estimator has to weigh the
evidence.

The spectral library predicts the relative intensity `L[f]` of every fragment of a precursor,
run-independent and blind to interference. This module estimates the run quantities of every
precursor from its fragment intensities anchored to that pattern.

# Model

For precursor `g`, fragment `f` and run `r`:

```text
log2(I[f, r] / s[f]) = q[g, r] + c[f] + e[f, r]
```

- `s[f] = L[f] / sum over the fragments of g of L` is the library share of the fragment, so
  that `2 ** q` is on the scale of the summed fragment intensity of the precursor.
- `q[g, r]` is the log2 run quantity of the precursor, the estimate.
- `c[f]` is the run-independent offset of the fragment from the library pattern (prediction
  error of the library). A random effect, shrunk towards 0.
- `e[f, r]` is the residual. Interference appears here as cells far above the model in some
  runs.

Missing cells carry no information and enter no sum. Fragments without a library intensity
(0 or not finite) are ignored.

# Input

| array       | shape              | content                                                  |
|-------------|--------------------|----------------------------------------------------------|
| `intensity` | `(n_runs, n_ions)` | linear fragment intensities, 0 marks a missing cell      |
| `library`   | `(n_ions,)`        | library intensity of every fragment                      |
| `precursor` | `(n_ions,)`        | precursor (group) index of every fragment, `< n_precursors` |

The arrays are run-major and C-contiguous like those of the intensity drift correction. The
fragments of a precursor need not be adjacent.

# Output

A dict with

- `quantity` — `(n_runs, n_precursors)` log2 run quantities, NaN where the run of the
  precursor has fewer than `min_ions` observed fragments.
- `residual_scale`, `loss_scale`, `offset_shrinkage` — the scalars of the fit, for logging.

# Algorithm

Iteratively reweighted least squares with a Cauchy loss. The weight of cell `(f, r)` is

```text
w[f, r] = s[f] / (1 + (e[f, r] / tau) ** 2)
```

The base weight `s[f]` is the expected share of the precursor signal: the inverse variance of
a log intensity under counting statistics. In every sweep, per precursor,

```text
c[f]    = sum_r w (y - q) / (sum_r w + lambda * s[f])
q[g, r] = sum_f w (y - c) / sum_f w
```

with `y = log2(I / s)`. `lambda` counts observations of unit weight, and every cell of
fragment `f` weighs `s[f]` before the loss, hence `lambda * s[f]`. The sweeps of a precursor stop when none of its quantities moves by
more than `tolerance` log2, at most `max_sweeps`.

The loss scale and the shrinkage are derived from the data, not tuned:

| quantity       | value | derivation |
|----------------|-------|------------|
| `tau`          | `2.385 * sigma`, `sigma = 1.4826 * MAD` of the residuals of the first, unweighted sweep over all precursors, taken around the unshrunk fragment offsets and scaled by `sqrt(n_f / (n_f - 1))` | Cauchy tuning constant for 95 % efficiency at the normal distribution (Holland & Welsch 1977); the library offsets are not noise |
| `lambda`       | `sigma ** 2 / sigma_c ** 2`, `sigma_c ** 2 = var(unshrunk per-fragment mean deviations) - mean(sigma ** 2 / n_f)`, floored | random-effects (ridge) shrinkage of a fragment offset towards the library pattern |

`sigma` is floored at a small value so that data whose majority of residuals is exactly zero
still downweights its outliers. Either value can be given instead of derived.
