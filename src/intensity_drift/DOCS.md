# Problem

The ion intensity of a run can drift along the gradient, for example when the spray or the
column changes during the run. A run that loses signal towards the end of the gradient makes
every late-eluting peptide look less abundant than in the other runs. A single normalization
factor per run cannot remove such a drift, because it changes along the retention time (RT).

This module estimates the drift of every run along the RT and divides it out of the fragment
intensities before label-free quantification. It removes only the shape along the gradient:
the mean log2 level of every run stays the same and is left to the LFQ normalization.

# Input

The matrices are run-major and C-contiguous. They are borrowed, not copied.

| array         | shape                  | content                                                  |
|---------------|------------------------|----------------------------------------------------------|
| `intensity`   | `(n_runs, n_ions)`     | linear fragment intensities, 0 marks a missing cell      |
| `correlation` | `(n_runs, n_ions)`     | cross-run fragment correlation                           |
| `rt`          | `(n_runs, n_precursors)` | observed precursor RT, NaN where the run has no identification |
| `precursor`   | `(n_ions,)`            | column of `rt` for every ion                             |

`intensity` is corrected in place.

# Output

A dict with one entry per run:

- `corrected` — whether the run was corrected. A run with fewer than `min_ions_per_run`
  usable ions, or with all of them in a single bin, is left unchanged.
- `max_abs_correction` — the largest absolute correction of the run, in log2; 0 for a run
  that is not corrected.

# Terms

- *ion*: a fragment ion, one column of the intensity matrix.
- *usable ion*: an ion observed in at least `min_observations` runs with a mean cross-run
  correlation of at least `min_correlation`.
- *reference level*: median log2 intensity of a usable ion over the runs it is observed in.
- *deviation*: log2 intensity of a cell minus the reference level of its ion.
- *bin*: an RT interval of `bin_seconds`, merged with its successors until it holds
  `min_ions_per_bin` usable ions observed in the run.
- *curve*: per run, the location of the deviations in every bin, smoothed over three bins and
  centred to zero mean over the run's observed cells.
- *correction*: the curve linearly interpolated at the RT of a cell, in log2.

# Algorithm

1. Compute the reference level of every usable ion.
2. Per run, collect the deviations of its observed usable ions, place each at the RT of its
   precursor in the run, and sort them by RT. An ion whose precursor has no identification in
   the run is placed at the mean RT of the precursor over the other runs.
3. Group the deviations into bins and take the location of every bin: the mode of the
   smoothed deviation histogram, which is robust to a subset of ions with a true abundance
   change between the runs. A bin whose deviations span too few histogram bins falls back to
   the median.
4. Smooth the locations into the curve, divide every observed cell by `2^correction`, and
   centre the curve so that the run keeps its mean log2 level. The centring keeps the LFQ
   normalization from being forced towards a ratio of 1 for the majority of ions.
