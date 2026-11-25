# Problem

The ion intensity of a run can drift along the gradient, for example when the spray or the
column changes during the run. A run that loses signal towards the end of the gradient makes
every late-eluting peptide look less abundant than in the other runs. A single normalization
factor per run cannot remove such a drift, because it changes along the retention time (RT).
The drift can also differ between light and heavy peptides at the same RT.

This module estimates the drift of every run and divides it out before label-free
quantification. It removes only the shape of the drift: the mean log2 level of every run stays
the same and is left to the LFQ normalization. There are two corrections:

- `IntensityDriftCorrection` corrects the fragment intensities along the RT.
- `PrecursorDriftCorrection` corrects the precursor intensities along the RT and the neutral
  precursor mass.

Neither uses the sample design: the reference of every ion or precursor is its mean over the
runs, never the median. With a median, one run of every ion or precursor sits exactly on its
reference, and the resulting spike of zero deviations pulls the location of every bin towards
no correction.

# IntensityDriftCorrection

## Input

The matrices are run-major and C-contiguous. They are borrowed, not copied.

| array         | shape                  | content                                                  |
|---------------|------------------------|----------------------------------------------------------|
| `intensity`   | `(n_runs, n_ions)`     | linear fragment intensities, 0 marks a missing cell      |
| `correlation` | `(n_runs, n_ions)`     | cross-run fragment correlation                           |
| `rt`          | `(n_runs, n_precursors)` | observed precursor RT, NaN where the run has no identification |
| `precursor`   | `(n_ions,)`            | column of `rt` for every ion                             |

`intensity` is corrected in place.

## Output

A dict with one entry per run:

- `corrected` — whether the run was corrected. A run with fewer than `min_ions_per_run`
  usable ions, or with all of them in a single RT window, is left unchanged.
- `max_abs_correction` — the largest absolute correction of the run, in log2; 0 for a run
  that is not corrected.

## Terms

- *ion*: a fragment ion, one column of the intensity matrix.
- *usable ion*: an ion observed in every run with a mean cross-run correlation of at least
  `min_correlation`.
- *reference level*: mean log2 intensity of a usable ion over the runs.
- *deviation*: log2 intensity of a cell minus the reference level of its ion.
- *RT window*: an RT interval of `window_seconds`, merged with its successors until it holds
  `min_ions_per_window` usable ions observed in the run.
- *curve*: per run, the location of the deviations in every RT window, smoothed over three RT windows and
  centred to zero mean over the run's observed cells.
- *correction*: the curve linearly interpolated at the RT of a cell, in log2.

## Algorithm

1. Compute the reference level of every usable ion.
2. Per run, collect the deviations of its observed usable ions, place each at the RT of its
   precursor in the run, and sort them by RT. An ion whose precursor has no identification in
   the run is placed at the mean RT of the precursor over the other runs.
3. Group the deviations into RT windows and take the location of every RT window: the mode of the
   smoothed deviation histogram, which is robust to a subset of ions with a true abundance
   change between the runs. An RT window whose deviations span too few histogram bins falls back to
   the median.
4. Smooth the locations into the curve, divide every observed cell by `2^correction`, and
   centre the curve so that the run keeps its mean log2 level. The centring keeps the LFQ
   normalization from being forced towards a ratio of 1 for the majority of ions.

# PrecursorDriftCorrection

## Input

The matrices are run-major and C-contiguous. They are borrowed, not copied.

| array       | shape                    | content                                                   |
|-------------|--------------------------|-----------------------------------------------------------|
| `intensity` | `(n_runs, n_precursors)` | linear precursor intensities, 0 or NaN marks a missing cell |
| `rt`        | `(n_runs, n_precursors)` | observed precursor RT, NaN where unknown                  |
| `mass`      | `(n_precursors,)`        | neutral precursor mass                                    |

`intensity` is corrected in place.

## Output

A dict:

- `corrected` — per run, whether the run was corrected. With fewer than two runs, every run is
  left unchanged. A run with fewer than `min_reference_per_run` reference precursors
  quantified in it is left unchanged; the other runs are still corrected.
- `max_abs_correction` — per run, the largest absolute correction, in log2; 0 for a run that
  is not corrected.
- `n_mass_bins` — the number of mass bins used, 0 without reference precursors.
- `reference_runs` — the number of runs every reference precursor is quantified in, at least;
  0 without reference precursors.

## Terms

- *reference precursor*: a precursor with a finite mass quantified in at least `k` runs. `k`
  starts at the number of runs and is lowered one run at a time until at least
  `min_reference_precursors` precursors with a finite mass are quantified in `k` or more runs,
  or `k` is two; a small data set therefore uses all precursors quantified in two runs, even if
  they are fewer. A precursor missing in a run was mostly not detected there, so
  its other intensities are selected on their own noise: complete rows are preferred, and fewer
  runs are required only where complete rows are too few, as with a near-blank run or in a
  large cohort.
- *reference level*: per reference precursor, the mean over the runs it is quantified in of its log2
  intensity minus the run offset.
- *run offset*: the surface of a run at the RT and mass of a reference precursor, minus the mean of
  the surfaces of all runs with a surface at that precursor (zero sum over the runs); 0 in a
  run without a surface. All run offsets are 0 before the first round.
- *deviation*: log2 intensity of a reference precursor in a run where it is quantified minus its
  reference level.
- *mass bin*: a neutral precursor mass interval. The edges are quantiles of the reference masses,
  equal edges collapse, so every mass bin holds reference precursors. Its centre is the median
  mass of its reference precursors. There are `n_mass_bins` mass bins, fewer when the reference
  precursors cannot fill `min_rt_bins_per_mass_bin` RT bins of `points_per_cell` each per mass
  bin, and at least one.
- *RT bin*: per run, an RT interval at quantiles of the RT of the reference precursors, about
  `points_per_cell` of them per mass bin and RT bin. Its centre is their median RT.
- *cell*: an RT bin of a run in a mass bin.
- *location*: the half-sample mode of the deviations of a cell with at least
  `min_points_per_cell` of them. The half-sample mode repeatedly keeps the half of the values
  in the narrowest interval, so it is robust to precursors with a true abundance change.
- *surface*: the locations smoothed with a Gaussian kernel of `sigma_rt` bins and
  `sigma_mass` mass bins, weighted by the number of deviations of every cell (Nadaraya-Watson);
  the grid continues beyond its edges with the edge values. Cells without a smoothed value
  are filled by linear interpolation along the RT (constant beyond the outer filled cells),
  then mass bins without any value from the neighbouring mass bins.
- *correction*: the surface interpolated bilinearly at the position of a cell between the
  RT bin and mass bin centres, constant beyond the outer centres, in log2, centred to zero mean
  over the corrected cells of the run.

## Algorithm

1. A precursor without an RT in a run is placed at its mean RT over the other runs.
2. Select the reference precursors and the mass bins.
3. Estimate the reference levels and the surfaces alternately, in at most three rounds:
   1. Compute the reference levels from the current run offsets. Stop if no reference level changed by more
      than 1e-6 since the last round; the surfaces of the last round stay.
   2. Per run, place the deviations of the reference precursors quantified in the run with an
      RT on the RT bin x mass bin grid and estimate the surface. A run with fewer than
      `min_reference_per_run` of them, or without a cell holding `min_points_per_cell`, has no
      surface.
   3. Compute the run offsets.

   With the plain mean over its quantified runs, a precursor missing in some runs would be
   measured against a level that carries the drift of exactly those runs. The run offsets remove that drift
   from the reference levels. Complete rows do not move, as their run offsets sum to zero, so with
   complete data the second round reproduces the first and the rounds stop.
4. Divide every quantified cell with an RT and a mass of a run with a surface by
   `2^correction`. The run keeps its mean log2 level. Runs without a surface are left
   unchanged.
