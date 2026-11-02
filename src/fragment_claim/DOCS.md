# Problem

A candidate scores well when the fragment traces it extracts rise and fall together. In
data-independent acquisition the traces at a fragment's m/z are not owned by that candidate:
every precursor fragmented in the same isolation window at the same time contributes to the
same MS2 spectra. A candidate for a peptide that is not in the sample (a decoy or a null
target) therefore matches signal that belongs to a peptide that is in the sample, and if that
peptide is abundant the borrowed traces co-elute perfectly. These are the well-matching
null candidates that set the score threshold.

The candidate context features (`candidate_context`) count how many other candidates could
claim the same library m/z values. This module goes one step further and looks at the
signal that was actually matched. For every candidate it extracts the same fragment traces as
the scorer, keeps the fragments with signal, and asks for each one whether another
candidate in the same window, with its apex at most `cycle_radius` cycles away and with a
higher ownership score, matched a trace at the same observed m/z. Such a fragment is
*claimed*. The features report how much of the candidate's evidence is claimed and what the
evidence looks like once the claimed fragments are taken away.

```text
one isolation window, one apex cycle; observed fragment m/z after extraction

  A  ownership score 20   matched { 300.10  400.20  500.30  600.40  700.50 }
  B  ownership score  9   matched { 300.10  400.20  500.30  830.90 }

  B: 3 of 4 matched fragments are also matched by the higher-scoring A
     clm_frac_claimed 0.75, clm_n_claimants 1, free evidence = { 830.90 } only
  A: nothing claimed (B scores lower), clm_frac_shared_any 0.6
```

# Input

`compute(dia_data, lib, candidates)` takes the same three objects as peak group scoring. The
ownership score of a candidate is its `score` in the candidate collection, i.e. the selection
score, which is a label-blind function of the data. The caller may overwrite it with any
other label-blind score. Parameters:

| parameter | default | meaning |
|---|---|---|
| `mass_tolerance` | required | ppm tolerance for the trace extraction, the same value as used for scoring |
| `top_k_fragments` | required | fragments per candidate, the same value as used for scoring |
| `claim_tolerance` | `mass_tolerance` | two observed fragment m/z are the same ion if they lie within this value in ppm |
| `cycle_radius` | 1 | another candidate can claim a fragment if its apex cycle is at most this far away |
| `soft_claims` | `False` | also compute the soft claims (`clm_soft_*`, see below) |

# Output

A dict of arrays with one row per candidate, in the candidate order of the caller:
`precursor_idx`, `rank` and the columns of `get_feature_names(soft_claims)`. The hard-claim
columns are always returned, the soft-claim columns only with `soft_claims`.

| column | meaning |
|---|---|
| `clm_n_matched` | fragments with signal in the candidate's window |
| `clm_n_claimed` | matched fragments also matched by a higher-scoring candidate |
| `clm_frac_claimed` | `clm_n_claimed / clm_n_matched` |
| `clm_frac_shared_any` | matched fragments also matched by any other candidate, divided by `clm_n_matched` |
| `clm_intensity_frac_claimed` | observed intensity of the claimed fragments over the observed intensity of all matched fragments |
| `clm_lib_frac_claimed` | library intensity of the claimed fragments over the library intensity of the matched fragments |
| `clm_top3_claimed` | how many of the three most intense library fragments are claimed |
| `clm_n_claimants` | distinct higher-scoring candidates that claim at least one fragment |
| `clm_claimant_log_ratio` | `ln(best claimant score / own score)`, 0 without a claimant |
| `clm_mean_corr_free` | mean profile correlation of the matched fragments that are not claimed |
| `clm_num_over_50_free` | unclaimed matched fragments with profile correlation above 0.5 |
| `clm_num_over_0_free` | unclaimed matched fragments with profile correlation above 0 |
| `clm_hyperscore_free` | library-intensity hyperscore over the unclaimed matched fragments |
| `clm_log10_intensity_free` | log10 of the observed intensity of the unclaimed matched fragments |

With `soft_claims`:

| column | meaning |
|---|---|
| `clm_soft_intensity_frac_claimed` | share of the matched observed intensity the mixture model attributes to other precursors |
| `clm_soft_top3_frac_claimed` | the same over the three most intense library fragments |
| `clm_soft_log10_intensity_own` | log10 of the observed intensity the mixture model attributes to the candidate |
| `clm_soft_lib_corr_own` | correlation of the candidate's own intensities with its library intensities |

A candidate without a window, without library fragments or without signal has every feature
at zero.

# Algorithm

**1. Extract.** For every candidate the dense fragment traces over its cycle window are
extracted as in the scorer (same precursor filter, same tolerance). A fragment is matched if
its trace carries any intensity. Its observed m/z is the intensity-weighted mean of the
centroids; its profile correlation is the correlation with the median of the non-zero
normalised traces, as in the scorer.

**2. Index.** Every matched fragment becomes one entry `(key, candidate, m/z)` with the key
`window << 48 | apex cycle << 32 | log-m/z bin`, the layout of `candidate_context`. The bin
width is `ln(1 + 2 · claim_tolerance)`.

**3. Query.** For every candidate and every matched fragment, the bins `bin - 1 .. bin + 1`
of each cycle within `cycle_radius` are scanned. An entry of another precursor within the
claim tolerance of the fragment's observed m/z is a match; the fragment is claimed if any
matching candidate has a strictly higher ownership score. The features are then plain
functions of the matched, claimed and free fragment sets.

**4. Soft claims.** Per window and cycle, the observed peaks of the co-isolated precursors are
fitted as a non-negative mixture of their library spectra
(`peak_group_scoring::deconvolution::compute_shares`): the ranks of one precursor share one
abundance, and library fragments without signal are zero rows, so that a precursor cannot take
a shared peak while predicting intensity where nothing was observed. Every candidate's share of
a shared peak is its part of the fitted intensity; a peak no other precursor matched stays its
own. The soft-claim columns compare the candidate's own intensity with the raw one. With soft
claims every candidate is extracted three times (scorer, shares, claims).

Both the extraction and the query run in parallel over the candidates. The index holds 16
bytes per matched fragment.

# Contract

- The features are symmetric for targets and decoys: the index holds both, and no feature
  looks at the decoy flag.
- Every precursor competes with every other on equal terms: there is no exclusion by decoy
  label, elution group, peptide or protein, in the hard or in the soft claims.
- Other candidates of the same precursor (other ranks) never claim and are never claimed.
- Only the first valid isolation window of a candidate is indexed, while the extraction (like
  the scorer) sums the traces of all valid windows. With overlapping windows a candidate
  is only claimed by candidates indexed in the same first window.
- The output does not depend on the number of threads.
- "Higher" means a strictly higher ownership score; equal scores do not claim.
- The claim tolerance is relative to the observed m/z of the fragment that is evaluated and
  inclusive (`<=`).
- The number of isolation windows and the cycle indices must fit into 16 bits each and the
  number of candidates into 32 bits, else `compute` raises `ValueError`.
