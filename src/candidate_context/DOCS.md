# Problem

In data-independent acquisition, all precursors of one isolation window are fragmented
together. Two peptides that elute at the same time in the same window therefore share one
mixed MS2 spectrum, and the search can report a candidate for each of them that claims the
same fragment peaks. A post-hoc filter that keeps only the best of such candidates removes
evidence from the classifier. This module instead measures the competition for every
candidate, target and decoy alike, **before** the classifier sees it: how many other
candidates elute in the same window at the same cycle, how many of them claim the same
fragment m/z values, and how much of this candidate's evidence a higher-scoring competitor
also claims.

```text
one isolation window, one cycle

  A  score 2.0   fragments { 300 400 500 600 700 800 }
  B  score 1.0   fragments { 300 400 500 610 720 830 }
  C  score 3.0   fragments { 300 400 500 600 700 800 }   three cycles later

  A ↔ B   same cycle, 3 shared m/z   →  competitors; A is the claimant (higher score)
  A ↔ C   3 cycles apart             →  never compared

  A: n_competitors 1, claimant_rank 1, shared_frac_any 0.5, shared_frac_higher 0.0
  B: n_competitors 1, claimant_rank 2, shared_frac_any 0.5, shared_frac_higher 0.5
  C: n_competitors 0
```

# Input

`compute(dia_data, lib, candidates)` takes the same three objects as peak group scoring. The
fragments of a candidate are the fragments the scorer uses: the `top_k_fragments` most intense
library fragments with non-zero intensity, without y1 ions. The parameters are set in the
constructor:

| parameter | default | meaning |
|---|---|---|
| `mass_tolerance` | required | two fragment m/z are the same ion if they lie within this value in ppm of each other |
| `top_k_fragments` | required | fragments per candidate, the same value as used for scoring |
| `min_shared` | 3 | two candidates compete if they share this many fragment m/z or more |
| `cycle_radius` | 1 | two candidates are compared if their apex cycles are at most this far apart |

# Output

A dict of arrays with one row per candidate, in the candidate order of the caller:
`precursor_idx`, `rank` and the eight columns of `get_feature_names()`.

| column | meaning |
|---|---|
| `ctx_candidate_density` | candidates in the same window within `cycle_radius` cycles, minus one |
| `ctx_n_competitors` | other precursors that share at least `min_shared` fragment m/z |
| `ctx_n_competitors_higher` | competitors with a higher selection score |
| `ctx_claimant_rank` | `1 + ctx_n_competitors_higher` |
| `ctx_shared_frac_any` | fraction of fragments matched by any other precursor |
| `ctx_shared_frac_higher` | fraction of fragments claimed by a higher-scoring competitor |
| `ctx_shared_lib_intensity_frac_higher` | the same fraction, weighted by library intensity |
| `ctx_competitor_log_ratio` | `ln(best competitor score / own score)`, 0 without competitors |

# Algorithm

**1. Index.** Every fragment of every candidate becomes one entry `(key, candidate, m/z)`.
The key packs the isolation window, the apex cycle of the candidate and a logarithmic m/z bin
into one integer, `window << 48 | cycle << 32 | bin`. The window is the first isolation
window that contains the precursor m/z. The bin width is `ln(1 + 2 · tolerance)`, twice the
tolerance, so that two m/z values within the tolerance always fall into the same or an
adjacent bin. The entries are sorted by key. A candidate whose precursor m/z lies in no
window, or whose precursor is not in the library, adds no entries and gets the values of a
candidate without competition: `ctx_claimant_rank` 1 and zero elsewhere.

**2. Query.** For every candidate and every fragment, the three bins `bin - 1 .. bin + 1` of
each cycle within `cycle_radius` form one contiguous key range. The entries in that range
that belong to another precursor and lie within the tolerance are matches. The matches are
counted per other candidate; a candidate with `min_shared` matches or more is a competitor.
The density is the number of candidates with the same window and a cycle within
`cycle_radius`, found in the same way in a second, smaller sorted index.

The index is built once per call and dropped afterwards. Its size is the number of candidate
fragments times 16 bytes. Both phases run in parallel over the candidates.

# Contract

- The features are symmetric for targets and decoys: the index holds both, and no feature
  looks at the decoy flag.
- Other candidates of the same precursor (other ranks) are never competitors and never
  count as a match, but they do count for the density.
- The m/z tolerance is relative to the fragment of the candidate that is evaluated, and it is
  inclusive (`<=`). The threshold for shared fragments is inclusive (`>=`).
- "Higher" means a strictly higher selection score. Two candidates with the same score are
  competitors, and neither is the claimant of the other.
- The number of isolation windows and the cycle indices must fit into 16 bits each, and the
  number of candidates into 32 bits. Otherwise `compute` raises `ValueError`.
