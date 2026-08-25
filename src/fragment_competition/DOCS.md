# Problem

In data-independent acquisition, the instrument fragments all precursors of an isolation
window together. Two peptides that elute at the same time in the same window therefore give
one mixed MS2 spectrum. The search can report both peptides, and each one claims the same
fragment peaks. Only one of them can own that signal.

Fragment competition makes this assignment exclusive. Two candidates of one isolation window
compete if their retention times are closer than `rt_tol_seconds`, and if they share three
fragment ions or more within `mass_tol_ppm`. The candidate with the better score keeps the
signal:

```text
one isolation window, candidates sorted by proba (best first)

               rt →   10.0 s    10.5 s    20.0 s
  A  proba 0.01         ●                            ions { a b c d e }
  B  proba 0.20                   ●                  ions { a b c f g }
  C  proba 0.40                              ●       ions { a b c f g }

  A ↔ B   0.5 s apart, 3 shared ions (a, b, c)   →  B loses its fragments
  B ↔ C   B already lost, so it invalidates nothing
  A ↔ C   10 s apart, thus they never compete

  result   A ✓   B ✗   C ✓
```

The second comparison shows the rule that is not obvious. A candidate that loses does not
compete again. It cannot invalidate another candidate, and no other candidate can invalidate
it. The order of the comparisons therefore decides which candidates survive. For this reason,
this module groups and sorts the candidates itself. It does not use the row order of the
caller.

# Input

These arrays give one value per candidate. They all have the length `n_candidates`:

| array | dtype | meaning |
|---|---|---|
| `precursor_mz` | f32 | observed precursor m/z. It selects the isolation window |
| `precursor_idx` | i64 | candidate identifier. It breaks a `proba` tie |
| `proba` | f64 | candidate score. A **lower** value is better |
| `rt_observed` | f32 | observed retention time in seconds |
| `frag_start_idx` | i64 | first ion of this candidate in `fragment_mz` |
| `frag_stop_idx` | i64 | last ion of this candidate in `fragment_mz`, exclusive |

Two more arrays are common to all candidates:

| array | dtype | meaning |
|---|---|---|
| `fragment_mz` | f32 | the fragment m/z of all candidates, in one flat array |
| `cycle` | f32 | isolation window limits, shape `(1, n_windows, n_scans, 2)` |

# Output

A `bool` mask with the length `n_candidates`, in the candidate order of the caller. A `true`
value shows that the candidate keeps its fragments. The candidate order has no effect on the
result. If you give the same candidates in a different order, the same candidates survive.

# Algorithm

**1. Group the candidates by isolation window.** Each window covers the largest isolation
range of all of its scans in `cycle`. A candidate belongs to the first window whose range
`[lower, upper)` contains the `precursor_mz` of the candidate. Candidates of different
windows cannot share fragment signal, thus they never compete.

```text
  window 0  [400, 425)      A  412.3  ┐
  window 1  [425, 450)      B  418.9  ┴→ window 0
  window 2  [450, 475)      C  431.0   → window 1
```

**2. Sort each window** by `proba` in increasing order. `precursor_idx` in increasing order
breaks a tie.

**3. Compare the candidates of each window** in this order. Each valid candidate competes
against all other valid candidates of the same window. If the two retention times are closer
than `rt_tol_seconds`, and if the two candidates share three ions or more, the candidate that
is later in the order loses its fragments.

To count the shared ions, the competition compares each pair of m/z values against
`mass_tol_ppm`. The ppm value is relative to the ions of the candidate that claims the
signal:

```text
mass_tol_ppm = 15

  A ions   500.100   612.250   730.400   845.550
  B ions   500.103   612.256   730.900
             6 ppm     10 ppm    684 ppm

  ions within the tolerance: 2  <  3  →  the overlap is a coincidence
```

The windows are independent, thus this module processes them in parallel. Inside one window,
the comparisons are sequential, because of the order rule above.

# Contract

- `precursor_idx` breaks a `proba` tie. The input values therefore give the result, and the
  row order has no effect.
- A float input that contains NaN is an error. A NaN loses each comparison. It can therefore
  change which candidates survive.
- The retention time tolerance and the ppm tolerance are exclusive (`<`). The threshold for
  shared ions is inclusive (`>= 3`).
- Window limits are half-open, `[lower, upper)`.
- If no window contains a precursor m/z, the precursor gets window 0. It then competes with
  the candidates of window 0. This occurs only if the m/z is outside the acquisition range.
- Each fragment range must be inside `fragment_mz`. A range outside `fragment_mz` is an
  error.
