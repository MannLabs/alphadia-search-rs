# Problem

In data-independent acquisition (DIA), spectra are acquired in repeating cycles of isolation
windows. The raw layout is spectrum-centric: the peaks of one spectrum lie contiguously in memory.
Peptide-centric analysis needs the opposite — every observation of one m/z across all cycles —
which in the raw layout is scattered over millions of spectra.

`DIAData` holds a transposed representation. Per isolation window, peaks are grouped by m/z bin and
ordered by cycle, so an extracted ion chromatogram is one contiguous slice:

```text
spectrum-centric (AlphaRawView)          m/z-centric (DIAData)

spectrum 122 ▸ 512.3  687.4              m/z 512.3 ▸ cycle 40  cycle 41
spectrum 125 ▸ 512.3  903.5              m/z 687.4 ▸ cycle 40  cycle 42
spectrum 128 ▸ 687.4                     m/z 903.5 ▸ cycle 41

one row = one scan                       one row = one chromatogram
```

# Terminology

| Field | Meaning |
|---|---|
| `spectrum_idx` | Position in the flat, global list of spectra. |
| `spectrum_delta_scan_idx` | Position of a scan *within* its cycle — the offset ("delta") from the cycle start, not a global counter. `0` is the MS1 survey scan, `1..n` the MS2 isolation windows in acquisition order. Since every cycle repeats the same sequence, one fixed value selects the same isolation window in every cycle. |
| `spectrum_cycle_idx` | Which cycle a spectrum belongs to. |
| `mz_index` | Bin in the global `MZIndex`, a 1 ppm grid spanning 150 to 2000 m/z (2_590_781 bins). |

# Data Transformation

## Input: AlphaRawView (spectrum-centric)

An acquisition with one MS1 survey scan and two MS2 windows per cycle. The excerpt below is taken
from mid-run, cycles 40 to 42:

```text
         cycle 40                     cycle 41                     cycle 42
 ┌─────┬─────────┬─────────┐  ┌─────┬─────────┬─────────┐  ┌─────┬─────────┬─────────┐
 │ MS1 │[400,425[│[425,450[│  │ MS1 │[400,425[│[425,450[│  │ MS1 │[400,425[│[425,450[│
 └─────┴─────────┴─────────┘  └─────┴─────────┴─────────┘  └─────┴─────────┴─────────┘
    Δ=0     Δ=1       Δ=2        Δ=0     Δ=1       Δ=2        Δ=0     Δ=1       Δ=2
   #120   #121      #122        #123   #124      #125        #126   #127      #128
                       ▲                            ▲                            ▲
                       └───────── tracked: same window, once per cycle ──────────┘
```

```text
spectrum_idx | spectrum_delta_scan_idx | spectrum_cycle_idx | isolation_window
-------------|-------------------------|--------------------|-----------------
     120     |            0            |         40         |  (MS1 survey)
     121     |            1            |         40         |    [400, 425[
     122     |            2            |         40         |    [425, 450[
     123     |            0            |         41         |  (MS1 survey)
     124     |            1            |         41         |    [400, 425[
     125     |            2            |         41         |    [425, 450[
     126     |            0            |         42         |  (MS1 survey)
     127     |            1            |         42         |    [400, 425[
     128     |            2            |         42         |    [425, 450[
```

The rest of this example follows `spectrum_delta_scan_idx == 2`, the `[425, 450[` window. That pick
is arbitrary: every MS2 window is built the same way, and all of them are built in parallel. Note
that `spectrum_idx` and `spectrum_cycle_idx` are unrelated — the window is visited once per cycle,
so its three spectra are 122, 125 and 128.

Peaks of those three spectra (referenced via `spectrum_peak_start_idx` / `spectrum_peak_stop_idx`).
These are fragment m/z, so they are unrelated to the precursor isolation window and generally fall
outside it:

```text
Spectrum 122 (cycle 40):     Spectrum 125 (cycle 41):     Spectrum 128 (cycle 42):
  peak_mz=512.3, int=12_450    peak_mz=512.3, int=18_820    peak_mz=687.4, int=9_130
  peak_mz=687.4, int=31_775    peak_mz=903.5, int=4_260
```

## Step 1: Group by spectrum_delta_scan_idx into QuadrupoleObservation

All spectra sharing a `spectrum_delta_scan_idx` become one observation. The isolation window is
taken from the first matching spectrum, `num_cycles` counts the distinct cycles observed:

```text
quadrupole_observations[2] = QuadrupoleObservation {
    isolation_window: [425.0, 450.0],
    num_cycles: 3,
    ...
}
```

## Step 2: Map peak_mz to mz_index via MZIndex::find_closest_index

Each m/z is mapped to the closest bin of the global 1 ppm grid:

```text
MZIndex::find_closest_index(512.3) -> mz_index 1_228_745
MZIndex::find_closest_index(687.4) -> mz_index 1_522_472
MZIndex::find_closest_index(903.5) -> mz_index 1_795_957
```

## Step 3: Build transposed arrays sorted by (mz_index, cycle_indices)

Peaks are sorted first by `mz_index`, then by cycle within each `mz_index`:

```text
position | mz_index  | cycle_indices | intensities
---------|-----------|---------------|------------
    0    | 1_228_745 |      40       |   12_450
    1    | 1_228_745 |      41       |   18_820
    2    | 1_522_472 |      40       |   31_775
    3    | 1_522_472 |      42       |    9_130
    4    | 1_795_957 |      41       |    4_260
```

Each quantity lives in its own range, which makes the arrays below easier to read: positions are
`0..5`, cycles are `40..43`, m/z bins are around `10^6`, intensities around `10^4`.

## Step 4: Create slice_starts index

`slice_starts[i]` marks where the data for `mz_index` `i` begins in `cycle_indices` / `intensities`,
and `slice_starts[i + 1]` marks its end:

```text
mz_index  | slice_starts[i] | slice_starts[i + 1] | slice
----------|-----------------|---------------------|----------------
1_228_745 |        0        |          2          | positions 0..2
1_522_472 |        2        |          4          | positions 2..4
1_795_957 |        4        |          5          | positions 4..5

cycle_indices: [40, 41, 40, 42, 41]
intensities:   [12_450, 18_820, 31_775, 9_130, 4_260]
```

The array covers the whole grid, so its length is `MZIndex::len() + 1` = 2_590_782 and the three
populated bins above are a tiny excerpt: every entry below 1_228_745 holds `0`, every entry above
1_795_957 holds `5`, the final one being the end sentinel. An empty bin has
`slice_starts[i] == slice_starts[i + 1]` and therefore yields a zero-length slice.

# Query Example

To retrieve all observations of m/z 512.3 (`mz_index` 1_228_745):

```text
let (cycles, ints) = observation.get_slice_data(1_228_745);
// start = slice_starts[1_228_745] = 0
// stop  = slice_starts[1_228_746] = 2

cycles = &cycle_indices[0..2] = [40, 41]              // observed in cycles 40 and 41
ints   = &intensities[0..2]   = [12_450.0, 18_820.0]
```

# Benefits

- Memory and allocation optimized storage of sparse arrays instead of list of structs
- O(1) slice lookup with contiguous memory reads, replacing scattered access across spectra

# Constraints

- Requires building transposed structure once upfront
- Upper limit of resolution needs to be known upfront
- The DIA cycle needs to be consistent across all spectra
