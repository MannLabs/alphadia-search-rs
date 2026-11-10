//! Spatial ubiquity of fragment ions: in how many other isolation windows of the same cycle a fragment m/z is lit.
//!
//! The detection background (`detection`) is a channel's hit rate over time in the precursor's own window. A fragment
//! ion that is lit in many windows at the same moment (short common y / b ions, contaminants) is a common ion: a hit
//! on it says little about this peptide. Windows within `EXCLUDE_WINDOWS` ranks of the precursor's own window are not
//! counted, because its isotope envelope reaches them and a present peptide fragments there too.
//!
//! The index holds, per DIA cycle, every MS2 centroid with intensity > 0 as (m/z bin, window rank), sorted, so the
//! ubiquity of one fragment is a range scan of the bins in its tolerance.

use rayon::prelude::*;
use std::time::Instant;

use crate::mz_index::MZIndex;
use crate::traits::{DIADataTrait, QuadrupoleObservationTrait};

/// Windows within this many m/z ranks of the precursor's own are not counted (co-isolation reach).
pub const EXCLUDE_WINDOWS: usize = 3;
/// Bits of a key that hold the window rank; the m/z bin takes the bits above.
const RANK_BITS: u32 = 10;
const MAX_WINDOWS: usize = 1 << RANK_BITS;
const RANK_MASK: u32 = (MAX_WINDOWS as u32) - 1;
const BITSET_WORDS: usize = MAX_WINDOWS / 64;

pub struct UbiquityIndex {
    /// entries cycle_start[c]..cycle_start[c + 1] of `keys` belong to cycle c
    cycle_start: Vec<usize>,
    /// (m/z bin << RANK_BITS) | window rank, sorted within each cycle
    keys: Vec<u32>,
    /// lower isolation bounds of the MS2 windows in rank order
    window_lower: Vec<f32>,
}

impl UbiquityIndex {
    /// The index of a run, or None when its windows or m/z bins do not fit the key.
    pub fn build<T: DIADataTrait>(dia_data: &T) -> Option<Self> {
        let start_time = Instant::now();
        let observations = dia_data.quadrupole_observations();
        let mut windows: Vec<(usize, f32)> = observations
            .iter()
            .enumerate()
            .map(|(idx, obs)| (idx, obs.isolation_window()))
            .filter(|(_, w)| w[1] > 0.0)
            .map(|(idx, w)| (idx, w[0]))
            .collect();
        let num_bins = dia_data.mz_index().len();
        if windows.is_empty()
            || windows.len() > MAX_WINDOWS
            || num_bins as u64 > (u32::MAX >> RANK_BITS) as u64
        {
            return None;
        }
        windows.sort_by(|a, b| a.1.total_cmp(&b.1));
        let num_cycles = dia_data.rt_index().rt.len();

        let mut counts = vec![0usize; num_cycles + 1];
        for &(obs, _) in &windows {
            for bin in 0..num_bins {
                let (cycles, intensities) = observations[obs].peaks_of_bin(bin);
                for (&c, &i) in cycles.iter().zip(intensities) {
                    if i > 0.0 && (c as usize) < num_cycles {
                        counts[c as usize + 1] += 1;
                    }
                }
            }
        }
        for c in 0..num_cycles {
            counts[c + 1] += counts[c];
        }
        let cycle_start = counts;

        let mut cursor = cycle_start.clone();
        let mut keys = vec![0u32; cycle_start[num_cycles]];
        for (rank, &(obs, _)) in windows.iter().enumerate() {
            for bin in 0..num_bins {
                let (cycles, intensities) = observations[obs].peaks_of_bin(bin);
                for (&c, &i) in cycles.iter().zip(intensities) {
                    let c = c as usize;
                    if i > 0.0 && c < num_cycles {
                        keys[cursor[c]] = ((bin as u32) << RANK_BITS) | rank as u32;
                        cursor[c] += 1;
                    }
                }
            }
        }
        let mut rest = keys.as_mut_slice();
        let mut segments = Vec::with_capacity(num_cycles);
        for c in 0..num_cycles {
            let (segment, tail) = rest.split_at_mut(cycle_start[c + 1] - cycle_start[c]);
            segments.push(segment);
            rest = tail;
        }
        segments
            .into_par_iter()
            .for_each(|segment| segment.sort_unstable());

        println!(
            "Built the fragment ubiquity index: {} centroids over {} windows and {} cycles in {:.2} s",
            keys.len(),
            windows.len(),
            num_cycles,
            start_time.elapsed().as_secs_f32()
        );
        Some(Self {
            cycle_start,
            keys,
            window_lower: windows.into_iter().map(|(_, lower)| lower).collect(),
        })
    }

    /// Rank of the precursor's own window: the last window whose lower bound is at most its m/z.
    fn own_rank(&self, precursor_mz: f32) -> usize {
        self.window_lower
            .partition_point(|&lower| lower <= precursor_mz)
            .saturating_sub(1)
    }

    /// Per fragment, the share of far windows (more than `EXCLUDE_WINDOWS` ranks from the precursor's own) that hold a
    /// centroid within `tolerance_ppm` of the fragment m/z in `cycle`. 0 for fragments with m/z <= 0.
    pub fn fragment_ubiquity(
        &self,
        mz_index: &MZIndex,
        cycle: usize,
        precursor_mz: f32,
        fragment_mz: &[f32],
        tolerance_ppm: f32,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; fragment_mz.len()];
        let num_windows = self.window_lower.len();
        if cycle + 1 >= self.cycle_start.len() {
            return out;
        }
        let own = self.own_rank(precursor_mz);
        let near =
            (own + EXCLUDE_WINDOWS).min(num_windows - 1) + 1 - own.saturating_sub(EXCLUDE_WINDOWS);
        let num_far = num_windows - near;
        if num_far == 0 {
            return out;
        }
        let segment = &self.keys[self.cycle_start[cycle]..self.cycle_start[cycle + 1]];
        let mut seen = [0u64; BITSET_WORDS];
        for (u, &mz) in out.iter_mut().zip(fragment_mz) {
            if mz <= 0.0 {
                continue;
            }
            let delta = mz * tolerance_ppm * 1e-6;
            let mut bins = mz_index.mz_range_indices(mz - delta, mz + delta);
            let Some(first_bin) = bins.next() else {
                continue;
            };
            let end_bin = bins.last().unwrap_or(first_bin) + 1;
            let lo = (first_bin as u32) << RANK_BITS;
            let hi = (end_bin as u32) << RANK_BITS;
            let first = segment.partition_point(|&k| k < lo);
            seen.fill(0);
            let mut hits = 0u32;
            for &key in segment[first..].iter().take_while(|&&k| k < hi) {
                let rank = (key & RANK_MASK) as usize;
                if rank.abs_diff(own) <= EXCLUDE_WINDOWS {
                    continue;
                }
                let (word, bit) = (rank / 64, 1u64 << (rank % 64));
                if seen[word] & bit == 0 {
                    seen[word] |= bit;
                    hits += 1;
                }
            }
            *u = hits as f32 / num_far as f32;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dia_data::AlphaRawView;
    use crate::dia_data_builder::DIADataBuilder;
    use numpy::ndarray::{ArrayView1, ArrayView4};

    const NUM_WINDOWS: usize = 10;
    const NUM_CYCLES: usize = 2;
    const FRAGMENT: f32 = 500.0;

    /// Ten 10 Th windows from 400 Th over two cycles. Cycle 0: the fragment m/z in windows 0, 1, 4 and 8 (window 0
    /// twice within the tolerance, window 3 at zero intensity); cycle 1: in window 9 only.
    fn data() -> crate::dia_data::DIAData {
        let mut peaks: Vec<Vec<(f32, f32)>> = vec![Vec::new(); NUM_WINDOWS * NUM_CYCLES];
        for w in [0, 1, 4, 8] {
            peaks[w].push((FRAGMENT, 100.0));
        }
        peaks[0].push((FRAGMENT * (1.0 + 3e-6), 50.0));
        peaks[3].push((FRAGMENT, 0.0));
        peaks[NUM_WINDOWS + 9].push((FRAGMENT, 100.0));
        for p in peaks.iter_mut() {
            p.push((300.0, 10.0));
        }

        let mut delta = Vec::new();
        let mut lower = Vec::new();
        let mut upper = Vec::new();
        let mut start = Vec::new();
        let mut stop = Vec::new();
        let mut cycle = Vec::new();
        let mut rt = Vec::new();
        let mut mz = Vec::new();
        let mut intensity = Vec::new();
        for c in 0..NUM_CYCLES {
            for w in 0..NUM_WINDOWS {
                delta.push(w as i64);
                lower.push(400.0 + 10.0 * w as f32);
                upper.push(410.0 + 10.0 * w as f32);
                start.push(mz.len() as i64);
                let mut spectrum = peaks[c * NUM_WINDOWS + w].clone();
                spectrum.sort_by(|a, b| a.0.total_cmp(&b.0));
                for (m, i) in spectrum {
                    mz.push(m);
                    intensity.push(i);
                }
                stop.push(mz.len() as i64);
                cycle.push(c as i64);
                rt.push(c as f32 + 0.01 * w as f32);
            }
        }
        let cycle_data = [0.0f32];
        let view = AlphaRawView {
            spectrum_delta_scan_idx: ArrayView1::from(&delta),
            isolation_lower_mz: ArrayView1::from(&lower),
            isolation_upper_mz: ArrayView1::from(&upper),
            spectrum_peak_start_idx: ArrayView1::from(&start),
            spectrum_peak_stop_idx: ArrayView1::from(&stop),
            spectrum_cycle_idx: ArrayView1::from(&cycle),
            spectrum_rt: ArrayView1::from(&rt),
            peak_mz: ArrayView1::from(&mz),
            peak_intensity: ArrayView1::from(&intensity),
            cycle: ArrayView4::from_shape([1, 1, 1, 1], &cycle_data[..]).unwrap(),
        };
        DIADataBuilder::from_alpha_raw(&view)
    }

    #[test]
    fn ubiquity_counts_distinct_far_windows_with_signal() {
        let dia_data = data();
        let index = UbiquityIndex::build(&dia_data).unwrap();
        let mz_index = MZIndex::global();
        // own window 5: windows 2..=8 are near, 0, 1 and 9 far; 0 and 1 hold the fragment
        let u = index.fragment_ubiquity(mz_index, 0, 455.0, &[FRAGMENT, 600.0, 0.0], 7.0);
        assert!((u[0] - 2.0 / 3.0).abs() < 1e-6);
        assert_eq!(u[1], 0.0);
        assert_eq!(u[2], 0.0);
        let u1 = index.fragment_ubiquity(mz_index, 1, 455.0, &[FRAGMENT], 7.0);
        assert!((u1[0] - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn ubiquity_skips_near_windows_and_zero_intensity() {
        let dia_data = data();
        let index = UbiquityIndex::build(&dia_data).unwrap();
        let mz_index = MZIndex::global();
        // own window 0: windows 4..=9 are far, 4 and 8 hold the fragment (window 3 only at zero intensity)
        let u = index.fragment_ubiquity(mz_index, 0, 401.0, &[FRAGMENT], 7.0);
        assert!((u[0] - 2.0 / 6.0).abs() < 1e-6);
        // a tolerance that misses the m/z bins of the fragment
        let narrow = index.fragment_ubiquity(mz_index, 0, 401.0, &[FRAGMENT * (1.0 + 20e-6)], 1.0);
        assert_eq!(narrow[0], 0.0);
    }

    #[test]
    fn cycles_outside_the_run_give_zero() {
        let dia_data = data();
        let index = UbiquityIndex::build(&dia_data).unwrap();
        let u = index.fragment_ubiquity(MZIndex::global(), 5, 455.0, &[FRAGMENT], 7.0);
        assert_eq!(u, vec![0.0]);
    }
}
