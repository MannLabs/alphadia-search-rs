use crate::mz_index::MZIndex;
use crate::traits::QuadrupoleObservationTrait;
use numpy::ndarray::ArrayViewMut1;

/// One diaPASEF window group: peaks bucketed on the global m/z index like
/// `QuadrupoleObservation`, each slice sorted by cycle, plus the scan of every peak.
pub struct TimsQuadrupoleObservation {
    /// Lowest and highest m/z isolated by any scan of the group.
    pub(crate) isolation_window: [f32; 2],

    /// Isolation window of every scan, `None` where the quadrupole was not set.
    pub(crate) scan_windows: Vec<Option<[f32; 2]>>,

    /// The peaks of m/z index `i` are `slice_starts[i]..slice_starts[i + 1]`.
    pub(crate) slice_starts: Vec<u32>,
    pub(crate) cycle_indices: Vec<u16>,
    pub(crate) scan_indices: Vec<u16>,
    pub(crate) intensities: Vec<f32>,
}

impl TimsQuadrupoleObservation {
    pub fn new(scan_windows: Vec<Option<[f32; 2]>>) -> Self {
        let isolation_window = scan_windows
            .iter()
            .flatten()
            .fold([f32::INFINITY, f32::NEG_INFINITY], |acc, w| {
                [acc[0].min(w[0]), acc[1].max(w[1])]
            });
        Self {
            isolation_window,
            scan_windows,
            slice_starts: Vec::new(),
            cycle_indices: Vec::new(),
            scan_indices: Vec::new(),
            intensities: Vec::new(),
        }
    }

    pub fn isolates(&self, scan: usize, precursor_mz: f32) -> bool {
        matches!(self.scan_windows[scan], Some([lo, hi]) if lo <= precursor_mz && precursor_mz <= hi)
    }

    /// Scans in `scans` that isolated the precursor, as a mask over all scans; `None` if there are none.
    pub fn scan_mask(&self, precursor_mz: f32, scans: std::ops::Range<usize>) -> Option<Vec<bool>> {
        if precursor_mz < self.isolation_window[0] || precursor_mz > self.isolation_window[1] {
            return None;
        }
        let mut mask = vec![false; self.scan_windows.len()];
        let mut any = false;
        for scan in scans {
            if self.isolates(scan, precursor_mz) {
                mask[scan] = true;
                any = true;
            }
        }
        any.then_some(mask)
    }

    pub fn num_peaks(&self) -> usize {
        self.intensities.len()
    }

    pub fn memory_footprint_bytes(&self) -> usize {
        self.slice_starts.len() * std::mem::size_of::<u32>()
            + self.num_peaks() * (std::mem::size_of::<u16>() * 2 + std::mem::size_of::<f32>())
            + self.scan_windows.len() * std::mem::size_of::<Option<[f32; 2]>>()
    }

    /// Calls `add(cycle - cycle_start, intensity, mz, scan)` for every peak within `mz ± mass_tolerance` ppm,
    /// in `cycle_start..cycle_stop` and on a scan set in `scan_mask`.
    #[inline]
    pub fn for_each_peak(
        &self,
        mz_index: &MZIndex,
        scan_mask: &[bool],
        cycle_start_idx: usize,
        cycle_stop_idx: usize,
        mass_tolerance: f32,
        mz: f32,
        mut add: impl FnMut(usize, f32, f32, usize),
    ) {
        let delta_mz = mz * mass_tolerance * 1e-6;
        for mz_idx in mz_index.mz_range_indices(mz - delta_mz, mz + delta_mz) {
            let start = self.slice_starts[mz_idx] as usize;
            let stop = self.slice_starts[mz_idx + 1] as usize;
            let cycle_indices = &self.cycle_indices[start..stop];
            let first = cycle_indices.partition_point(|c| (*c as usize) < cycle_start_idx);
            for i in first..cycle_indices.len() {
                let cycle_idx = cycle_indices[i] as usize;
                if cycle_idx >= cycle_stop_idx {
                    break;
                }
                let peak = start + i;
                let scan = self.scan_indices[peak] as usize;
                if scan_mask[scan] {
                    add(
                        cycle_idx - cycle_start_idx,
                        self.intensities[peak],
                        mz_index.mz[mz_idx],
                        scan,
                    );
                }
            }
        }
    }
}

impl QuadrupoleObservationTrait for TimsQuadrupoleObservation {
    /// Mask over scans: whether a peak on that scan belongs to the precursor.
    type ScanFilter = Vec<bool>;

    fn fill_xic_slice(
        &self,
        mz_index: &MZIndex,
        scan_filter: &Vec<bool>,
        dense_xic: &mut ArrayViewMut1<f32>,
        cycle_start_idx: usize,
        cycle_stop_idx: usize,
        mass_tolerance: f32,
        mz: f32,
    ) {
        self.for_each_peak(
            mz_index,
            scan_filter,
            cycle_start_idx,
            cycle_stop_idx,
            mass_tolerance,
            mz,
            |idx, intensity, _, _| dense_xic[idx] += intensity,
        );
    }

    fn fill_xic_and_mz_slice(
        &self,
        mz_index: &MZIndex,
        scan_filter: &Vec<bool>,
        dense_xic: &mut ArrayViewMut1<f32>,
        dense_mz: &mut ArrayViewMut1<f32>,
        cycle_start_idx: usize,
        cycle_stop_idx: usize,
        mass_tolerance: f32,
        mz: f32,
    ) {
        // Same running intensity-weighted mean as QuadrupoleObservation.
        self.for_each_peak(
            mz_index,
            scan_filter,
            cycle_start_idx,
            cycle_stop_idx,
            mass_tolerance,
            mz,
            |idx, intensity, actual_mz, _| {
                dense_xic[idx] += intensity;
                if intensity > 0.0 {
                    let previous = dense_xic[idx] - intensity;
                    dense_mz[idx] = if previous == 0.0 {
                        actual_mz
                    } else {
                        (dense_mz[idx] * previous + actual_mz * intensity) / dense_xic[idx]
                    };
                }
            },
        );
    }
}
