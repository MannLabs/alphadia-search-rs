use rayon::prelude::*;
use timsrust::core::{Converter, TofIndex};
use timsrust::tdf::TdfFrameReader;
use timsrust::{MzConverter, TimsTofPath};

use super::layout::{Ms2Frame, RunLayout};
use super::observation::TimsQuadrupoleObservation;
use crate::mz_index::{MZIndex, MZ_END, MZ_START};

/// TOF indices below this are binned through a lookup table, the rest one by one.
const TOF_TABLE_LEN: u32 = 1 << 19;
const OUT_OF_RANGE: u32 = u32::MAX;

/// Maps a TOF index to its closest bin on the global m/z index.
struct TofBinner<'a> {
    mz_converter: &'a MzConverter,
    mz_index: &'a MZIndex,
    table: Vec<u32>,
}

impl<'a> TofBinner<'a> {
    fn new(mz_converter: &'a MzConverter, mz_index: &'a MZIndex) -> Self {
        let mut binner = Self {
            mz_converter,
            mz_index,
            table: Vec::new(),
        };
        binner.table = (0..TOF_TABLE_LEN)
            .into_par_iter()
            .map(|tof| binner.convert(tof))
            .collect();
        binner
    }

    fn convert(&self, tof: u32) -> u32 {
        let Ok(tof) = TofIndex::try_from(tof) else {
            return OUT_OF_RANGE;
        };
        let mz = f64::from(self.mz_converter.convert(tof)) as f32;
        if !(MZ_START..=MZ_END).contains(&mz) {
            return OUT_OF_RANGE;
        }
        self.mz_index.find_closest_index(mz) as u32
    }

    fn bin(&self, tof: u32) -> u32 {
        match self.table.get(tof as usize) {
            Some(bin) => *bin,
            None => self.convert(tof),
        }
    }
}

/// One peak before sorting: (m/z bin, cycle, scan, intensity).
type Staged = (u32, u16, u16, f32);

/// Builds one observation per window group, one group at a time, so only that
/// group's staging buffer (12 bytes per peak) is alive next to the finished ones.
/// Peaks that land on the same (m/z bin, cycle, scan) are summed.
pub fn build_observations(
    path: &TimsTofPath,
    reader: &TdfFrameReader,
    layout: &RunLayout,
) -> Result<Vec<TimsQuadrupoleObservation>, String> {
    let mz_converter = path.mz_converter().ok_or("no m/z converter")?;
    let mz_index = MZIndex::global();
    let binner = TofBinner::new(&mz_converter, mz_index);
    layout
        .scan_windows
        .iter()
        .enumerate()
        .map(|(group, scan_windows)| {
            let mut obs = TimsQuadrupoleObservation::new(scan_windows.clone());
            let staged = stage_frames(
                reader,
                &frames_of(layout, group),
                &binner,
                scan_windows.len(),
            )?;
            fill_group(&mut obs, staged, mz_index.len())?;
            Ok(obs)
        })
        .collect()
}

fn frames_of(layout: &RunLayout, group: usize) -> Vec<&Ms2Frame> {
    layout
        .ms2_frames
        .iter()
        .filter(|f| f.group == group)
        .collect()
}

/// Sorts the staged peaks of one window group by (m/z bin, cycle) into `obs`.
pub(super) fn fill_group(
    obs: &mut TimsQuadrupoleObservation,
    per_frame: Vec<Vec<Staged>>,
    num_bins: usize,
) -> Result<(), String> {
    let mut staged: Vec<Staged> = Vec::with_capacity(per_frame.iter().map(Vec::len).sum::<usize>());
    for frame in per_frame {
        staged.extend(frame);
    }
    // scan and intensity in the key keep the order, and so the f32 sums, reproducible
    staged.par_sort_unstable_by_key(|p| (p.0, p.1, p.2, p.3.to_bits()));
    merge_duplicates(&mut staged);
    u32::try_from(staged.len()).map_err(|_| "more than 4e9 peaks in one window group")?;

    let mut slice_starts = vec![0u32; num_bins + 1];
    for p in &staged {
        slice_starts[p.0 as usize + 1] += 1;
    }
    for bin in 0..num_bins {
        slice_starts[bin + 1] += slice_starts[bin];
    }

    obs.slice_starts = slice_starts;
    obs.cycle_indices = staged.iter().map(|p| p.1).collect();
    obs.scan_indices = staged.iter().map(|p| p.2).collect();
    obs.intensities = staged.iter().map(|p| p.3).collect();
    Ok(())
}

/// Sums sorted peaks on the same (m/z bin, cycle, scan) in place.
fn merge_duplicates(sorted: &mut Vec<Staged>) {
    sorted.dedup_by(|peak, kept| {
        let same = (peak.0, peak.1, peak.2) == (kept.0, kept.1, kept.2);
        if same {
            kept.3 += peak.3;
        }
        same
    });
    sorted.shrink_to_fit();
}

/// All raw peaks of every frame. The scan count is taken from the first frame of the run;
/// peaks on scans beyond it have no isolation window and are left out.
fn stage_frames(
    reader: &TdfFrameReader,
    frames: &[&Ms2Frame],
    binner: &TofBinner,
    num_scans: usize,
) -> Result<Vec<Vec<Staged>>, String> {
    frames
        .par_iter()
        .map(|frame| {
            let ions = reader
                .get_ions(frame.frame_index)
                .map_err(|e| e.to_string())?;
            let mut staged = Vec::with_capacity(ions.tof_indices().len());
            for scan in 0..ions.scan_count().min(num_scans) {
                for (tof, intensity) in ions.read_scan(scan) {
                    let bin = binner.bin(u32::from(tof));
                    if bin != OUT_OF_RANGE {
                        staged.push((bin, frame.cycle, scan as u16, u32::from(intensity) as f32));
                    }
                }
            }
            Ok(staged)
        })
        .collect()
}
