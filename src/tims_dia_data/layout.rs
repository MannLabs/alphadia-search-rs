//! Which frames of a diaPASEF run belong to which cycle and window group, and
//! which precursor m/z each (window group, scan) isolated.
//!
//! A cycle starts at every MS1 frame, like alphaDIA's Thermo cycle: one pass over
//! all isolation windows. Each window group becomes one quadrupole observation.

use timsrust::core::{Converter, MSLevel, QuadrupoleSettings, ScanIndex};
use timsrust::tdf::TdfFrameReader;
use timsrust::TimsTofPath;

/// An MS2 frame placed in the run's cycles.
pub(super) struct Ms2Frame {
    pub frame_index: usize,
    pub cycle: u16,
    /// Window group, numbered in order of first appearance.
    pub group: usize,
    pub rt_seconds: f32,
}

/// What `assign_cycles` needs to know about a frame.
pub(super) struct FrameMeta {
    pub frame_index: usize,
    pub is_ms1: bool,
    /// Quadrupole settings id of an MS2 frame.
    pub settings_id: usize,
    pub rt_seconds: f32,
}

/// Frames grouped into cycles.
pub(super) struct Cycles {
    /// RT of the MS1 frame of every cycle.
    pub cycle_rt_seconds: Vec<f32>,
    pub ms2_frames: Vec<Ms2Frame>,
    /// Quadrupole settings id of every window group.
    pub settings_ids: Vec<usize>,
}

/// Assigns frames, sorted by frame index, to cycles and window groups. MS2 frames before
/// the first MS1 frame have no cycle and are left out.
pub(super) fn assign_cycles(frames: impl IntoIterator<Item = FrameMeta>) -> Result<Cycles, String> {
    let mut cycles = Cycles {
        cycle_rt_seconds: Vec::new(),
        ms2_frames: Vec::new(),
        settings_ids: Vec::new(),
    };
    for frame in frames {
        if frame.is_ms1 {
            cycles.cycle_rt_seconds.push(frame.rt_seconds);
            continue;
        }
        let Some(last_cycle) = cycles.cycle_rt_seconds.len().checked_sub(1) else {
            continue;
        };
        let cycle = u16::try_from(last_cycle)
            .map_err(|_| "runs with more than 65,536 cycles are not supported")?;
        let group = match cycles
            .settings_ids
            .iter()
            .position(|id| *id == frame.settings_id)
        {
            Some(group) => group,
            None => {
                cycles.settings_ids.push(frame.settings_id);
                cycles.settings_ids.len() - 1
            }
        };
        cycles.ms2_frames.push(Ms2Frame {
            frame_index: frame.frame_index,
            cycle,
            group,
            rt_seconds: frame.rt_seconds,
        });
    }
    if cycles.cycle_rt_seconds.is_empty() || cycles.settings_ids.is_empty() {
        return Err("no MS1 cycles or no diaPASEF window groups found".into());
    }
    Ok(cycles)
}

pub(super) struct RunLayout {
    /// RT in seconds of each frame, in cycle order: slot 0 is the MS1 frame,
    /// slot `g + 1` the frame of window group `g`. Length `num_cycles * cycle_len`.
    pub frame_rt_seconds: Vec<f32>,
    pub ms2_frames: Vec<Ms2Frame>,
    /// Per window group, the isolation window of every scan, `None` where the quadrupole was not set.
    pub scan_windows: Vec<Vec<Option<[f32; 2]>>>,
    /// 1/K0 of every scan, decreasing with the scan index.
    pub scan_mobility: Vec<f32>,
}

impl RunLayout {
    pub fn read(
        path: &TimsTofPath,
        reader: &TdfFrameReader,
        num_scans: usize,
    ) -> Result<Self, String> {
        let im = path.im_converter().ok_or("no ion mobility converter")?;
        let scan_mobility = (0..num_scans)
            .map(|s| {
                let scan = ScanIndex::try_from(s as u32).map_err(|e| e.to_string())?;
                Ok(f64::from(im.convert(scan)) as f32)
            })
            .collect::<Result<Vec<_>, String>>()?;

        // Frame ids come from a HashMap, so their order is arbitrary.
        let mut indices: Vec<usize> = reader.iter_indices().collect();
        indices.sort_unstable();

        let mut frames = Vec::with_capacity(indices.len());
        let mut settings_by_id = std::collections::HashMap::new();
        for index in indices {
            let info = reader.get_info(index).map_err(|e| e.to_string())?;
            let is_ms1 = match info.ms_level() {
                MSLevel::MS1 => true,
                MSLevel::MS2 => false,
                _ => continue,
            };
            let settings = info.quadrupole_settings();
            if !is_ms1 {
                settings_by_id
                    .entry(settings.index)
                    .or_insert_with(|| settings.clone());
            }
            frames.push(FrameMeta {
                frame_index: index,
                is_ms1,
                settings_id: settings.index,
                rt_seconds: info.rt_in_seconds() as f32,
            });
        }
        let cycles = assign_cycles(frames)?;

        let scan_windows = cycles
            .settings_ids
            .iter()
            .map(|id| window_per_scan(&settings_by_id[id], num_scans))
            .collect::<Vec<_>>();
        if scan_windows
            .iter()
            .any(|windows| windows.iter().all(Option::is_none))
        {
            return Err("MS2 frames without diaPASEF isolation windows, not a diaPASEF run".into());
        }

        Ok(Self {
            frame_rt_seconds: frame_rt_seconds(&cycles),
            ms2_frames: cycles.ms2_frames,
            scan_windows,
            scan_mobility,
        })
    }

    pub fn cycle_len(&self) -> usize {
        1 + self.scan_windows.len()
    }

    /// RT of the MS1 frame of every cycle.
    pub fn cycle_rt_seconds(&self) -> Vec<f32> {
        self.frame_rt_seconds
            .iter()
            .step_by(self.cycle_len())
            .copied()
            .collect()
    }
}

/// RT of every frame slot in cycle order; a group missing from a cycle gets the MS1 RT.
pub(super) fn frame_rt_seconds(cycles: &Cycles) -> Vec<f32> {
    let cycle_len = 1 + cycles.settings_ids.len();
    let mut rt = Vec::with_capacity(cycles.cycle_rt_seconds.len() * cycle_len);
    for cycle_rt in &cycles.cycle_rt_seconds {
        rt.extend(std::iter::repeat_n(*cycle_rt, cycle_len));
    }
    for frame in &cycles.ms2_frames {
        rt[frame.cycle as usize * cycle_len + frame.group + 1] = frame.rt_seconds;
    }
    rt
}

/// The isolation window of every scan; `scan_ends` are exclusive, as in Bruker's
/// `DiaFrameMsMsWindows` where one window's end is the next window's start.
pub(super) fn window_per_scan(
    settings: &QuadrupoleSettings,
    num_scans: usize,
) -> Vec<Option<[f32; 2]>> {
    let mut windows = vec![None; num_scans];
    for ((start, end), window) in settings
        .scan_starts
        .iter()
        .zip(&settings.scan_ends)
        .zip(&settings.isolation_windows)
    {
        let bounds = [
            f64::from(window.lower()) as f32,
            f64::from(window.upper()) as f32,
        ];
        for slot in windows.iter_mut().take((*end).min(num_scans)).skip(*start) {
            *slot = Some(bounds);
        }
    }
    windows
}
