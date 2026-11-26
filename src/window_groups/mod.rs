//! One rule for "which isolation window is this precursor in", used by XIC extraction, the window-share features
//! and fragment competition.
//!
//! Terms:
//! - **quadrupole observation**: all spectra of one scan position in the DIA cycle (`delta_scan_idx`).
//! - **isolation window**: the m/z range `[lower, upper)` of a quadrupole observation. The upper limit is exclusive.
//! - **window group**: the set of quadrupole observations whose isolation windows contain a precursor m/z.
//!
//! The window limits cut the m/z axis into segments. All m/z of one segment have the same window group, and
//! segments with the same set of quadrupole observations share one group. Every m/z therefore has exactly one
//! window group, so the groups partition the precursors. Group 0 is the empty set: m/z outside all isolation
//! windows.
//!
//! Without overlap, every window group is one isolation window. With overlap, the overlap is a group of its own:
//!
//! ```text
//!   isolation windows    0: [400, 425)    1: [424, 450)
//!
//!   m/z →     400          424  425                 450
//!   segment    │    {0}     │{0,1}│        {1}         │
//!   group      │     1      │  2  │         3          │     (0 outside)
//! ```
//!
//! A precursor at 424.5 is read from both quadrupole observations, so it is compared only with other precursors
//! that are read from both.

use std::collections::HashMap;

#[cfg(test)]
mod tests;

/// The window group of every m/z, built once from the isolation windows of a run.
#[derive(Debug, Clone)]
pub struct WindowGroups {
    /// the distinct window limits, ascending
    edges: Vec<f32>,
    /// window group of each segment `(-inf, edges[0])`, `[edges[0], edges[1])`, ..., `[edges[last], inf)`
    segment_groups: Vec<u32>,
    /// the quadrupole observations of each window group, ascending; group 0 is empty
    groups: Vec<Vec<usize>>,
}

/// Window group of an m/z outside all isolation windows.
pub const NO_WINDOW_GROUP: u32 = 0;

impl Default for WindowGroups {
    fn default() -> Self {
        Self::new(&[])
    }
}

impl WindowGroups {
    /// `windows[i]` is the isolation window `[lower, upper)` of quadrupole observation `i`. A window with
    /// `lower >= upper` or a NaN limit (for example an MS1 scan) contains no m/z.
    pub fn new(windows: &[[f32; 2]]) -> Self {
        let is_valid = |window: &[f32; 2]| window[0] < window[1];

        let mut edges: Vec<f32> = windows
            .iter()
            .filter(|window| is_valid(window))
            .flat_map(|window| [window[0], window[1]])
            .collect();
        edges.sort_by(f32::total_cmp);
        edges.dedup();

        let mut groups: Vec<Vec<usize>> = vec![Vec::new()];
        let mut group_ids: HashMap<Vec<usize>, u32> =
            HashMap::from([(Vec::new(), NO_WINDOW_GROUP)]);
        let mut segment_groups = Vec::with_capacity(edges.len() + 1);

        for segment in 0..=edges.len() {
            // the outer segments lie outside all windows
            let members: Vec<usize> = if segment == 0 || segment == edges.len() {
                Vec::new()
            } else {
                let (lower, upper) = (edges[segment - 1], edges[segment]);
                windows
                    .iter()
                    .enumerate()
                    .filter(|(_, window)| {
                        is_valid(window) && window[0] <= lower && upper <= window[1]
                    })
                    .map(|(idx, _)| idx)
                    .collect()
            };
            let group = match group_ids.get(&members) {
                Some(&group) => group,
                None => {
                    let group = groups.len() as u32;
                    group_ids.insert(members.clone(), group);
                    groups.push(members);
                    group
                }
            };
            segment_groups.push(group);
        }

        Self {
            edges,
            segment_groups,
            groups,
        }
    }

    /// The window group of `mz`; `NO_WINDOW_GROUP` outside all isolation windows and for NaN.
    pub fn group_of(&self, mz: f32) -> u32 {
        let segment = self.edges.partition_point(|&edge| edge <= mz);
        self.segment_groups[segment]
    }

    /// The quadrupole observations of a window group, ascending.
    pub fn quadrupole_observations(&self, group: u32) -> &[usize] {
        &self.groups[group as usize]
    }

    /// The quadrupole observations whose isolation windows contain `mz`, ascending.
    pub fn quadrupole_observations_of(&self, mz: f32) -> &[usize] {
        self.quadrupole_observations(self.group_of(mz))
    }

    /// Number of window groups, `NO_WINDOW_GROUP` included.
    pub fn num_groups(&self) -> usize {
        self.groups.len()
    }
}
