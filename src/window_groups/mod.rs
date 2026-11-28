//! One rule for "which isolation window is this precursor in", shared by the window-share features and fragment
//! competition.
//!
//! A precursor's **window group** is the set of isolation windows `[lower, upper)` that contain its m/z. These are
//! the quadrupole observations its XICs are read from. Every m/z has exactly one group. Without overlap, a group is
//! one isolation window; an overlap is a group of its own, and a nested window splits its outer window into three
//! ranges, of which the outer two share one group:
//!
//! ```text
//!   overlap   0: [400, 425)  1: [424, 450)      nested   0: [400, 450)  1: [420, 430)
//!   m/z →     400     424  425      450                   400   420   430   450
//!   group      │  {0}  │{0,1}│  {1}  │                     │ {0} │{0,1}│ {0} │
//! ```

use std::collections::HashMap;

#[cfg(test)]
mod tests;

pub struct WindowGroups {
    /// the distinct window limits, ascending
    limits: Vec<f32>,
    /// window group of each range between two neighbouring limits, and of the two ranges outside them
    range_groups: Vec<u32>,
}

impl WindowGroups {
    /// `windows[i]` is isolation window `i`. A window with `lower >= upper` or a NaN limit (an MS1 scan) contains
    /// nothing.
    pub fn new(windows: &[[f32; 2]]) -> Self {
        let windows: Vec<(usize, [f32; 2])> = windows
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, [lower, upper])| lower < upper)
            .collect();
        let mut limits: Vec<f32> = windows.iter().flat_map(|(_, window)| *window).collect();
        limits.sort_by(f32::total_cmp);
        limits.dedup();

        // the same set of windows is the same group; the empty set (outside all windows) is group 0
        let mut group_ids = HashMap::from([(Vec::new(), 0)]);
        let range_groups = (0..=limits.len())
            .map(|range| {
                let members: Vec<usize> = match (range.checked_sub(1), limits.get(range)) {
                    (Some(start), Some(&stop)) => windows
                        .iter()
                        .filter(|(_, [lower, upper])| *lower <= limits[start] && stop <= *upper)
                        .map(|&(idx, _)| idx)
                        .collect(),
                    _ => Vec::new(),
                };
                let next_id = group_ids.len() as u32;
                *group_ids.entry(members).or_insert(next_id)
            })
            .collect();

        Self {
            limits,
            range_groups,
        }
    }

    /// The window group of `mz`; 0 outside all windows and for NaN.
    pub fn group_of(&self, mz: f32) -> u32 {
        self.range_groups[self.limits.partition_point(|&limit| limit <= mz)]
    }
}
