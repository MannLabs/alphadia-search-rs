//! Evidence from the library fragments ranked just below the scored top fragments ("next ions").
//!
//! Scoring uses the top-k library fragments. A faint real peptide at low input still shows some of its next most
//! intense b/y ions at the apex scan, a chance match does not: in offline replays the count of next ions hit at the
//! apex scan added information beyond the detection LLR of the top fragments, symmetric between decoys and false
//! targets, while the weaker rest of the ladder added nothing.
//!
//! The count is compared with a null on the same scan: the same m/z set shifted by whole multiples of the peptide
//! mass-grid spacing (the probes stay on the grid where other peptides' fragments sit and clear of the ions' own
//! isotopes), so a busy window does not read as evidence.

/// Spacing of the peptide fragment mass grid (Th per nominal mass unit at charge 1).
pub const GRID_SPACING: f32 = 1.000495;
/// Grid steps of the null probes: +-3 .. +-12, clear of isotopes (+-1, +-2).
pub const SHIFT_STEPS: [i32; 20] = [
    -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
];
/// Added to the null variance so that a candidate whose null is never hit does not get an unbounded z.
const VARIANCE_OFFSET: f32 = 1.0;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct NextIonFeatures {
    /// next ions with signal at the apex scan
    pub next_ion_hits: f32,
    /// (next_ion_hits - null mean) / sqrt(null variance + VARIANCE_OFFSET) over the shifted probe sets
    pub next_ion_hits_z: f32,
    /// next ions the library lists for the precursor
    pub next_ion_count: f32,
}

impl NextIonFeatures {
    /// Feature values for a precursor without next ions.
    pub fn missing() -> Self {
        Self::default()
    }
}

/// The next ions' library m/z moved by the calibration of the precursor's top fragments: each takes the ppm offset
/// (extraction m/z vs library m/z) of the top fragment closest in library m/z.
pub fn calibrate(next_mz_library: &[f32], top_mz: &[f32], top_mz_library: &[f32]) -> Vec<f32> {
    next_mz_library
        .iter()
        .map(|&mz| {
            let nearest = top_mz_library
                .iter()
                .enumerate()
                .filter(|(_, &library)| library > 0.0)
                .min_by(|a, b| (a.1 - mz).abs().total_cmp(&(b.1 - mz).abs()));
            match nearest {
                Some((i, &library)) => mz * top_mz[i] / library,
                None => mz,
            }
        })
        .collect()
}

/// The m/z to extract: the next ions, then one shifted copy per entry of SHIFT_STEPS.
pub fn probe_mz(next_mz: &[f32]) -> Vec<f32> {
    let mut probes = next_mz.to_vec();
    for &step in SHIFT_STEPS.iter() {
        let shift = step as f32 * GRID_SPACING;
        probes.extend(next_mz.iter().map(|&mz| mz + shift));
    }
    probes
}

/// Features from the apex-scan intensities of `probe_mz(next_mz)` (`num_next` ions per set).
pub fn next_ion_features(apex_intensity: &[f32], num_next: usize) -> NextIonFeatures {
    let num_sets = SHIFT_STEPS.len() + 1;
    if num_next == 0 || apex_intensity.len() != num_next * num_sets {
        return NextIonFeatures::missing();
    }
    let hits: Vec<f32> = apex_intensity
        .chunks(num_next)
        .map(|set| set.iter().filter(|&&x| x > 0.0).count() as f32)
        .collect();
    let observed = hits[0];
    let null = &hits[1..];
    let mean = null.iter().sum::<f32>() / null.len() as f32;
    let variance = null.iter().map(|&h| (h - mean).powi(2)).sum::<f32>() / null.len() as f32;
    NextIonFeatures {
        next_ion_hits: observed,
        next_ion_hits_z: (observed - mean) / (variance + VARIANCE_OFFSET).sqrt(),
        next_ion_count: num_next as f32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibration_takes_the_nearest_top_fragment_offset() {
        let top_library = [300.0, 800.0];
        let top = [300.0 * (1.0 + 2e-6), 800.0 * (1.0 - 3e-6)];
        let out = calibrate(&[310.0, 790.0], &top, &top_library);
        assert!(((out[0] / 310.0 - 1.0) * 1e6 - 2.0).abs() < 0.3);
        assert!(((out[1] / 790.0 - 1.0) * 1e6 + 3.0).abs() < 0.3);
        assert_eq!(calibrate(&[500.0], &[], &[]), vec![500.0]);
    }

    #[test]
    fn probes_hold_the_ions_then_every_shifted_copy() {
        let probes = probe_mz(&[400.0, 600.0]);
        assert_eq!(probes.len(), 2 * (SHIFT_STEPS.len() + 1));
        assert_eq!(&probes[..2], &[400.0, 600.0]);
        assert!((probes[2] - (400.0 - 12.0 * GRID_SPACING)).abs() < 1e-3);
        assert!((probes[3] - (600.0 - 12.0 * GRID_SPACING)).abs() < 1e-3);
    }

    #[test]
    fn hits_above_a_quiet_null_give_a_positive_z() {
        let num_sets = SHIFT_STEPS.len() + 1;
        let mut intensity = vec![0.0f32; 3 * num_sets];
        intensity[0] = 10.0;
        intensity[2] = 5.0;
        let f = next_ion_features(&intensity, 3);
        assert_eq!(f.next_ion_hits, 2.0);
        assert_eq!(f.next_ion_count, 3.0);
        assert!((f.next_ion_hits_z - 2.0).abs() < 1e-6);
    }

    #[test]
    fn a_busy_null_lowers_the_z() {
        let num_sets = SHIFT_STEPS.len() + 1;
        let mut busy = vec![1.0f32; 3 * num_sets];
        busy[1] = 0.0;
        let f = next_ion_features(&busy, 3);
        assert_eq!(f.next_ion_hits, 2.0);
        assert!(f.next_ion_hits_z < 0.0);
    }

    #[test]
    fn no_next_ions_or_wrong_length_give_missing_values() {
        assert_eq!(next_ion_features(&[], 0), NextIonFeatures::missing());
        assert_eq!(
            next_ion_features(&[1.0, 2.0], 2),
            NextIonFeatures::missing()
        );
    }
}
