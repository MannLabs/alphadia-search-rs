//! Ion mobility of a candidate's fragment peaks at its apex, for data with an ion mobility dimension (timsTOF).
//!
//! The fragments of a real precursor are released from one ion cloud, so their peaks share its mobility: their
//! intensity-weighted mean mobilities agree with each other and with the library. Fragments of an interfering
//! precursor, which the mobility window lets through when it elutes nearby, sit at another mobility and widen the
//! between-fragment spread. The observed mobility also feeds the mobility calibration.
//!
//! Data without a mobility dimension, and candidates without fragment signal at the apex, get `missing()`: all values 0
//! apart from `mobility_observed`, which is the library mobility then (0 without one).

use crate::candidate::Candidate;
use crate::peak_group_scoring::feature_block::feature_block;
use crate::precursor::Precursor;
use crate::traits::{DIADataTrait, MobilityFilter};

/// The mobility is read from the cycles within this many cycles of the candidate's apex.
pub const MOBILITY_APEX_RADIUS: usize = 1;

feature_block! {
    /// The ion mobility features of one candidate.
    pub struct MobilityFeatures {
        /// intensity-weighted mean 1/K0 of the fragment peaks at the apex (for the mobility calibration, not a
        /// classifier feature)
        pub mobility_observed: f32,
        /// `mobility_observed` minus the library (or calibrated) mobility
        pub mobility_error: f32,
        /// intensity-weighted SD of the fragments' own mean mobilities around `mobility_observed`
        pub mobility_fragment_sd: f32,
        /// intensity-weighted SD of the peaks around their fragment's mean mobility: the mobility peak width
        pub mobility_peak_sd: f32,
        /// share of the apex fragment intensity outside half the mobility tolerance of the library mobility
        pub mobility_outer_frac: f32,
    }
}

impl MobilityFeatures {
    fn missing(library_mobility: f32) -> Self {
        Self {
            mobility_observed: if library_mobility.is_nan() {
                0.0
            } else {
                library_mobility
            },
            ..Self::default()
        }
    }
}

/// The cycles around a candidate's apex the mobility is read from.
fn apex_cycles<T: DIADataTrait>(dia_data: &T, cycle_center: usize) -> (usize, usize) {
    let num_cycles = dia_data.rt_index().rt.len();
    (
        cycle_center.saturating_sub(MOBILITY_APEX_RADIUS),
        (cycle_center + MOBILITY_APEX_RADIUS + 1).min(num_cycles),
    )
}

/// The window of `half_width` around the intensity-weighted mean mobility of the precursor's fragment peaks within
/// `wide`, at the candidate's apex: a mobility peak is much narrower than the tolerance around the library mobility,
/// so the XICs of the narrow window carry less background. `wide` itself without a mobility dimension, for a
/// `half_width` that is not positive and finite, or without fragment signal at the apex.
pub fn recentered_mobility_filter<T: DIADataTrait>(
    dia_data: &T,
    precursor: &Precursor,
    cycle_center: usize,
    wide: MobilityFilter,
    mass_tolerance: f32,
    half_width: f32,
) -> MobilityFilter {
    if !(half_width > 0.0 && half_width.is_finite()) {
        return wide;
    }
    let (start, stop) = apex_cycles(dia_data, cycle_center);
    let observations = dia_data.select_observations(precursor.mz, wide);
    let mut sums = [0.0f64; 2];
    for &mz in &precursor.fragment_mz {
        let Some(moments) = dia_data.mobility_moments(
            &observations,
            start,
            stop,
            mass_tolerance,
            mz,
            f32::NAN,
            0.0,
        ) else {
            return wide;
        };
        sums[0] += moments.sums[0];
        sums[1] += moments.sums[1];
    }
    if sums[0] <= 0.0 {
        return wide;
    }
    MobilityFilter::Window {
        mobility: (sums[1] / sums[0]) as f32,
        tolerance: half_width,
    }
}

/// The mobility features of one candidate over the observations of `filter`, the window around its library mobility.
pub fn calculate_mobility_features<T: DIADataTrait>(
    dia_data: &T,
    precursor: &Precursor,
    candidate: &Candidate,
    filter: MobilityFilter,
    mass_tolerance: f32,
    mobility_tolerance: f32,
) -> MobilityFeatures {
    let (start, stop) = apex_cycles(dia_data, candidate.cycle_center);
    let observations = dia_data.select_observations(precursor.mz, filter);

    // per fragment: [Σw, Σw·m, Σw·m²]
    let mut fragments: Vec<[f64; 3]> = Vec::with_capacity(precursor.fragment_mz.len());
    let mut outer = 0.0f64;
    let half_tolerance = 0.5 * mobility_tolerance as f64;
    for &mz in &precursor.fragment_mz {
        let Some(moments) = dia_data.mobility_moments(
            &observations,
            start,
            stop,
            mass_tolerance,
            mz,
            precursor.mobility,
            half_tolerance as f32,
        ) else {
            return MobilityFeatures::missing(precursor.mobility);
        };
        if moments.sums[0] > 0.0 {
            fragments.push(moments.sums);
            outer += moments.outer_weight;
        }
    }

    let total: f64 = fragments.iter().map(|m| m[0]).sum();
    if total <= 0.0 {
        return MobilityFeatures::missing(precursor.mobility);
    }
    let observed = fragments.iter().map(|m| m[1]).sum::<f64>() / total;
    let between = fragments
        .iter()
        .map(|m| m[0] * (m[1] / m[0] - observed).powi(2))
        .sum::<f64>()
        / total;
    let within = fragments
        .iter()
        .map(|m| (m[2] - m[1] * m[1] / m[0]).max(0.0))
        .sum::<f64>()
        / total;

    let library = precursor.mobility;
    MobilityFeatures {
        mobility_observed: observed as f32,
        mobility_error: if library.is_nan() {
            0.0
        } else {
            observed as f32 - library
        },
        mobility_fragment_sd: between.sqrt() as f32,
        mobility_peak_sd: within.sqrt() as f32,
        mobility_outer_frac: if library.is_nan() {
            0.0
        } else {
            (outer / total) as f32
        },
    }
}
