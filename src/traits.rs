use crate::mz_index::MZIndex;
use crate::rt_index::RTIndex;
use numpy::ndarray::ArrayViewMut1;

/// Default for the `mobility_tolerance` parameter of selection, scoring and quantification.
pub const DEFAULT_MOBILITY_TOLERANCE: f32 = 0.04;

/// Checks a `mobility_tolerance` parameter: not NaN and not negative (infinite means no
/// mobility restriction).
pub fn check_mobility_tolerance(tolerance: f32) -> Result<f32, String> {
    if tolerance.is_nan() || tolerance < 0.0 {
        Err(format!(
            "mobility_tolerance must not be NaN or negative, got {tolerance}"
        ))
    } else {
        Ok(tolerance)
    }
}

/// Ion mobility (1/K0) of a precursor and how far from it a peak may lie.
///
/// Data without an ion mobility dimension ignores it. Data with one restricts peaks to
/// `mobility ± tolerance` only when `mobility` is finite and positive and `tolerance` is
/// finite and not negative; otherwise (e.g. NaN for a library without mobility) it applies no
/// mobility restriction.
#[derive(Clone, Copy, Debug)]
pub struct MobilityFilter {
    pub mobility: f32,
    pub tolerance: f32,
}

impl MobilityFilter {
    /// No mobility restriction: every scan that isolated the precursor counts.
    pub const ANY: Self = Self {
        mobility: f32::NAN,
        tolerance: f32::INFINITY,
    };
}

/// The scan restriction of one quadrupole observation for one precursor.
pub type ScanFilterOf<T> =
    <<T as DIADataTrait>::QuadrupoleObservation as QuadrupoleObservationTrait>::ScanFilter;

/// Trait for DIA data structures that support peak group scoring
pub trait DIADataTrait {
    type QuadrupoleObservation: QuadrupoleObservationTrait;

    /// Observations that isolated the precursor, each with the scan restriction to apply while filling XICs.
    fn select_observations(
        &self,
        precursor_mz: f32,
        mobility: MobilityFilter,
    ) -> Vec<(usize, ScanFilterOf<Self>)>;

    fn mz_index(&self) -> &MZIndex;
    fn rt_index(&self) -> &RTIndex;
    fn quadrupole_observations(&self) -> &[Self::QuadrupoleObservation];

    // Common functionality that both implementations have
    fn num_observations(&self) -> usize {
        self.quadrupole_observations().len()
    }

    fn memory_footprint_bytes(&self) -> usize;

    fn memory_footprint_mb(&self) -> f64 {
        self.memory_footprint_bytes() as f64 / (1024.0 * 1024.0)
    }
}

/// Trait for quadrupole observation types that support XIC slice filling
pub trait QuadrupoleObservationTrait {
    /// Which peaks of the observation belong to the precursor, `()` when all of them do.
    type ScanFilter;

    fn fill_xic_slice(
        &self,
        mz_index: &MZIndex,
        scan_filter: &Self::ScanFilter,
        dense_xic: &mut ArrayViewMut1<f32>,
        cycle_start_idx: usize,
        cycle_stop_idx: usize,
        mass_tolerance: f32,
        mz: f32,
    );

    #[allow(clippy::too_many_arguments)]
    fn fill_xic_and_mz_slice(
        &self,
        mz_index: &MZIndex,
        scan_filter: &Self::ScanFilter,
        dense_xic: &mut ArrayViewMut1<f32>,
        dense_mz: &mut ArrayViewMut1<f32>,
        cycle_start_idx: usize,
        cycle_stop_idx: usize,
        mass_tolerance: f32,
        mz: f32,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mobility_tolerance_must_not_be_nan_or_negative() {
        assert_eq!(check_mobility_tolerance(0.0), Ok(0.0));
        assert_eq!(check_mobility_tolerance(f32::INFINITY), Ok(f32::INFINITY));
        assert!(check_mobility_tolerance(-0.01).is_err());
        assert!(check_mobility_tolerance(f32::NAN).is_err());
    }
}
