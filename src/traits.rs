use crate::mz_index::MZIndex;
use crate::rt_index::RTIndex;
use numpy::ndarray::ArrayViewMut1;

/// Which ion mobility (1/K0) range a precursor's peaks may come from.
///
/// Data without an ion mobility dimension ignores it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MobilityFilter {
    /// No mobility restriction: every scan that isolated the precursor counts.
    Any,
    /// Only peaks whose scan lies within `mobility ± tolerance`.
    Window { mobility: f32, tolerance: f32 },
}

impl MobilityFilter {
    /// The filter for a library mobility: `Any` when the library has none (NaN) or the tolerance
    /// is infinite, otherwise a window of `tolerance` around it.
    ///
    /// A NaN or negative tolerance, or a mobility that is infinite or not positive, is an error
    /// rather than a silently dropped restriction.
    pub fn from_library(mobility: f32, tolerance: f32) -> Result<Self, String> {
        if tolerance.is_nan() || tolerance < 0.0 {
            return Err(format!(
                "mobility tolerance must not be NaN or negative, got {tolerance}"
            ));
        }
        if mobility.is_nan() {
            return Ok(Self::Any);
        }
        if !(mobility.is_finite() && mobility > 0.0) {
            return Err(format!(
                "mobility must be finite and positive, got {mobility}"
            ));
        }
        if tolerance == f32::INFINITY {
            return Ok(Self::Any);
        }
        Ok(Self::Window {
            mobility,
            tolerance,
        })
    }
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
    use super::MobilityFilter;

    #[test]
    fn test_from_library_without_mobility_is_any() {
        assert_eq!(
            MobilityFilter::from_library(f32::NAN, 0.05),
            Ok(MobilityFilter::Any)
        );
    }

    #[test]
    fn test_from_library_makes_a_window() {
        assert_eq!(
            MobilityFilter::from_library(1.0, 0.05),
            Ok(MobilityFilter::Window {
                mobility: 1.0,
                tolerance: 0.05
            })
        );
        assert!(MobilityFilter::from_library(1.0, 0.0).is_ok());
    }

    #[test]
    fn test_from_library_with_an_infinite_tolerance_is_any() {
        assert_eq!(
            MobilityFilter::from_library(1.0, f32::INFINITY),
            Ok(MobilityFilter::Any)
        );
    }

    #[test]
    fn test_from_library_rejects_a_bad_tolerance_even_without_mobility() {
        for tolerance in [-0.05, f32::NAN] {
            assert!(MobilityFilter::from_library(1.0, tolerance).is_err());
            assert!(MobilityFilter::from_library(f32::NAN, tolerance).is_err());
        }
    }

    #[test]
    fn test_from_library_rejects_a_bad_mobility() {
        for mobility in [0.0, -1.0, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(MobilityFilter::from_library(mobility, 0.05).is_err());
        }
    }
}
