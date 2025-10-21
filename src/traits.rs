use crate::dense_xic_observation::{DenseXICMZObservation, DenseXICObservation};
use crate::mz_index::MZIndex;
use crate::rt_index::RTIndex;

/// Core trait for DIA (Data-Independent Acquisition) data structures.
///
/// This trait provides a clean, abstract interface for DIA data access without exposing
/// internal implementation details. All XIC extraction logic is encapsulated within the
/// trait methods, allowing different implementations to use completely different internal
/// storage formats.
///
/// # Required Methods
///
/// * `get_dense_xic_observation` - Extracts intensity XICs for fragments
/// * `get_dense_xic_mz_observation` - Extracts both intensity and m/z XICs
/// * `mz_index` - Returns the global m/z index for mass range queries
/// * `rt_index` - Returns the retention time index for temporal queries
///
pub trait DIADataTrait {
    /// Extract dense XIC (intensity only) for given precursor and fragments.
    ///
    /// # Parameters
    ///
    /// * `precursor_mz` - Precursor m/z to find relevant isolation windows
    /// * `cycle_start_idx` - Start cycle index (inclusive)
    /// * `cycle_stop_idx` - End cycle index (exclusive)
    /// * `mass_tolerance` - Mass tolerance in ppm for fragment extraction
    /// * `fragment_mz` - Slice of fragment m/z values to extract
    ///
    /// # Returns
    ///
    /// A `DenseXICObservation` containing the extracted intensity matrix
    fn get_dense_xic_observation(
        &self,
        precursor_mz: f32,
        cycle_start_idx: usize,
        cycle_stop_idx: usize,
        mass_tolerance: f32,
        fragment_mz: &[f32],
    ) -> DenseXICObservation;

    /// Extract dense XIC with m/z tracking for given precursor and fragments.
    ///
    /// # Parameters
    ///
    /// * `precursor_mz` - Precursor m/z to find relevant isolation windows
    /// * `cycle_start_idx` - Start cycle index (inclusive)
    /// * `cycle_stop_idx` - End cycle index (exclusive)
    /// * `mass_tolerance` - Mass tolerance in ppm for fragment extraction
    /// * `fragment_mz` - Slice of fragment m/z values to extract
    ///
    /// # Returns
    ///
    /// A `DenseXICMZObservation` containing both intensity and m/z matrices
    fn get_dense_xic_mz_observation(
        &self,
        precursor_mz: f32,
        cycle_start_idx: usize,
        cycle_stop_idx: usize,
        mass_tolerance: f32,
        fragment_mz: &[f32],
    ) -> DenseXICMZObservation;

    /// Returns the global m/z index for mass range queries.
    fn mz_index(&self) -> &MZIndex;

    /// Returns the retention time index for temporal queries.
    fn rt_index(&self) -> &RTIndex;
}
