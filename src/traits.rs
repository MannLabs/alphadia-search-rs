use crate::dense_xic_observation::{DenseXICMZObservation, DenseXICObservation};
use crate::mz_index::MZIndex;
use crate::rt_index::RTIndex;

/// Core trait for DIA (Data-Independent Acquisition) data structures.
///
/// This trait is the single abstraction that `peak_group_selection`, `peak_group_scoring`
/// and `peak_group_quantification` are generic over — none of them ever see a concrete
/// `DIAData`. All XIC extraction logic lives behind `hydrate_xic`/`hydrate_xic_mz`, so a
/// backend is free to use a completely different internal storage layout (e.g. a future
/// `TimsDIAData` with a native ion-mobility dimension) as long as it can answer these
/// four questions.
///
/// # Required Methods
///
/// * `hydrate_xic` - Extracts intensity XICs for fragments
/// * `hydrate_xic_mz` - Extracts both intensity and m/z XICs
/// * `mz_index` - Returns the global m/z index for mass range queries
/// * `rt_index` - Returns the retention time index for temporal queries
///
/// # Extending for ion mobility
///
/// `DIAData::has_mobility()` and `mobility_values()` are placeholders today — there is no
/// mobility-aware backend yet. When one lands (e.g. `TimsDIAData`), prefer adding it as a
/// second implementor of this same trait over branching on an enum: every consumer here is
/// already generic (`T: DIADataTrait`), so a new implementor slots in with zero changes to
/// `peak_group_selection`/`scoring`/`quantification`. The extraction methods would grow an
/// `Option<MobilityFilter>` parameter and a `Result` return, so that a mobility-free backend
/// (like today's `DIAData`) can ignore it and always return `Ok`, while a mobility-aware one
/// can error out when the filter it structurally requires wasn't supplied.
pub trait DIADataTrait {
    /// Extract a dense XIC (intensity only) for a precursor's fragments.
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
    fn hydrate_xic(
        &self,
        precursor_mz: f32,
        cycle_start_idx: usize,
        cycle_stop_idx: usize,
        mass_tolerance: f32,
        fragment_mz: &[f32],
    ) -> DenseXICObservation;

    /// Extract a dense XIC with m/z tracking for a precursor's fragments.
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
    fn hydrate_xic_mz(
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
