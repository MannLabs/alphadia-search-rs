use crate::traits::DIADataTrait;
use numpy::ndarray::Array2;

/// A dense extracted ion chromatogram (XIC) with metadata about its construction.
///
/// This structure provides an efficient representation of fragment ion intensities
/// across DIA cycles for a given precursor.
///
/// # Fields
///
/// * `dense_xic` - 2D intensity matrix with shape `[n_fragments, n_cycles]`
/// * `contributing_obs_indices` - Indices of quadrupole observations used in construction. If dia data has isolation windows `[(500,520),(520,540),(540,560)]`, and precursor is 530.0, then the contributing observations are `[1]`.
/// * `cycle_start_idx` - Start of the dia cycles that were used to construct the XIC (inclusive)
/// * `cycle_stop_idx` - End of the dia cycles that were used to construct the XIC (exclusive)
/// * `mass_tolerance` - Mass tolerance in ppm used for fragment extraction
///
/// ```
pub struct DenseXICObservation {
    pub dense_xic: Array2<f32>,
    #[allow(dead_code)]
    pub contributing_obs_indices: Vec<usize>,
    #[allow(dead_code)]
    pub cycle_start_idx: usize,
    #[allow(dead_code)]
    pub cycle_stop_idx: usize,
    #[allow(dead_code)]
    pub mass_tolerance: f32,
}

impl DenseXICObservation {
    /// Create a new DenseXICObservation from DIA data and parameters.
    ///
    /// This is a convenience factory method that delegates to the trait implementation.
    ///
    /// # Parameters
    ///
    /// * `dia_data` - DIA data source implementing [`DIADataTrait`]
    /// * `precursor_mz` - Precursor m/z to find relevant isolation windows
    /// * `cycle_start_idx` - Start cycle index (inclusive)
    /// * `cycle_stop_idx` - End cycle index (exclusive)
    /// * `mass_tolerance` - Mass tolerance in ppm for fragment extraction
    /// * `fragment_mz` - Slice of fragment m/z values to extract
    #[inline]
    pub fn new<T: DIADataTrait>(
        dia_data: &T,
        precursor_mz: f32,
        cycle_start_idx: usize,
        cycle_stop_idx: usize,
        mass_tolerance: f32,
        fragment_mz: &[f32],
    ) -> Self {
        dia_data.get_dense_xic_observation(
            precursor_mz,
            cycle_start_idx,
            cycle_stop_idx,
            mass_tolerance,
            fragment_mz,
        )
    }
}

/// A dense XIC and m/z matrix pair with construction metadata.
///
/// This structure extends [`DenseXICObservation`] by also tracking the actual
/// measured m/z values for each intensity data point, enabling mass accuracy
/// analysis and quality control.
///
/// # Fields
///
/// * `dense_xic` - 2D intensity matrix with shape `[n_fragments, n_cycles]`
/// * `dense_mz` - 2D m/z matrix with shape `[n_fragments, n_cycles]` containing measured m/z values
/// * `contributing_obs_indices` - Indices of quadrupole observations used in construction. If dia data has isolation windows `[(500,520),(520,540),(540,560)]`, and precursor is 530.0, then the contributing observations are `[1]`.
/// * `cycle_start_idx` - Start of the dia cycles that were used to construct the XIC (inclusive)
/// * `cycle_stop_idx` - End of the dia cycles that were used to construct the XIC (exclusive)
/// * `mass_tolerance` - Mass tolerance in ppm used for fragment extraction
///
/// # Use Cases
///
/// - Mass error analysis and calibration
/// - Isotope pattern verification
/// - Quality control metrics
/// - Advanced scoring methods that incorporate mass accuracy
///
/// # Example
///
/// ```ignore
/// let obs = DenseXICMZObservation::new(
///     &dia_data,
///     precursor_mz,
///     cycle_start_idx,
///     cycle_stop_idx,
///     mass_tolerance,
///     &fragment_mz,
/// );
/// ```
#[allow(dead_code)]
pub struct DenseXICMZObservation {
    pub dense_xic: Array2<f32>,
    pub dense_mz: Array2<f32>,
    #[allow(dead_code)]
    pub contributing_obs_indices: Vec<usize>,
    #[allow(dead_code)]
    pub cycle_start_idx: usize,
    #[allow(dead_code)]
    pub cycle_stop_idx: usize,
    #[allow(dead_code)]
    pub mass_tolerance: f32,
}

impl DenseXICMZObservation {
    /// Create a new DenseXICMZObservation from DIA data and parameters.
    ///
    /// This is a convenience factory method that delegates to the trait implementation.
    ///
    /// # Parameters
    ///
    /// * `dia_data` - DIA data source implementing [`DIADataTrait`]
    /// * `precursor_mz` - Precursor m/z to find relevant isolation windows
    /// * `cycle_start_idx` - Start cycle index (inclusive)
    /// * `cycle_stop_idx` - End cycle index (exclusive)
    /// * `mass_tolerance` - Mass tolerance in ppm for fragment extraction
    /// * `fragment_mz` - Slice of fragment m/z values to extract
    #[inline]
    #[allow(dead_code)]
    pub fn new<T: DIADataTrait>(
        dia_data: &T,
        precursor_mz: f32,
        cycle_start_idx: usize,
        cycle_stop_idx: usize,
        mass_tolerance: f32,
        fragment_mz: &[f32],
    ) -> Self {
        dia_data.get_dense_xic_mz_observation(
            precursor_mz,
            cycle_start_idx,
            cycle_stop_idx,
            mass_tolerance,
            fragment_mz,
        )
    }
}

#[cfg(test)]
mod tests;
