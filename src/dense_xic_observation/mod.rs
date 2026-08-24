use numpy::ndarray::Array2;

/// A dense extracted ion chromatogram (XIC) with metadata about its construction.
///
/// This structure provides an efficient representation of fragment ion intensities
/// across DIA cycles for a given precursor. Built via
/// [`DIADataTrait::hydrate_xic`](crate::traits::DIADataTrait::hydrate_xic).
///
/// # Fields
///
/// * `dense_xic` - 2D intensity matrix with shape `[n_fragments, n_cycles]`
/// * `contributing_obs_indices` - 0-indexed positions into the DIA data's quadrupole
///   observations that overlap the precursor's isolation window. E.g. for isolation windows
///   `[(500,520), (520,540), (540,560)]` and precursor m/z 530.0, the only overlapping window
///   is the second one, so this is `[1]`.
/// * `cycle_start_idx` - Start of the dia cycles that were used to construct the XIC (inclusive)
/// * `cycle_stop_idx` - End of the dia cycles that were used to construct the XIC (exclusive)
/// * `mass_tolerance` - Mass tolerance in ppm used for fragment extraction
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

/// A dense XIC and m/z matrix pair with construction metadata.
///
/// This structure extends [`DenseXICObservation`] by also tracking the actual
/// measured m/z values for each intensity data point, enabling mass accuracy
/// analysis and quality control. Built via
/// [`DIADataTrait::hydrate_xic_mz`](crate::traits::DIADataTrait::hydrate_xic_mz).
///
/// # Fields
///
/// * `dense_xic` - 2D intensity matrix with shape `[n_fragments, n_cycles]`
/// * `dense_mz` - 2D m/z matrix with shape `[n_fragments, n_cycles]` containing measured m/z values
/// * `contributing_obs_indices` - 0-indexed positions into the DIA data's quadrupole
///   observations that overlap the precursor's isolation window. E.g. for isolation windows
///   `[(500,520), (520,540), (540,560)]` and precursor m/z 530.0, the only overlapping window
///   is the second one, so this is `[1]`.
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

#[cfg(test)]
mod tests;
