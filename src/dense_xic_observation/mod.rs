use crate::traits::{DIADataTrait, QuadrupoleObservationTrait};
use numpy::ndarray::Array2;

/// Encapsulates a dense XIC matrix with metadata about its construction
pub struct DenseXICObservation {
    /// Dense XIC matrix: [fragment_index, cycle_index] -> intensity
    pub dense_xic: Array2<f32>,

    /// Indices of observations that contributed to this dense XIC
    #[allow(dead_code)] // Metadata for debugging/analysis
    pub contributing_obs_indices: Vec<usize>,

    /// Cycle range metadata
    #[allow(dead_code)] // Metadata for debugging/analysis
    pub cycle_start_idx: usize,
    #[allow(dead_code)] // Metadata for debugging/analysis
    pub cycle_stop_idx: usize,

    /// Mass tolerance used for extraction
    #[allow(dead_code)] // Metadata for debugging/analysis
    pub mass_tolerance: f32,
}

impl DenseXICObservation {
    /// Create a new DenseXICObservation from DIA data and parameters
    ///
    /// This constructor pattern allows for zero-cost abstractions and full
    /// compiler optimization through monomorphization.
    #[inline]
    pub fn new<T: DIADataTrait>(
        dia_data: &T,
        precursor_mz: f32,
        cycle_start_idx: usize,
        cycle_stop_idx: usize,
        mass_tolerance: f32,
        fragment_mz: &[f32], // Use slice for better performance
    ) -> Self {
        let mut dense_xic = Array2::zeros((fragment_mz.len(), cycle_stop_idx - cycle_start_idx));

        let valid_obs_idxs = dia_data.get_valid_observations(precursor_mz);

        for &obs_idx in &valid_obs_idxs {
            let obs = &dia_data.quadrupole_observations()[obs_idx];

            for (f_idx, &f_mz) in fragment_mz.iter().enumerate() {
                obs.fill_xic_slice(
                    dia_data.mz_index(),
                    &mut dense_xic.row_mut(f_idx),
                    cycle_start_idx,
                    cycle_stop_idx,
                    mass_tolerance,
                    f_mz,
                );
            }
        }

        Self {
            dense_xic,
            contributing_obs_indices: valid_obs_idxs,
            cycle_start_idx,
            cycle_stop_idx,
            mass_tolerance,
        }
    }
}

/// Encapsulates dense XIC and m/z matrices with metadata about their construction
#[allow(dead_code)]
pub struct DenseXICMZObservation {
    /// Dense XIC matrix: [fragment_index, cycle_index] -> intensity
    pub dense_xic: Array2<f32>,

    /// Dense m/z matrix: [fragment_index, cycle_index] -> m/z
    pub dense_mz: Array2<f32>,

    /// Indices of observations that contributed to this dense XIC
    #[allow(dead_code)] // Metadata for debugging/analysis
    pub contributing_obs_indices: Vec<usize>,

    /// Cycle range metadata
    #[allow(dead_code)] // Metadata for debugging/analysis
    pub cycle_start_idx: usize,
    #[allow(dead_code)] // Metadata for debugging/analysis
    pub cycle_stop_idx: usize,

    /// Mass tolerance used for extraction
    #[allow(dead_code)] // Metadata for debugging/analysis
    pub mass_tolerance: f32,
}

impl DenseXICMZObservation {
    /// Create a new DenseXICMZObservation from DIA data and parameters
    ///
    /// This constructor pattern allows for zero-cost abstractions and full
    /// compiler optimization through monomorphization.
    #[inline]
    #[allow(dead_code)]
    pub fn new<T: DIADataTrait>(
        dia_data: &T,
        precursor_mz: f32,
        cycle_start_idx: usize,
        cycle_stop_idx: usize,
        mass_tolerance: f32,
        fragment_mz: &[f32], // Use slice for better performance
    ) -> Self {
        let n_fragments = fragment_mz.len();
        let n_cycles = cycle_stop_idx - cycle_start_idx;
        let mut dense_xic = Array2::zeros((n_fragments, n_cycles));
        let mut dense_mz = Array2::zeros((n_fragments, n_cycles));

        let valid_obs_idxs = dia_data.get_valid_observations(precursor_mz);

        for &obs_idx in &valid_obs_idxs {
            let obs = &dia_data.quadrupole_observations()[obs_idx];

            for (f_idx, &f_mz) in fragment_mz.iter().enumerate() {
                obs.fill_xic_and_mz_slice(
                    dia_data.mz_index(),
                    &mut dense_xic.row_mut(f_idx),
                    &mut dense_mz.row_mut(f_idx),
                    cycle_start_idx,
                    cycle_stop_idx,
                    mass_tolerance,
                    f_mz,
                );
            }
        }

        Self {
            dense_xic,
            dense_mz,
            contributing_obs_indices: valid_obs_idxs,
            cycle_start_idx,
            cycle_stop_idx,
            mass_tolerance,
        }
    }
}

#[cfg(test)]
mod tests;
