use crate::isotope_envelope::ChannelSet;
use crate::traits::{DIADataTrait, QuadrupoleObservationTrait};
use numpy::ndarray::Array2;

/// Encapsulates a dense XIC matrix with metadata about its construction
///
/// Each fragment row is the sum of the fragment's extraction channels.
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
        channels: &ChannelSet,
        cycle_start_idx: usize,
        cycle_stop_idx: usize,
        mass_tolerance: f32,
    ) -> Self {
        let num_fragments = channels.fragment_gain.len();
        let mut dense_xic = Array2::zeros((num_fragments, cycle_stop_idx - cycle_start_idx));

        for channel in &channels.channels {
            dia_data.quadrupole_observations()[channel.observation_idx].fill_xic_slice(
                dia_data.mz_index(),
                &mut dense_xic.row_mut(channel.fragment_idx),
                cycle_start_idx,
                cycle_stop_idx,
                mass_tolerance,
                channel.mz,
            );
        }

        Self {
            dense_xic,
            contributing_obs_indices: channels.observation_indices(),
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
        channels: &ChannelSet,
        cycle_start_idx: usize,
        cycle_stop_idx: usize,
        mass_tolerance: f32,
    ) -> Self {
        let n_cycles = cycle_stop_idx - cycle_start_idx;
        let num_fragments = channels.fragment_gain.len();
        let mut dense_xic = Array2::zeros((num_fragments, n_cycles));
        let mut dense_mz = Array2::zeros((num_fragments, n_cycles));

        for channel in &channels.channels {
            dia_data.quadrupole_observations()[channel.observation_idx].fill_xic_and_mz_slice(
                dia_data.mz_index(),
                &mut dense_xic.row_mut(channel.fragment_idx),
                &mut dense_mz.row_mut(channel.fragment_idx),
                cycle_start_idx,
                cycle_stop_idx,
                mass_tolerance,
                channel.mz,
                channel.mz_shift,
            );
        }

        Self {
            dense_xic,
            dense_mz,
            contributing_obs_indices: channels.observation_indices(),
            cycle_start_idx,
            cycle_stop_idx,
            mass_tolerance,
        }
    }
}

#[cfg(test)]
mod tests;
