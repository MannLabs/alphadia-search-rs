use crate::mz_index::MZIndex;
use crate::rt_index::RTIndex;
use numpy::ndarray::ArrayViewMut1;

/// Trait for DIA data structures that support peak group scoring
pub trait DIADataTrait {
    type QuadrupoleObservation: QuadrupoleObservationTrait;

    fn get_valid_observations(&self, precursor_mz: f32) -> Vec<usize>;
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
    /// Lower and upper m/z of the isolation window
    fn isolation_window(&self) -> [f32; 2];

    fn fill_xic_slice(
        &self,
        mz_index: &MZIndex,
        dense_xic: &mut ArrayViewMut1<f32>,
        cycle_start_idx: usize,
        cycle_stop_idx: usize,
        mass_tolerance: f32,
        mz: f32,
    );

    /// As `fill_xic_slice`, and keeps the intensity-weighted mean of the observed m/z minus `mz_shift` in `dense_mz`.
    #[allow(clippy::too_many_arguments)]
    fn fill_xic_and_mz_slice(
        &self,
        mz_index: &MZIndex,
        dense_xic: &mut ArrayViewMut1<f32>,
        dense_mz: &mut ArrayViewMut1<f32>,
        cycle_start_idx: usize,
        cycle_stop_idx: usize,
        mass_tolerance: f32,
        mz: f32,
        mz_shift: f32,
    );
}
