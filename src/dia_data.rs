use pyo3::prelude::*;
use numpy::ndarray::{ArrayBase, ViewRepr, Dim};
use numpy::PyReadonlyArray1;
use rayon::prelude::*;
use crate::xic_index::MZIndex;
use crate::quadrupole_observation::QuadrupoleObservation;
use crate::AlphaRawView;

#[pyclass]
pub struct DIAData {
    mz_index: MZIndex,
    //rt_index: RTIndex,
    quadrupole_observations: Vec<QuadrupoleObservation>,
}

#[pymethods]
impl DIAData {
    #[new]
    fn new() -> Self {
        Self {
            mz_index: MZIndex::new(),
            quadrupole_observations: Vec::new(),
        }
    }

    #[getter]
    fn num_observations(&self) -> usize {
        self.quadrupole_observations.len()
    }

    #[getter]
    fn num_peaks(&self) -> usize {
        self.quadrupole_observations.iter().map(|q| q.xic_slices.iter().map(|x| x.cycle_index.len()).sum::<usize>()).sum()
    }

    fn from_alpha_raw<'py>(
        &mut self,
        spectrum_delta_scan_idx: PyReadonlyArray1<'py, i64>,
        isolation_lower_mz: PyReadonlyArray1<'py, f32>,
        isolation_upper_mz: PyReadonlyArray1<'py, f32>,
        spectrum_peak_start_idx: PyReadonlyArray1<'py, i64>,
        spectrum_peak_stop_idx: PyReadonlyArray1<'py, i64>,
        spectrum_cycle_idx: PyReadonlyArray1<'py, i64>,
        peak_mz: PyReadonlyArray1<'py, f32>,
        peak_intensity: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<()> {
        let alpha_raw_view = AlphaRawView::new(
            spectrum_delta_scan_idx.as_array(),
            isolation_lower_mz.as_array(),
            isolation_upper_mz.as_array(),
            spectrum_peak_start_idx.as_array(),
            spectrum_peak_stop_idx.as_array(),
            spectrum_cycle_idx.as_array(),
            peak_mz.as_array(),
            peak_intensity.as_array(),
        );

        let dia_data = Self::from_alpha_raw_internal(&alpha_raw_view);
        self.mz_index = dia_data.mz_index;
        self.quadrupole_observations = dia_data.quadrupole_observations;
        Ok(())
    }
}

impl DIAData {
    pub fn from_alpha_raw_internal(alpha_raw_view: &AlphaRawView) -> Self {
        let mz_index = MZIndex::new();
        let num_quadrupole_observations = alpha_raw_view.spectrum_delta_scan_idx.iter().max().unwrap() + 1;
        
        // Parallel iteration over quadrupole observations
        let quadrupole_observations: Vec<QuadrupoleObservation> = (0..num_quadrupole_observations)
            .into_par_iter()
            .map(|i| QuadrupoleObservation::from_alpha_raw(&alpha_raw_view, i, &mz_index))
            .collect();
        
        Self {
            mz_index,
            quadrupole_observations,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::Array1;
    use numpy::ndarray::array;

    #[test]
    fn test_dia_data_new() {
        let dia_data = DIAData::new();
        assert_eq!(dia_data.num_observations(), 0);
        assert_eq!(dia_data.num_peaks(), 0);
    }

    #[test]
    fn test_dia_data_from_alpha_raw() {
        // Create test data
        let spectrum_delta_scan_idx = array![0, 0, 1, 1];
        let isolation_lower_mz = array![100.0, 100.0, 200.0, 200.0];
        let isolation_upper_mz = array![200.0, 200.0, 300.0, 300.0];
        let spectrum_peak_start_idx = array![0, 2, 0, 2];
        let spectrum_peak_stop_idx = array![2, 4, 2, 4];
        let spectrum_cycle_idx = array![0, 0, 1, 1];
        let peak_mz = array![150.0, 250.0, 150.0, 250.0];
        let peak_intensity = array![1.0, 2.0, 3.0, 4.0];

        let alpha_raw_view = AlphaRawView::new(
            spectrum_delta_scan_idx.view(),
            isolation_lower_mz.view(),
            isolation_upper_mz.view(),
            spectrum_peak_start_idx.view(),
            spectrum_peak_stop_idx.view(),
            spectrum_cycle_idx.view(),
            peak_mz.view(),
            peak_intensity.view(),
        );

        let dia_data = DIAData::from_alpha_raw_internal(&alpha_raw_view);
        assert_eq!(dia_data.num_observations(), 2); // Two unique delta scan indices
    }
} 