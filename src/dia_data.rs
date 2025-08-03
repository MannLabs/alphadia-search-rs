use numpy::ndarray::{ArrayBase, Dim, ViewRepr};

pub struct AlphaRawView<'py> {
    pub spectrum_delta_scan_idx: ArrayBase<ViewRepr<&'py i64>, Dim<[usize; 1]>>,
    pub isolation_lower_mz: ArrayBase<ViewRepr<&'py f32>, Dim<[usize; 1]>>,
    pub isolation_upper_mz: ArrayBase<ViewRepr<&'py f32>, Dim<[usize; 1]>>,
    pub spectrum_peak_start_idx: ArrayBase<ViewRepr<&'py i64>, Dim<[usize; 1]>>,
    pub spectrum_peak_stop_idx: ArrayBase<ViewRepr<&'py i64>, Dim<[usize; 1]>>,
    pub spectrum_cycle_idx: ArrayBase<ViewRepr<&'py i64>, Dim<[usize; 1]>>,
    pub spectrum_rt: ArrayBase<ViewRepr<&'py f32>, Dim<[usize; 1]>>,
    pub peak_mz: ArrayBase<ViewRepr<&'py f32>, Dim<[usize; 1]>>,
    pub peak_intensity: ArrayBase<ViewRepr<&'py f32>, Dim<[usize; 1]>>,
}

impl<'py> AlphaRawView<'py> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        spectrum_delta_scan_idx: ArrayBase<ViewRepr<&'py i64>, Dim<[usize; 1]>>,
        isolation_lower_mz: ArrayBase<ViewRepr<&'py f32>, Dim<[usize; 1]>>,
        isolation_upper_mz: ArrayBase<ViewRepr<&'py f32>, Dim<[usize; 1]>>,
        spectrum_peak_start_idx: ArrayBase<ViewRepr<&'py i64>, Dim<[usize; 1]>>,
        spectrum_peak_stop_idx: ArrayBase<ViewRepr<&'py i64>, Dim<[usize; 1]>>,
        spectrum_cycle_idx: ArrayBase<ViewRepr<&'py i64>, Dim<[usize; 1]>>,
        spectrum_rt: ArrayBase<ViewRepr<&'py f32>, Dim<[usize; 1]>>,
        peak_mz: ArrayBase<ViewRepr<&'py f32>, Dim<[usize; 1]>>,
        peak_intensity: ArrayBase<ViewRepr<&'py f32>, Dim<[usize; 1]>>,
    ) -> Self {
        Self {
            spectrum_delta_scan_idx,
            isolation_lower_mz,
            isolation_upper_mz,
            spectrum_peak_start_idx,
            spectrum_peak_stop_idx,
            spectrum_cycle_idx,
            spectrum_rt,
            peak_mz,
            peak_intensity,
        }
    }
}
