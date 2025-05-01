use numpy::ndarray::Array1;

use crate::AlphaRawView;

pub struct RTIndex {
    pub rt: Array1<f32>,
}

impl RTIndex {
    pub fn new() -> Self {
        Self {
            rt: Array1::from_vec(Vec::new()),
        }
    }

    pub fn from_alpha_raw(alpha_raw_view: &AlphaRawView) -> Self {
        let mut rt = Vec::new();

        for i in 0..alpha_raw_view.spectrum_delta_scan_idx.len() {
            if alpha_raw_view.spectrum_delta_scan_idx[i] == 0 {
                rt.push(alpha_raw_view.spectrum_rt[i]);
            }
        }

        Self {
            rt: Array1::from_vec(rt),
        }
    }
} 