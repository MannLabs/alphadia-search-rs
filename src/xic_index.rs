use numpy::ndarray::Array1;

pub fn ppm_index(resolution_ppm: f32, mz_start: f32, mz_end: f32) -> Array1<f32> {
    let mz_start_safe = mz_start.max(50.0);
    
    let mut index: Vec<f32> = Vec::from([mz_start_safe]);
    let mut current_mz = mz_start_safe;

    while current_mz < mz_end {
        current_mz += current_mz * (resolution_ppm / 1e6);
        index.push(current_mz);
    }

    Array1::from_vec(index)
}

pub struct XICIndex {
    pub mz_index: Array1<f32>,
    pub xic_slices: Vec<XICSlice>,
}

impl XICIndex {
    pub fn new(mz_index: Array1<f32>, xic_slices: Vec<XICSlice>) -> Self {
        Self { mz_index, xic_slices }
    }
}

pub struct XICSlice {
    pub rt: Array1<f32>,
    pub intensity: Array1<f32>,
}

impl XICSlice {
    pub fn new(rt: Array1<f32>, intensity: Array1<f32>) -> Self {
        Self { rt, intensity }
    }

    pub fn empty() -> Self {
        Self {
            rt: Array1::from_vec(Vec::new()),
            intensity: Array1::from_vec(Vec::new()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ppm_index_len() {
        let result = ppm_index(1.0, 100.0, 2000.0);
        assert_eq!(result.len(), 2995974);
    }

    #[test]
    fn test_binary_search() {
        let dia_data = XICIndex::new(
            vec![100.0, 200.0, 300.0, 400.0, 500.0].into(),
            vec![
                XICSlice::empty(),
                XICSlice::empty(),
                XICSlice::empty(),
                XICSlice::empty(),
                XICSlice::empty(),
            ],
        );
    }
} 