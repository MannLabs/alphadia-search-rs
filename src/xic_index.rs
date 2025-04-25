use numpy::ndarray::Array1;
use rand::Rng;
use std::collections::HashSet;

const MAX_CYCLE_INDEX: usize = 1000;

pub const RESOLUTION_PPM: f32 = 1.0;
pub const MZ_START: f32 = 150.0;
pub const MZ_END: f32 = 2000.0;

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

pub struct MZIndex {
    pub mz: Array1<f32>,
}

impl MZIndex {
    pub fn new() -> Self {
        Self {
            mz: ppm_index(RESOLUTION_PPM, MZ_START, MZ_END),
        }
    }

    pub fn len(&self) -> usize {
        self.mz.len()
    }
}


#[derive(Clone)]
pub struct XICSlice {
    pub cycle_index: Vec<u16>,
    pub intensity: Vec<f32>,
}

impl XICSlice {
    pub fn new(cycle_index: Vec<u16>, intensity: Vec<f32>) -> Self {
        Self { cycle_index, intensity }
    }

    pub fn empty() -> Self {
        Self {
            cycle_index: Vec::new(),
            intensity: Vec::new(),
        }
    }

    /// Creates a new XICSlice with random data for testing purposes.
    ///
    /// This function generates a random XICSlice with:
    /// - Unique cycle indices between 0 and MAX_CYCLE_INDEX (up to 100 indices)
    /// - Random intensity values between 0.0 and 1.0 for each cycle index
    /// - Sorted cycle indices in ascending order
    ///
    /// # Returns
    /// A new XICSlice instance with random cycle indices and intensities
    pub fn random(max_elements: usize) -> Self {
        let mut unique_cycle_indices = HashSet::new();
        
        // Generate unique random cycle indices
        while unique_cycle_indices.len() < max_elements && unique_cycle_indices.len() < MAX_CYCLE_INDEX {
            unique_cycle_indices.insert(rand::thread_rng().gen_range(0..MAX_CYCLE_INDEX) as u16);
        }
        
        let mut random_cycle_index: Vec<u16> = unique_cycle_indices.into_iter().collect();
        random_cycle_index.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut random_intensity = Vec::with_capacity(random_cycle_index.len());
        for _ in 0..random_cycle_index.len() {
            random_intensity.push(rand::thread_rng().gen_range(0.0..1.0));
        }

        
        Self {
            cycle_index: random_cycle_index,
            intensity: random_intensity,
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

        let max_elements = 1000;


        let dia_data = XICIndex::new(
            vec![100.0, 200.0, 300.0, 400.0, 500.0].into(),
            vec![
                XICSlice::random(max_elements),
                XICSlice::random(max_elements),
                XICSlice::random(max_elements),
                XICSlice::random(max_elements),
                XICSlice::random(max_elements),
            ],
        );

        // Test exact match
        assert!(dia_data.closest_index(300.0).is_some());
        
        // Test closest value
        let closest = dia_data.closest_index(350.0).unwrap();
        assert_eq!(closest, 3);

        
    }
} 