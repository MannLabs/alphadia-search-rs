use rand::Rng;
use std::collections::HashSet;

const MAX_CYCLE_INDEX: usize = 1000;

#[derive(Debug, Clone)]
pub struct XICSlice {
    pub cycle_index: Vec<u16>,
    pub intensity: Vec<f32>,
}

impl XICSlice {
    pub fn new(cycle_index: Vec<u16>, intensity: Vec<f32>) -> Self {
        Self {
            cycle_index,
            intensity,
        }
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
        while unique_cycle_indices.len() < max_elements
            && unique_cycle_indices.len() < MAX_CYCLE_INDEX
        {
            unique_cycle_indices.insert(rand::rng().random_range(0..MAX_CYCLE_INDEX) as u16);
        }

        let mut random_cycle_index: Vec<u16> = unique_cycle_indices.into_iter().collect();
        random_cycle_index.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut random_intensity = Vec::with_capacity(random_cycle_index.len());
        for _ in 0..random_cycle_index.len() {
            random_intensity.push(rand::rng().random_range(0.0..1.0));
        }

        Self {
            cycle_index: random_cycle_index,
            intensity: random_intensity,
        }
    }
}

#[cfg(test)]
mod tests;
