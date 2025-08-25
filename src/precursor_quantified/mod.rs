//! Module for quantified precursor data structures.
//!
//! This module contains the `PrecursorQuantified` struct which represents
//! precursor ions with their associated fragment data and quantification results.

pub struct PrecursorQuantified {
    pub idx: usize,
    pub mz_library: f32,
    pub mz: f32,
    pub rt_library: f32,
    pub rt: f32,
    pub naa: u8,
    pub rank: usize,
    pub rt_observed: f32,
    pub fragment_mz_library: Vec<f32>,
    pub fragment_mz: Vec<f32>,
    pub fragment_intensity: Vec<f32>,
    pub fragment_cardinality: Vec<u8>,
    pub fragment_charge: Vec<u8>,
    pub fragment_loss_type: Vec<u8>,
    pub fragment_number: Vec<u8>,
    pub fragment_position: Vec<u8>,
    pub fragment_type: Vec<u8>,
    pub fragment_mz_observed: Vec<f32>,
    pub fragment_correlation_observed: Vec<f32>,
    pub fragment_mass_error_observed: Vec<f32>,
}

impl PrecursorQuantified {
    pub fn filter_fragments_by_intensity(&self, min_intensity: f32) -> Option<PrecursorQuantified> {
        let valid_indices: Vec<usize> = self
            .fragment_intensity
            .iter()
            .enumerate()
            .filter_map(|(idx, &intensity)| {
                if intensity > min_intensity {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect();

        if valid_indices.is_empty() {
            return None;
        }

        Some(PrecursorQuantified {
            idx: self.idx,
            mz_library: self.mz_library,
            mz: self.mz,
            rt_library: self.rt_library,
            rt: self.rt,
            naa: self.naa,
            rank: self.rank,
            rt_observed: self.rt_observed,
            fragment_mz_library: valid_indices
                .iter()
                .map(|&i| self.fragment_mz_library[i])
                .collect(),
            fragment_mz: valid_indices.iter().map(|&i| self.fragment_mz[i]).collect(),
            fragment_intensity: valid_indices
                .iter()
                .map(|&i| self.fragment_intensity[i])
                .collect(),
            fragment_cardinality: valid_indices
                .iter()
                .map(|&i| self.fragment_cardinality[i])
                .collect(),
            fragment_charge: valid_indices
                .iter()
                .map(|&i| self.fragment_charge[i])
                .collect(),
            fragment_loss_type: valid_indices
                .iter()
                .map(|&i| self.fragment_loss_type[i])
                .collect(),
            fragment_number: valid_indices
                .iter()
                .map(|&i| self.fragment_number[i])
                .collect(),
            fragment_position: valid_indices
                .iter()
                .map(|&i| self.fragment_position[i])
                .collect(),
            fragment_type: valid_indices
                .iter()
                .map(|&i| self.fragment_type[i])
                .collect(),
            fragment_mz_observed: valid_indices
                .iter()
                .map(|&i| self.fragment_mz_observed[i])
                .collect(),
            fragment_correlation_observed: valid_indices
                .iter()
                .map(|&i| self.fragment_correlation_observed[i])
                .collect(),
            fragment_mass_error_observed: valid_indices
                .iter()
                .map(|&i| self.fragment_mass_error_observed[i])
                .collect(),
        })
    }
}

#[cfg(test)]
mod tests;
