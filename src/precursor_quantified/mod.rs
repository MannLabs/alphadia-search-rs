//! Module for quantified precursor data structures.
//!
//! This module contains the `PrecursorQuantified` struct which represents
//! precursor ions with their associated fragment data and quantification results.

pub struct PrecursorQuantified {
    pub precursor_idx: usize,
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
    pub fn filter_fragments_by_intensity(
        &self,
        intensity_threshold: f32,
    ) -> Option<PrecursorQuantified> {
        // Count valid fragments first to pre-allocate vectors
        let valid_count = self
            .fragment_intensity
            .iter()
            .filter(|&&intensity| intensity > intensity_threshold)
            .count();

        if valid_count == 0 {
            return None;
        }

        // Pre-allocate all vectors with exact capacity
        let mut fragment_mz_library = Vec::with_capacity(valid_count);
        let mut fragment_mz = Vec::with_capacity(valid_count);
        let mut fragment_intensity = Vec::with_capacity(valid_count);
        let mut fragment_cardinality = Vec::with_capacity(valid_count);
        let mut fragment_charge = Vec::with_capacity(valid_count);
        let mut fragment_loss_type = Vec::with_capacity(valid_count);
        let mut fragment_number = Vec::with_capacity(valid_count);
        let mut fragment_position = Vec::with_capacity(valid_count);
        let mut fragment_type = Vec::with_capacity(valid_count);
        let mut fragment_mz_observed = Vec::with_capacity(valid_count);
        let mut fragment_correlation_observed = Vec::with_capacity(valid_count);
        let mut fragment_mass_error_observed = Vec::with_capacity(valid_count);

        // Single iteration to fill all vectors
        for (i, &intensity) in self.fragment_intensity.iter().enumerate() {
            if intensity > intensity_threshold {
                fragment_mz_library.push(self.fragment_mz_library[i]);
                fragment_mz.push(self.fragment_mz[i]);
                fragment_intensity.push(intensity);
                fragment_cardinality.push(self.fragment_cardinality[i]);
                fragment_charge.push(self.fragment_charge[i]);
                fragment_loss_type.push(self.fragment_loss_type[i]);
                fragment_number.push(self.fragment_number[i]);
                fragment_position.push(self.fragment_position[i]);
                fragment_type.push(self.fragment_type[i]);
                fragment_mz_observed.push(self.fragment_mz_observed[i]);
                fragment_correlation_observed.push(self.fragment_correlation_observed[i]);
                fragment_mass_error_observed.push(self.fragment_mass_error_observed[i]);
            }
        }

        Some(PrecursorQuantified {
            precursor_idx: self.precursor_idx,
            mz_library: self.mz_library,
            mz: self.mz,
            rt_library: self.rt_library,
            rt: self.rt,
            naa: self.naa,
            rank: self.rank,
            rt_observed: self.rt_observed,
            fragment_mz_library,
            fragment_mz,
            fragment_intensity,
            fragment_cardinality,
            fragment_charge,
            fragment_loss_type,
            fragment_number,
            fragment_position,
            fragment_type,
            fragment_mz_observed,
            fragment_correlation_observed,
            fragment_mass_error_observed,
        })
    }
}

#[cfg(test)]
mod tests;
