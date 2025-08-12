//! Module for quantified precursor data structures.
//!
//! This module contains the `PrecursorQuantified` struct which represents
//! precursor ions with their associated fragment data and quantification results.

pub struct PrecursorQuantified {
    pub idx: usize,
    pub mz: f32,
    pub rt: f32,
    pub naa: u8,
    pub rank: usize,
    pub rt_observed: f32,
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

#[cfg(test)]
mod tests;
