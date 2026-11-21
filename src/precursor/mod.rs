use crate::traits::MobilityFilter;

pub struct Precursor {
    pub precursor_idx: usize,
    pub mz: f32,
    pub mz_library: f32,
    pub rt: f32,
    pub rt_library: f32,
    pub naa: u8,
    pub fragment_mz: Vec<f32>,
    pub fragment_mz_library: Vec<f32>,
    pub fragment_intensity: Vec<f32>,
    pub fragment_cardinality: Vec<u8>,
    pub fragment_charge: Vec<u8>,
    pub fragment_loss_type: Vec<u8>,
    pub fragment_number: Vec<u8>,
    pub fragment_position: Vec<u8>,
    pub fragment_type: Vec<u8>,
}

impl Precursor {
    /// The mobility restriction for this precursor's XICs. Selection, scoring and
    /// quantification all take it from here, so they read the same peaks. `Any` until
    /// precursors carry a library mobility.
    pub fn mobility_filter(&self) -> MobilityFilter {
        MobilityFilter::Any
    }
}

#[cfg(test)]
mod tests;
