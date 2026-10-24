pub struct Precursor {
    pub precursor_idx: usize,
    pub mz: f32,
    pub mz_library: f32,
    pub rt: f32,
    pub rt_library: f32,
    pub naa: u8,
    /// Precursor charge, 0 if the library provides no isotopes
    pub charge: u8,
    /// Library isotope abundances M0, M1, ..., empty if the library provides none
    pub isotope_intensity: Vec<f32>,
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

#[cfg(test)]
mod tests;
