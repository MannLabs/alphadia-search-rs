pub struct Precursor {
    pub idx: usize,
    pub mz: f32,
    pub rt: f32,
    pub naa: u8,
    pub fragment_mz: Vec<f32>,
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
