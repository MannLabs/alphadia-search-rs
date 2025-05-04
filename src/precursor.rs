
pub struct Precursor {
    pub idx: usize,
    pub mz: f32,
    pub rt: f32,
    pub fragment_mz: Vec<f32>,
    pub fragment_intensity: Vec<f32>,
}