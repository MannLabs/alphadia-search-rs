use crate::traits::MobilityFilter;

pub struct Precursor {
    pub precursor_idx: usize,
    pub mz: f32,
    pub mz_library: f32,
    pub rt: f32,
    pub rt_library: f32,
    /// Ion mobility (1/K0), NaN when the library has none.
    pub mobility: f32,
    /// Factor on the mobility tolerance for this precursor (1 for most): charge states whose mobility the library
    /// predicts less well can search a wider window.
    pub mobility_tolerance_scale: f32,
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
    /// The mobility restriction for this precursor's XICs: a window of `tolerance` around its
    /// library mobility, or `Any` when it has none. Selection, scoring and quantification all
    /// take it from here, so they read the same peaks.
    pub fn mobility_filter(&self, tolerance: f32) -> MobilityFilter {
        MobilityFilter::from_library(self.mobility, self.mobility_tolerance(tolerance))
            .expect("set_precursor_mobility and the mobility_tolerance setters validate both")
    }

    /// The mobility tolerance of this precursor for a run-wide `tolerance`.
    pub fn mobility_tolerance(&self, tolerance: f32) -> f32 {
        tolerance * self.mobility_tolerance_scale
    }
}

#[cfg(test)]
mod tests;
