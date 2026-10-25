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

/// Library m/z values closer than this (relative) are the same fragment: the excluded values are copies of this
/// precursor's own library m/z, so the margin only absorbs float32 round-trips.
const SAME_FRAGMENT_RELATIVE_TOLERANCE: f32 = 1e-6;

impl Precursor {
    /// This precursor without the fragments whose library m/z is in `excluded_mz_library`.
    pub fn without_fragments(&self, excluded_mz_library: &[f32]) -> Precursor {
        let keep: Vec<bool> = self
            .fragment_mz_library
            .iter()
            .map(|&mz| {
                !excluded_mz_library
                    .iter()
                    .any(|&ex| (ex - mz).abs() <= SAME_FRAGMENT_RELATIVE_TOLERANCE * mz)
            })
            .collect();
        fn select<T: Copy>(values: &[T], keep: &[bool]) -> Vec<T> {
            values
                .iter()
                .zip(keep)
                .filter(|(_, &k)| k)
                .map(|(&v, _)| v)
                .collect()
        }
        Precursor {
            precursor_idx: self.precursor_idx,
            mz: self.mz,
            mz_library: self.mz_library,
            rt: self.rt,
            rt_library: self.rt_library,
            naa: self.naa,
            fragment_mz: select(&self.fragment_mz, &keep),
            fragment_mz_library: select(&self.fragment_mz_library, &keep),
            fragment_intensity: select(&self.fragment_intensity, &keep),
            fragment_cardinality: select(&self.fragment_cardinality, &keep),
            fragment_charge: select(&self.fragment_charge, &keep),
            fragment_loss_type: select(&self.fragment_loss_type, &keep),
            fragment_number: select(&self.fragment_number, &keep),
            fragment_position: select(&self.fragment_position, &keep),
            fragment_type: select(&self.fragment_type, &keep),
        }
    }
}

#[cfg(test)]
mod tests;
