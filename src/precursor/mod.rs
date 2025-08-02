pub struct Precursor {
    pub idx: usize,
    pub mz: f32,
    pub rt: f32,
    pub fragment_mz: Vec<f32>,
    pub fragment_intensity: Vec<f32>,
}

impl Precursor {
    pub fn get_fragments_filtered(
        &self,
        non_zero: bool,
        top_k_fragments: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut fragment_triplets: Vec<(f32, f32, usize)> = self
            .fragment_mz
            .iter()
            .zip(self.fragment_intensity.iter())
            .enumerate()
            .map(|(idx, (&mz, &intensity))| (mz, intensity, idx))
            .collect();

        // Filter non-zero intensities if requested
        if non_zero {
            fragment_triplets.retain(|(_, intensity, _)| *intensity > 0.0);
        }

        // Use partial sorting for top-k selection - much faster than full sort
        let k = top_k_fragments.min(fragment_triplets.len());
        if k < fragment_triplets.len() {
            // Partial sort: only sort the k-th element and everything before it
            fragment_triplets.select_nth_unstable_by(k, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            fragment_triplets.truncate(k);
        }
        // Sort by original index to maintain original m/z ordering
        fragment_triplets.sort_by_key(|(_, _, idx)| *idx);

        let (fragment_mz, fragment_intensity): (Vec<f32>, Vec<f32>) = fragment_triplets
            .into_iter()
            .map(|(mz, intensity, _)| (mz, intensity))
            .unzip();

        (fragment_mz, fragment_intensity)
    }
}

#[cfg(test)]
mod tests;
