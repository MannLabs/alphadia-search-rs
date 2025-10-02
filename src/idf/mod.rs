use crate::mz_index::MZIndex;

#[cfg(test)]
mod tests;

pub struct InverseDocumentFrequency {
    bin_counts: Vec<u32>,
    pub total_fragments: f32,
    mz_index: &'static MZIndex,
}

impl InverseDocumentFrequency {
    pub fn new(fragment_mz: &[f32]) -> Self {
        let mz_index = MZIndex::global();
        let total_fragments = fragment_mz.len() as f32;

        // Create a vector to count fragments per bin
        let mut bin_counts = vec![0u32; mz_index.len()];

        // Count fragments in each bin (only if we have fragments)
        if !fragment_mz.is_empty() {
            for &mz in fragment_mz {
                let bin_index = mz_index.find_closest_index(mz);
                bin_counts[bin_index] += 1;
            }
        }

        Self {
            bin_counts,
            total_fragments,
            mz_index,
        }
    }

    pub fn get_idf(&self, fragment_mz: &[f32]) -> Vec<f32> {
        if self.total_fragments == 0.0 {
            return vec![1.0; fragment_mz.len()];
        }

        fragment_mz
            .iter()
            .map(|&mz| {
                let bin_index = self.mz_index.find_closest_index(mz);
                let document_frequency = self.bin_counts[bin_index].max(1) as f32;
                (self.total_fragments / document_frequency).ln()
            })
            .collect()
    }
}
