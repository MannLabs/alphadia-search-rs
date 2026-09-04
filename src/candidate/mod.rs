mod entry;
mod features;

pub use entry::{Candidate, CandidateCollection};
pub use features::{CandidateFeature, CandidateFeatureCollection, NUM_FEATURES};

#[cfg(test)]
mod tests;
