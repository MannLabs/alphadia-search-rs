//! Candidate scoring and feature extraction for neural network classification.
//!

mod entry;
mod features;

pub use entry::{Candidate, CandidateCollection};
pub use features::{CandidateFeature, CandidateFeatureCollection};

#[cfg(test)]
mod tests;
