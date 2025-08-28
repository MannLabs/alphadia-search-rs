//! Semi-supervised SVM implementation for Percolator-style algorithms
//!
//! This module provides a direct port of Percolator's L2-SVM with Modified Finite Newton
//! optimization, specifically designed for target-decoy competition in proteomics.
//!
//! # Key Components
//!
//! - `L2SvmMfn`: Main L2-SVM trainer with Modified Finite Newton method
//! - `CglsSolver`: Conjugate Gradient Least Squares solver
//! - `LineSearch`: Line search optimization for step size
//! - `SvmData`: Input data structure for training
//!
//! # Algorithm Overview
//!
//! The algorithm implements semi-supervised learning for target-decoy competition:
//! 1. Start with decoys as known negatives, targets as unlabeled mixture
//! 2. Use FDR to select initial positive set from targets
//! 3. Train L2-SVM on {selected targets=positive, all decoys=negative}
//! 4. Re-score all targets, update positive set via FDR
//! 5. Iterate until convergence
//!
//! This is a faithful port of the algorithms from Percolator's ssl.cpp.

pub mod cgls;
pub mod data;
pub mod l2_svm_mfn;
pub mod line_search;
pub mod tests;
pub mod utils;

// Re-export main types for convenience
pub use cgls::CglsSolver;
pub use data::{SvmData, SvmOptions};
pub use l2_svm_mfn::L2SvmMfn;
pub use line_search::LineSearch;

/// Main entry point for L2-SVM training
pub fn train_l2_svm(
    data: &SvmData,
    options: &SvmOptions,
    cpos: f64,
    cneg: f64,
) -> Result<(Vec<f64>, Vec<f64>), Box<dyn std::error::Error>> {
    let mut trainer = L2SvmMfn::new(data, options);
    trainer.train(cpos, cneg)
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_l2_svm_basic_training() {
        // Simple 2D separable dataset
        let features = vec![
            vec![1.0, 2.0, 1.0],   // positive example (with bias term)
            vec![2.0, 3.0, 1.0],   // positive example
            vec![-1.0, -2.0, 1.0], // negative example
            vec![-2.0, -3.0, 1.0], // negative example
        ];

        let labels = vec![1.0, 1.0, -1.0, -1.0]; // +1 for positive, -1 for negative

        let data = SvmData::new(features, labels);
        let options = SvmOptions::default();

        let (weights, outputs) =
            train_l2_svm(&data, &options, 1.0, 1.0).expect("Training should succeed");

        // Check that we learned reasonable weights
        assert_eq!(weights.len(), 3); // 2 features + bias
        assert_eq!(outputs.len(), 4); // 4 training examples

        // Outputs should have correct signs (positive for positive examples, etc.)
        assert!(
            outputs[0] > 0.0,
            "Positive example should have positive output"
        );
        assert!(
            outputs[1] > 0.0,
            "Positive example should have positive output"
        );
        assert!(
            outputs[2] < 0.0,
            "Negative example should have negative output"
        );
        assert!(
            outputs[3] < 0.0,
            "Negative example should have negative output"
        );
    }
}
