//! Data structures for SVM training
//!
//! This module defines the core data structures used by the L2-SVM algorithm,
//! closely mirroring the structures from Percolator's ssl.h

use std::fmt;

/// SVM training data structure
///
/// Corresponds to `AlgIn` class from ssl.h in the original Percolator code
#[derive(Debug, Clone)]
pub struct SvmData {
    /// Number of training examples
    pub m: usize,
    /// Number of features (including bias term)
    pub n: usize,
    /// Number of positive examples
    pub positives: usize,
    /// Number of negative examples
    pub negatives: usize,
    /// Feature matrix: vals[example][feature]
    /// Each row has n features, last element is bias term (usually 1.0)
    pub vals: Vec<Vec<f64>>,
    /// Labels: +1 for positive class, -1 for negative class
    pub labels: Vec<f64>,
}

impl SvmData {
    /// Create new SVM training data
    ///
    /// # Arguments
    /// * `features` - Feature vectors (should include bias term as last element)
    /// * `labels` - Class labels (+1.0 or -1.0)
    pub fn new(features: Vec<Vec<f64>>, labels: Vec<f64>) -> Self {
        assert_eq!(
            features.len(),
            labels.len(),
            "Features and labels must have same length"
        );
        assert!(
            !features.is_empty(),
            "Must have at least one training example"
        );

        let m = features.len();
        let n = features[0].len();

        // Verify all feature vectors have same length
        for (i, feat) in features.iter().enumerate() {
            assert_eq!(feat.len(), n, "Feature vector {i} has wrong length");
        }

        // Count positive and negative examples
        let mut positives = 0;
        let mut negatives = 0;

        for &label in &labels {
            if label > 0.0 {
                positives += 1;
            } else {
                negatives += 1;
            }
        }

        Self {
            m,
            n,
            positives,
            negatives,
            vals: features,
            labels,
        }
    }

    /// Create SVM data with automatic bias term addition
    ///
    /// This convenience method adds bias=1.0 to each feature vector
    pub fn new_with_bias(features: Vec<Vec<f64>>, labels: Vec<f64>) -> Self {
        let features_with_bias: Vec<Vec<f64>> = features
            .into_iter()
            .map(|mut feat| {
                feat.push(1.0); // Add bias term
                feat
            })
            .collect();

        Self::new(features_with_bias, labels)
    }

    /// Get feature vector for specific example
    pub fn get_features(&self, example_idx: usize) -> &[f64] {
        &self.vals[example_idx]
    }

    /// Get label for specific example
    pub fn get_label(&self, example_idx: usize) -> f64 {
        self.labels[example_idx]
    }

    /// Get target/decoy ratio for parameter initialization
    pub fn target_decoy_ratio(&self) -> f64 {
        self.positives as f64 / self.negatives as f64
    }
}

/// SVM training options
///
/// Corresponds to `options` struct from ssl.h
#[derive(Debug, Clone)]
pub struct SvmOptions {
    /// L2 regularization parameter
    pub lambda: f64,
    /// Regularization parameter for unlabeled examples (not used in basic L2-SVM)
    pub lambda_u: f64,
    /// Convergence tolerance
    pub epsilon: f64,
    /// Maximum iterations for conjugate gradient
    pub cgitermax: usize,
    /// Maximum iterations for Modified Finite Newton
    pub mfnitermax: usize,
}

impl Default for SvmOptions {
    fn default() -> Self {
        Self {
            lambda: 1.0,      // Fixed value from CrossValidation.cpp:276
            lambda_u: 0.0,    // Not used in basic L2-SVM
            epsilon: 1e-7,    // EPSILON from ssl.h:22
            cgitermax: 10000, // CGITERMAX from ssl.h:20
            mfnitermax: 50,   // MFNITERMAX from ssl.h:25
        }
    }
}

/// Vector structure for weights and outputs
///
/// Corresponds to `vector_double` from ssl.h
#[derive(Debug, Clone)]
pub struct SvmVector {
    /// Number of elements
    pub d: usize,
    /// Vector elements
    pub vec: Vec<f64>,
}

impl SvmVector {
    /// Create new vector with given size, initialized to zero
    pub fn new(size: usize) -> Self {
        Self {
            d: size,
            vec: vec![0.0; size],
        }
    }

    /// Create vector from existing data
    pub fn from_vec(data: Vec<f64>) -> Self {
        let d = data.len();
        Self { d, vec: data }
    }

    /// Get mutable reference to underlying vector
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.vec
    }

    /// Get reference to underlying vector
    pub fn as_slice(&self) -> &[f64] {
        &self.vec
    }
}

impl fmt::Display for SvmData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SvmData {{ m: {}, n: {}, positives: {}, negatives: {} }}",
            self.m, self.n, self.positives, self.negatives
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_svm_data_creation() {
        let features = vec![
            vec![1.0, 2.0, 1.0],
            vec![2.0, 3.0, 1.0],
            vec![-1.0, -2.0, 1.0],
        ];
        let labels = vec![1.0, 1.0, -1.0];

        let data = SvmData::new(features, labels);

        assert_eq!(data.m, 3);
        assert_eq!(data.n, 3);
        assert_eq!(data.positives, 2);
        assert_eq!(data.negatives, 1);

        assert_relative_eq!(data.get_label(0), 1.0);
        assert_relative_eq!(data.get_label(2), -1.0);

        assert_eq!(data.get_features(0), &[1.0, 2.0, 1.0]);
    }

    #[test]
    fn test_svm_data_with_bias() {
        let features = vec![vec![1.0, 2.0], vec![-1.0, -2.0]];
        let labels = vec![1.0, -1.0];

        let data = SvmData::new_with_bias(features, labels);

        assert_eq!(data.n, 3); // 2 features + bias
        assert_eq!(data.get_features(0), &[1.0, 2.0, 1.0]);
        assert_eq!(data.get_features(1), &[-1.0, -2.0, 1.0]);
    }

    #[test]
    fn test_target_decoy_ratio() {
        let features = vec![
            vec![1.0],
            vec![1.0],
            vec![1.0], // 3 positives
            vec![1.0],
            vec![1.0], // 2 negatives
        ];
        let labels = vec![1.0, 1.0, 1.0, -1.0, -1.0];

        let data = SvmData::new(features, labels);
        assert_relative_eq!(data.target_decoy_ratio(), 1.5);
    }

    #[test]
    fn test_svm_vector() {
        let mut vec = SvmVector::new(3);
        assert_eq!(vec.d, 3);
        assert_eq!(vec.as_slice(), &[0.0, 0.0, 0.0]);

        vec.as_mut_slice()[1] = 5.0;
        assert_relative_eq!(vec.as_slice()[1], 5.0);

        let vec2 = SvmVector::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(vec2.d, 3);
        assert_eq!(vec2.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    #[should_panic(expected = "Features and labels must have same length")]
    fn test_mismatched_features_labels() {
        let features = vec![vec![1.0], vec![2.0]];
        let labels = vec![1.0]; // Wrong length
        SvmData::new(features, labels);
    }
}
