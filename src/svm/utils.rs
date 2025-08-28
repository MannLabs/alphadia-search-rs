//! Utility functions for SVM implementation
//!
//! This module provides helper functions and utilities used across the SVM implementation

use crate::svm::data::SvmData;

/// Compute dot product between two vectors
pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Vectors must have same length");
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Compute L2 norm of a vector
pub fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Compute L2 norm squared of a vector
pub fn l2_norm_squared(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum()
}

/// Apply SVM prediction to feature vector
///
/// Computes w^T * x where w is weight vector and x is feature vector
pub fn predict_single(weights: &[f64], features: &[f64]) -> f64 {
    assert_eq!(
        weights.len(),
        features.len(),
        "Weight and feature dimensions must match"
    );
    dot_product(weights, features)
}

/// Apply SVM predictions to multiple feature vectors
///
/// Returns vector of predictions for each input example
pub fn predict_batch(weights: &[f64], data: &SvmData) -> Vec<f64> {
    let mut predictions = Vec::with_capacity(data.m);

    for i in 0..data.m {
        let features = data.get_features(i);
        predictions.push(predict_single(weights, features));
    }

    predictions
}

/// Convert SVM scores to probabilities using sigmoid function
///
/// Applies sigmoid transformation: p = 1 / (1 + exp(-score))
pub fn scores_to_probabilities(scores: &[f64]) -> Vec<f64> {
    scores
        .iter()
        .map(|&score| 1.0 / (1.0 + (-score).exp()))
        .collect()
}

/// Compute classification accuracy
///
/// Returns fraction of examples where sign(prediction) == sign(label)
pub fn classification_accuracy(predictions: &[f64], labels: &[f64]) -> f64 {
    assert_eq!(predictions.len(), labels.len());

    let correct = predictions
        .iter()
        .zip(labels.iter())
        .filter(|(&pred, &label)| pred.signum() == label.signum())
        .count();

    correct as f64 / predictions.len() as f64
}

/// FDR/Q-value calculation using target-decoy competition
/// Based on Percolator's approach for proteomics FDR estimation
pub fn compute_q_values_target_decoy(scores: &[f64], labels: &[f64]) -> Vec<f64> {
    // Create score-label pairs and sort by score descending
    let mut score_label_pairs: Vec<(f64, bool)> = scores
        .iter()
        .zip(labels.iter())
        .map(|(&score, &label)| (score, label > 0.0))
        .collect();

    // Sort by score descending (highest scores first)
    score_label_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let mut q_values = vec![1.0; scores.len()];
    let mut targets = 0;
    let mut decoys = 0;

    // Target-decoy FDR estimation: q = decoys / max(targets, 1)
    // Pi0 = 1.0 (conservative assumption - no null hypothesis correction)
    for (i, (_score, is_target)) in score_label_pairs.iter().enumerate() {
        if *is_target {
            targets += 1;
        } else {
            decoys += 1;
        }

        // FDR = decoys / targets (with +1 correction for small datasets)
        let fdr = (decoys + 1) as f64 / (targets + 1).max(1) as f64;
        q_values[i] = fdr.min(1.0);
    }

    // Monotonicity correction - ensure q-values are non-decreasing
    for i in (0..q_values.len() - 1).rev() {
        if q_values[i] > q_values[i + 1] {
            q_values[i] = q_values[i + 1];
        }
    }

    // Reorder q-values to match original score order
    let mut original_order_qvals = vec![0.0; scores.len()];
    let original_indices: Vec<usize> = (0..scores.len()).collect();

    // Create mapping from sorted position back to original position
    let mut sorted_indices = original_indices.clone();
    sorted_indices.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());

    for (sorted_pos, &original_pos) in sorted_indices.iter().enumerate() {
        original_order_qvals[original_pos] = q_values[sorted_pos];
    }

    original_order_qvals
}

/// Count positive predictions using FDR threshold
/// Returns number of targets with q-value < fdr_threshold
pub fn count_positives_at_fdr(scores: &[f64], labels: &[f64], fdr_threshold: f64) -> usize {
    let q_values = compute_q_values_target_decoy(scores, labels);

    scores
        .iter()
        .zip(labels.iter())
        .zip(q_values.iter())
        .filter(|((&_score, &label), &q_val)| label > 0.0 && q_val < fdr_threshold)
        .count()
}

/// Compute Area Under ROC Curve (AUC)
///
/// Simple implementation for binary classification
pub fn compute_auc(scores: &[f64], labels: &[f64]) -> f64 {
    assert_eq!(scores.len(), labels.len());

    // Create (score, label) pairs and sort by score descending
    let mut pairs: Vec<(f64, f64)> = scores
        .iter()
        .zip(labels.iter())
        .map(|(&s, &l)| (s, l))
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let mut tp = 0.0; // true positives
    let mut _fp = 0.0; // false positives
    let mut auc = 0.0;

    let total_pos = labels.iter().filter(|&&l| l > 0.0).count() as f64;
    let total_neg = labels.iter().filter(|&&l| l <= 0.0).count() as f64;

    if total_pos == 0.0 || total_neg == 0.0 {
        return 0.5; // Undefined case, return neutral
    }

    for (_, label) in pairs {
        if label > 0.0 {
            tp += 1.0;
        } else {
            _fp += 1.0;
            auc += tp / total_pos; // Add TPR for each FP increment
        }
    }

    auc / total_neg
}

/// Create stratified train/test split
///
/// Maintains class balance in both training and test sets
pub fn stratified_split(data: &SvmData, test_fraction: f64, seed: u64) -> (Vec<usize>, Vec<usize>) {
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    let mut rng = StdRng::seed_from_u64(seed);

    // Separate positive and negative indices
    let mut pos_indices = Vec::new();
    let mut neg_indices = Vec::new();

    for i in 0..data.m {
        if data.get_label(i) > 0.0 {
            pos_indices.push(i);
        } else {
            neg_indices.push(i);
        }
    }

    // Shuffle indices
    use rand::seq::SliceRandom;
    pos_indices.shuffle(&mut rng);
    neg_indices.shuffle(&mut rng);

    // Split each class
    let pos_test_size = (pos_indices.len() as f64 * test_fraction).round() as usize;
    let neg_test_size = (neg_indices.len() as f64 * test_fraction).round() as usize;

    let mut train_indices = Vec::new();
    let mut test_indices = Vec::new();

    // Add positive examples
    train_indices.extend(&pos_indices[pos_test_size..]);
    test_indices.extend(&pos_indices[..pos_test_size]);

    // Add negative examples
    train_indices.extend(&neg_indices[neg_test_size..]);
    test_indices.extend(&neg_indices[..neg_test_size]);

    (train_indices, test_indices)
}

/// Extract subset of SVM data based on indices
pub fn extract_subset(data: &SvmData, indices: &[usize]) -> SvmData {
    let mut features = Vec::new();
    let mut labels = Vec::new();

    for &i in indices {
        features.push(data.get_features(i).to_vec());
        labels.push(data.get_label(i));
    }

    SvmData::new(features, labels)
}

/// Normalize feature vectors using z-score normalization
///
/// Returns (normalized_data, means, stdevs) for later denormalization
pub fn normalize_features(data: &SvmData) -> (SvmData, Vec<f64>, Vec<f64>) {
    let n_features = data.n - 1; // Exclude bias term
    let mut means = vec![0.0; n_features];
    let mut stdevs = vec![1.0; n_features];

    // Compute means (excluding bias term)
    for i in 0..data.m {
        let features = data.get_features(i);
        for j in 0..n_features {
            means[j] += features[j];
        }
    }
    for mean in &mut means {
        *mean /= data.m as f64;
    }

    // Compute standard deviations
    for i in 0..data.m {
        let features = data.get_features(i);
        for j in 0..n_features {
            let diff = features[j] - means[j];
            stdevs[j] += diff * diff;
        }
    }
    for stdev in stdevs.iter_mut() {
        *stdev = (*stdev / data.m as f64).sqrt();
        if *stdev < 1e-8 {
            *stdev = 1.0; // Avoid division by zero
        }
    }

    // Normalize features
    let mut normalized_features = Vec::new();
    for i in 0..data.m {
        let features = data.get_features(i);
        let mut norm_features = vec![0.0; data.n];

        // Normalize features (not bias)
        for j in 0..n_features {
            norm_features[j] = (features[j] - means[j]) / stdevs[j];
        }
        // Keep bias term unchanged
        norm_features[n_features] = features[n_features];

        normalized_features.push(norm_features);
    }

    let normalized_data = SvmData::new(normalized_features, data.labels.clone());
    (normalized_data, means, stdevs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::svm::data::SvmData;
    use approx::assert_relative_eq;

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let result = dot_product(&a, &b);
        assert_relative_eq!(result, 32.0); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_l2_norm() {
        let v = vec![3.0, 4.0];
        assert_relative_eq!(l2_norm(&v), 5.0); // sqrt(3^2 + 4^2) = 5
        assert_relative_eq!(l2_norm_squared(&v), 25.0); // 3^2 + 4^2 = 25
    }

    #[test]
    fn test_predict_batch() {
        let features = vec![vec![1.0, 2.0, 1.0], vec![2.0, 1.0, 1.0]];
        let labels = vec![1.0, -1.0];
        let data = SvmData::new(features, labels);

        let weights = vec![0.5, -0.5, 0.1];
        let predictions = predict_batch(&weights, &data);

        assert_eq!(predictions.len(), 2);
        assert_relative_eq!(predictions[0], 1.0 * 0.5 + 2.0 * (-0.5) + 1.0 * 0.1); // -0.4
        assert_relative_eq!(predictions[1], 2.0 * 0.5 + 1.0 * (-0.5) + 1.0 * 0.1);
        // 0.6
    }

    #[test]
    fn test_scores_to_probabilities() {
        let scores = vec![0.0, 2.0, -2.0];
        let probs = scores_to_probabilities(&scores);

        assert_relative_eq!(probs[0], 0.5, epsilon = 1e-6); // sigmoid(0) = 0.5
        assert!(probs[1] > 0.8); // sigmoid(2) should be high
        assert!(probs[2] < 0.2); // sigmoid(-2) should be low
    }

    #[test]
    fn test_classification_accuracy() {
        let predictions = vec![1.5, -0.5, 2.0, -1.0];
        let labels = vec![1.0, -1.0, 1.0, -1.0];

        let accuracy = classification_accuracy(&predictions, &labels);
        assert_relative_eq!(accuracy, 1.0); // All correct

        let predictions_mixed = vec![1.0, 1.0, -1.0, -1.0];
        let accuracy_mixed = classification_accuracy(&predictions_mixed, &labels);
        assert_relative_eq!(accuracy_mixed, 0.5); // 2 out of 4 correct
    }

    #[test]
    fn test_stratified_split() {
        let features = vec![
            vec![1.0, 1.0],
            vec![1.1, 1.1],
            vec![1.2, 1.2], // 3 positive
            vec![-1.0, 1.0],
            vec![-1.1, 1.0], // 2 negative
        ];
        let labels = vec![1.0, 1.0, 1.0, -1.0, -1.0];
        let data = SvmData::new(features, labels);

        let (train_indices, test_indices) = stratified_split(&data, 0.4, 42);

        // Should maintain approximate class balance
        assert!(train_indices.len() + test_indices.len() == 5);
        assert!(test_indices.len() >= 1); // Should have some test examples
        assert!(train_indices.len() >= 1); // Should have some train examples
    }

    #[test]
    fn test_normalize_features() {
        let features = vec![
            vec![1.0, 10.0, 1.0], // bias term should be preserved
            vec![3.0, 20.0, 1.0],
            vec![5.0, 30.0, 1.0],
        ];
        let labels = vec![1.0, 1.0, 1.0];
        let data = SvmData::new(features, labels);

        let (normalized_data, means, stdevs) = normalize_features(&data);

        // Check that means are computed correctly
        assert_relative_eq!(means[0], 3.0); // (1+3+5)/3
        assert_relative_eq!(means[1], 20.0); // (10+20+30)/3

        // Check that bias terms are preserved
        for i in 0..normalized_data.m {
            let norm_features = normalized_data.get_features(i);
            assert_relative_eq!(norm_features[2], 1.0); // Bias term unchanged
        }

        // Check that normalized features have approximately zero mean
        let mut sum0 = 0.0;
        let mut sum1 = 0.0;
        for i in 0..normalized_data.m {
            let features = normalized_data.get_features(i);
            sum0 += features[0];
            sum1 += features[1];
        }
        assert_relative_eq!(sum0 / 3.0, 0.0, epsilon = 1e-10);
        assert_relative_eq!(sum1 / 3.0, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_compute_auc_perfect_classifier() {
        let scores = vec![0.9, 0.8, 0.1, 0.2];
        let labels = vec![1.0, 1.0, -1.0, -1.0];

        let auc = compute_auc(&scores, &labels);
        assert_relative_eq!(auc, 1.0, epsilon = 1e-6); // Perfect separation
    }

    #[test]
    fn test_compute_auc_random_classifier() {
        let scores = vec![0.5, 0.5, 0.5, 0.5];
        let labels = vec![1.0, 1.0, -1.0, -1.0];

        let auc = compute_auc(&scores, &labels);
        assert_relative_eq!(auc, 0.5, epsilon = 0.1); // Random performance
    }

    #[test]
    fn test_fdr_q_values_simple() {
        // Simple case: 2 targets, 1 decoy, sorted by score
        let scores = vec![0.9, 0.8, 0.1]; // High to low scores
        let labels = vec![1.0, 1.0, -1.0]; // Target, Target, Decoy

        let q_values = compute_q_values_target_decoy(&scores, &labels);

        // First target: 0 decoys, 1 target -> q = (0+1)/(1+1) = 0.5
        // Second target: 0 decoys, 2 targets -> q = (0+1)/(2+1) = 0.33
        // Decoy: 1 decoy, 2 targets -> q = (1+1)/(2+1) = 0.67

        assert!(q_values[0] > 0.3 && q_values[0] < 0.6); // First target
        assert!(q_values[1] > 0.2 && q_values[1] < 0.5); // Second target
        assert!(q_values[2] > 0.5 && q_values[2] < 0.8); // Decoy
    }

    #[test]
    fn test_count_positives_at_fdr() {
        let scores = vec![0.9, 0.8, 0.7, 0.1, 0.05];
        let labels = vec![1.0, 1.0, 1.0, -1.0, -1.0]; // 3 targets, 2 decoys

        // At very strict FDR (0.01), should find very few positives
        let strict_count = count_positives_at_fdr(&scores, &labels, 0.01);

        // At lenient FDR (0.5), should find more positives
        let lenient_count = count_positives_at_fdr(&scores, &labels, 0.5);

        assert!(strict_count <= lenient_count);
        assert!(lenient_count <= 3); // Can't exceed number of targets
    }

    #[test]
    fn test_fdr_crossvalidation_scenario() {
        // Replicate the CrossValidation test scenario
        const N: usize = 10; // Smaller for unit test

        let mut scores = Vec::new();
        let mut labels = Vec::new();

        // Create targets with slightly increasing scores
        for i in 0..(2 * N) {
            scores.push((i as f64) / (N as f64) - 1e-6);
            labels.push(1.0); // TARGET
        }

        // Create decoys with slightly higher scores
        for i in 0..N {
            scores.push((i as f64) / (N as f64));
            labels.push(-1.0); // DECOY
        }

        // Count positives at FDR=0.02 (matching C++ test)
        let fdr_positives = count_positives_at_fdr(&scores, &labels, 0.02);

        // Should find fewer positives than naive score > 0 counting
        let naive_positives = scores
            .iter()
            .zip(labels.iter())
            .filter(|(&score, &label)| label > 0.0 && score > 0.0)
            .count();

        assert!(fdr_positives <= naive_positives);
        println!(
            "FDR positives: {}, Naive positives: {}",
            fdr_positives, naive_positives
        );
    }
}
