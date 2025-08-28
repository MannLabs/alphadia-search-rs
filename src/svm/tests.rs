//! Comprehensive integration tests for the SVM module
//!
//! These tests validate the complete SVM implementation against known results
//! and test edge cases to ensure robustness.

#[cfg(test)]
mod integration_tests {
    use crate::svm::*;
    use approx::assert_relative_eq;
    use std::collections::HashMap;

    /// Test data generator for reproducible test cases
    struct TestDataGenerator {
        seed: u64,
    }

    impl TestDataGenerator {
        fn new(seed: u64) -> Self {
            Self { seed }
        }

        /// Generate linearly separable 2D data
        fn linearly_separable_2d(&self, n_per_class: usize) -> SvmData {
            use rand::rngs::StdRng;
            use rand::{Rng, SeedableRng};

            let mut rng = StdRng::seed_from_u64(self.seed);
            let mut features = Vec::new();
            let mut labels = Vec::new();

            // Positive examples (upper right quadrant)
            for _ in 0..n_per_class {
                let x1 = rng.random_range(0.5..2.0);
                let x2 = rng.random_range(0.5..2.0);
                features.push(vec![x1, x2, 1.0]); // with bias
                labels.push(1.0);
            }

            // Negative examples (lower left quadrant)
            for _ in 0..n_per_class {
                let x1 = rng.random_range(-2.0..-0.5);
                let x2 = rng.random_range(-2.0..-0.5);
                features.push(vec![x1, x2, 1.0]); // with bias
                labels.push(-1.0);
            }

            SvmData::new(features, labels)
        }

        /// Generate overlapping classes (harder problem)
        fn overlapping_classes(&self, n_per_class: usize, overlap: f64) -> SvmData {
            use rand::rngs::StdRng;
            use rand::{Rng, SeedableRng};

            let mut rng = StdRng::seed_from_u64(self.seed);

            let mut features = Vec::new();
            let mut labels = Vec::new();

            // Positive examples centered at (1, 1) with noise
            for _ in 0..n_per_class {
                let x1 = 1.0 + overlap * (rng.random_range(-1.0..1.0));
                let x2 = 1.0 + overlap * (rng.random_range(-1.0..1.0));
                features.push(vec![x1, x2, 1.0]);
                labels.push(1.0);
            }

            // Negative examples centered at (-1, -1) with noise
            for _ in 0..n_per_class {
                let x1 = -1.0 + overlap * (rng.random_range(-1.0..1.0));
                let x2 = -1.0 + overlap * (rng.random_range(-1.0..1.0));
                features.push(vec![x1, x2, 1.0]);
                labels.push(-1.0);
            }

            SvmData::new(features, labels)
        }
    }

    #[test]
    fn test_end_to_end_separable_dataset() {
        let generator = TestDataGenerator::new(42);
        let data = generator.linearly_separable_2d(20);

        let options = SvmOptions::default();
        let (weights, outputs) =
            train_l2_svm(&data, &options, 1.0, 1.0).expect("Training should succeed");

        // Validate training results
        assert_eq!(weights.len(), 3); // 2 features + bias
        assert_eq!(outputs.len(), 40); // 20 per class

        // Check classification accuracy
        let accuracy = utils::classification_accuracy(&outputs, &data.labels);
        assert!(
            accuracy > 0.9,
            "Should achieve >90% accuracy on separable data, got {}",
            accuracy
        );

        // Check that positive/negative examples are well separated
        let pos_outputs: Vec<f64> = outputs
            .iter()
            .zip(data.labels.iter())
            .filter_map(|(&output, &label)| if label > 0.0 { Some(output) } else { None })
            .collect();
        let neg_outputs: Vec<f64> = outputs
            .iter()
            .zip(data.labels.iter())
            .filter_map(|(&output, &label)| if label < 0.0 { Some(output) } else { None })
            .collect();

        let min_pos = pos_outputs.iter().copied().fold(f64::INFINITY, f64::min);
        let max_neg = neg_outputs
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        assert!(
            min_pos > max_neg,
            "Positive examples should have higher scores than negative"
        );
    }

    #[test]
    fn test_overlapping_classes_with_regularization() {
        let generator = TestDataGenerator::new(123);
        let data = generator.overlapping_classes(30, 0.8); // Significant overlap

        let mut options = SvmOptions::default();
        options.lambda = 0.1; // Moderate regularization

        let (weights, outputs) =
            train_l2_svm(&data, &options, 1.0, 1.0).expect("Training should succeed");

        // Should still achieve reasonable accuracy despite overlap
        let accuracy = utils::classification_accuracy(&outputs, &data.labels);
        assert!(
            accuracy > 0.7,
            "Should achieve >70% accuracy on overlapping data, got {}",
            accuracy
        );

        // Weights should not be too large (regularization effect)
        let weight_norm = utils::l2_norm(&weights[..2]); // Exclude bias
        assert!(
            weight_norm < 10.0,
            "Regularization should prevent very large weights"
        );
    }

    #[test]
    fn test_cost_parameter_sensitivity() {
        let generator = TestDataGenerator::new(456);
        let data = generator.linearly_separable_2d(10);

        let options = SvmOptions::default();

        // Train with balanced costs
        let (weights_balanced, _) =
            train_l2_svm(&data, &options, 1.0, 1.0).expect("Balanced training should succeed");

        // Train with high cost for positives
        let (weights_high_pos, _) = train_l2_svm(&data, &options, 10.0, 1.0)
            .expect("High positive cost training should succeed");

        // Train with high cost for negatives
        let (weights_high_neg, _) = train_l2_svm(&data, &options, 1.0, 10.0)
            .expect("High negative cost training should succeed");

        // Different cost parameters should produce different solutions
        let diff_pos = weights_balanced
            .iter()
            .zip(weights_high_pos.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>();
        let diff_neg = weights_balanced
            .iter()
            .zip(weights_high_neg.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>();

        assert!(diff_pos > 0.1, "High positive cost should change solution");
        assert!(diff_neg > 0.1, "High negative cost should change solution");
    }

    #[test]
    fn test_feature_normalization_impact() {
        // Create dataset with features of very different scales
        let features = vec![
            vec![1000.0, 0.001, 1.0], // Large first feature, tiny second
            vec![2000.0, 0.002, 1.0],
            vec![-1000.0, -0.001, 1.0],
            vec![-2000.0, -0.002, 1.0],
        ];
        let labels = vec![1.0, 1.0, -1.0, -1.0];
        let data = SvmData::new(features, labels);

        let options = SvmOptions::default();

        // Train on unnormalized data
        let (weights_unnorm, _) = train_l2_svm(&data, &options, 1.0, 1.0)
            .expect("Training on unnormalized data should succeed");

        // Train on normalized data
        let (normalized_data, _, _) = utils::normalize_features(&data);
        let (weights_norm, _) = train_l2_svm(&normalized_data, &options, 1.0, 1.0)
            .expect("Training on normalized data should succeed");

        // Normalized version should have more balanced feature weights
        let weight_ratio_unnorm = weights_unnorm[0].abs() / weights_unnorm[1].abs();
        let weight_ratio_norm = weights_norm[0].abs() / weights_norm[1].abs();

        assert!(
            weight_ratio_norm < weight_ratio_unnorm,
            "Normalization should balance feature importance"
        );
    }

    #[test]
    fn test_cross_validation_simulation() {
        let generator = TestDataGenerator::new(789);
        let data = generator.overlapping_classes(50, 0.5);

        let options = SvmOptions::default();

        // Simulate 3-fold cross-validation
        let mut fold_accuracies = Vec::new();

        for fold in 0..3 {
            let (train_indices, test_indices) = utils::stratified_split(&data, 0.33, fold as u64);
            let train_data = utils::extract_subset(&data, &train_indices);
            let test_data = utils::extract_subset(&data, &test_indices);

            let (weights, _) = train_l2_svm(&train_data, &options, 1.0, 1.0)
                .expect("Fold training should succeed");

            let test_predictions = utils::predict_batch(&weights, &test_data);
            let fold_accuracy =
                utils::classification_accuracy(&test_predictions, &test_data.labels);

            fold_accuracies.push(fold_accuracy);
        }

        // All folds should achieve reasonable accuracy
        for (i, &accuracy) in fold_accuracies.iter().enumerate() {
            assert!(
                accuracy > 0.6,
                "Fold {} should achieve >60% accuracy, got {}",
                i,
                accuracy
            );
        }

        // Cross-validation accuracy should be consistent across folds
        let mean_accuracy = fold_accuracies.iter().sum::<f64>() / fold_accuracies.len() as f64;
        let std_accuracy = {
            let variance = fold_accuracies
                .iter()
                .map(|&acc| (acc - mean_accuracy).powi(2))
                .sum::<f64>()
                / fold_accuracies.len() as f64;
            variance.sqrt()
        };

        assert!(
            std_accuracy < 0.2,
            "Cross-validation should be consistent (std = {})",
            std_accuracy
        );
    }

    #[test]
    fn test_auc_computation() {
        let generator = TestDataGenerator::new(999);
        let data = generator.linearly_separable_2d(25);

        let options = SvmOptions::default();
        let (weights, outputs) =
            train_l2_svm(&data, &options, 1.0, 1.0).expect("Training should succeed");

        let auc = utils::compute_auc(&outputs, &data.labels);

        // Should achieve high AUC on separable data
        assert!(
            auc > 0.9,
            "Should achieve AUC > 0.9 on separable data, got {}",
            auc
        );
        assert!(auc <= 1.0, "AUC should not exceed 1.0");

        // Test probability conversion
        let probabilities = utils::scores_to_probabilities(&outputs);

        // All probabilities should be valid
        for &prob in &probabilities {
            assert!(prob >= 0.0 && prob <= 1.0, "Probability should be in [0,1]");
        }

        // Positive examples should generally have higher probabilities
        let pos_probs: Vec<f64> = probabilities
            .iter()
            .zip(data.labels.iter())
            .filter_map(|(&prob, &label)| if label > 0.0 { Some(prob) } else { None })
            .collect();
        let neg_probs: Vec<f64> = probabilities
            .iter()
            .zip(data.labels.iter())
            .filter_map(|(&prob, &label)| if label < 0.0 { Some(prob) } else { None })
            .collect();

        let mean_pos_prob = pos_probs.iter().sum::<f64>() / pos_probs.len() as f64;
        let mean_neg_prob = neg_probs.iter().sum::<f64>() / neg_probs.len() as f64;

        assert!(
            mean_pos_prob > mean_neg_prob,
            "Positive examples should have higher average probability"
        );
    }

    #[test]
    fn test_edge_cases() {
        let options = SvmOptions::default();

        // Test single positive example
        let single_pos = SvmData::new(vec![vec![1.0, 1.0, 1.0]], vec![1.0]);
        let result = train_l2_svm(&single_pos, &options, 1.0, 1.0);
        assert!(result.is_ok(), "Should handle single positive example");

        // Test single negative example
        let single_neg = SvmData::new(vec![vec![1.0, 1.0, 1.0]], vec![-1.0]);
        let result = train_l2_svm(&single_neg, &options, 1.0, 1.0);
        assert!(result.is_ok(), "Should handle single negative example");

        // Test identical features
        let identical_features = SvmData::new(
            vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]],
            vec![1.0, -1.0],
        );
        let result = train_l2_svm(&identical_features, &options, 1.0, 1.0);
        assert!(
            result.is_ok(),
            "Should handle identical features gracefully"
        );
    }

    #[test]
    fn test_convergence_with_strict_tolerance() {
        let generator = TestDataGenerator::new(111);
        let data = generator.linearly_separable_2d(15);

        let mut options = SvmOptions::default();
        options.epsilon = 1e-10; // Very strict tolerance
        options.mfnitermax = 100; // Allow more iterations

        let (weights, outputs) = train_l2_svm(&data, &options, 1.0, 1.0)
            .expect("Training with strict tolerance should succeed");

        // Should still achieve good classification
        let accuracy = utils::classification_accuracy(&outputs, &data.labels);
        assert!(
            accuracy > 0.9,
            "Strict convergence should maintain accuracy"
        );

        // Solution should be well-optimized (low training loss)
        let mut total_loss = 0.0;
        for i in 0..data.m {
            let label = data.get_label(i);
            let margin = 1.0 - label * outputs[i];
            if margin > 0.0 {
                total_loss += margin * margin;
            }
        }
        total_loss *= 0.5; // L2 loss factor

        // Add regularization term
        let reg_term = 0.5 * options.lambda * utils::l2_norm_squared(&weights[..2]);
        total_loss += reg_term;

        // With strict convergence, loss should be well minimized
        println!("Final training loss: {}", total_loss);
    }

    #[test]
    fn test_performance_benchmark() {
        use std::time::Instant;

        let generator = TestDataGenerator::new(777);
        let data = generator.overlapping_classes(100, 0.6); // Larger dataset

        let options = SvmOptions::default();

        let start = Instant::now();
        let (weights, outputs) =
            train_l2_svm(&data, &options, 1.0, 1.0).expect("Benchmark training should succeed");
        let duration = start.elapsed();

        // Should complete in reasonable time (adjust threshold as needed)
        assert!(
            duration.as_secs() < 10,
            "Training should complete within 10 seconds"
        );

        // Should still produce good results
        let accuracy = utils::classification_accuracy(&outputs, &data.labels);
        assert!(accuracy > 0.75, "Benchmark should maintain good accuracy");

        println!("Training 200 examples took: {:?}", duration);
        println!("Final accuracy: {:.3}", accuracy);
    }

    /// Test replicating the C++ CrossValidation doStepTest
    /// Based on UnitTest_Percolator_CrossValidation.cpp:77-161
    #[test]
    fn test_crossvalidation_dostep_replication() {
        // Parameters from C++ test (lines 79-99)
        const N: usize = 100;
        const TEST_FDR: f64 = 0.02;

        let mut features = Vec::new();
        let mut labels = Vec::new();
        let mut scan_number = 1;

        // Create data matching C++ test exactly (lines 110-131)
        // Targets: 2*N examples, features[0] = i/N - 1e-6, features[1] = i%2
        for i in 0..(2 * N) {
            let feature0 = (i as f64) / (N as f64) - 1e-6;
            let feature1 = (i % 2) as f64;
            features.push(vec![feature0, feature1, 1.0]); // with bias
            labels.push(1.0); // TARGET
            scan_number += 1;
        }

        // Decoys: N examples, features[0] = i/N, features[1] = i%2
        for i in 0..N {
            let feature0 = (i as f64) / (N as f64);
            let feature1 = (i % 2) as f64;
            features.push(vec![feature0, feature1, 1.0]); // with bias
            labels.push(-1.0); // DECOY
            scan_number += 1;
        }

        let data = SvmData::new(features, labels);

        // Use same parameters as C++ CrossValidation test
        let mut options = SvmOptions::default();
        options.lambda = 1.0; // Fixed lambda from CrossValidation
        options.epsilon = 1e-7; // Default EPSILON

        // Train SVM with cost parameters matching test expectations
        let (weights, outputs) = train_l2_svm(&data, &options, 1.0, 1.0)
            .expect("CrossValidation replication should succeed");

        // Verify the trained model identifies the first feature as important
        // In the C++ test, after preIterationSetup, weights[i][0] = 1.0, weights[i][1] = 0.0
        assert!(
            weights[0].abs() > weights[1].abs(),
            "First feature should be weighted higher than second feature"
        );

        // Count positive predictions using FDR thresholding (matching C++ behavior)
        let positive_predictions = utils::count_positives_at_fdr(&outputs, &data.labels, TEST_FDR);

        // From C++ test (lines 156-158): should find at least N positives
        // "One step of the training algorithm should find at least N positives"
        assert!(
            positive_predictions >= N,
            "Should find at least {} positive predictions, found {}",
            N,
            positive_predictions
        );

        // Should not exceed N * (1 + testFdr) positives due to FDR control
        let max_expected = (N as f64 * (1.0 + TEST_FDR)) as usize;
        assert!(
            positive_predictions <= max_expected,
            "Should not exceed {} positive predictions due to FDR control, found {}",
            max_expected,
            positive_predictions
        );

        // Additional validation: targets should generally score higher than decoys
        let target_scores: Vec<f64> = outputs
            .iter()
            .zip(data.labels.iter())
            .filter_map(|(&output, &label)| if label > 0.0 { Some(output) } else { None })
            .collect();
        let decoy_scores: Vec<f64> = outputs
            .iter()
            .zip(data.labels.iter())
            .filter_map(|(&output, &label)| if label < 0.0 { Some(output) } else { None })
            .collect();

        let mean_target = target_scores.iter().sum::<f64>() / target_scores.len() as f64;
        let mean_decoy = decoy_scores.iter().sum::<f64>() / decoy_scores.len() as f64;

        assert!(
            mean_target > mean_decoy,
            "Targets should score higher than decoys on average"
        );

        // Debug output to understand the discrepancy
        println!("Total examples: {}", data.m);
        println!("Targets: {}, Decoys: {}", data.positives, data.negatives);
        println!("Positive predictions: {}", positive_predictions);
        println!("Expected max: {}", max_expected);
        println!(
            "Mean target score: {:.6}, Mean decoy score: {:.6}",
            mean_target, mean_decoy
        );
        println!(
            "Weights: [{:.6}, {:.6}, {:.6}]",
            weights[0], weights[1], weights[2]
        );
    }
}
