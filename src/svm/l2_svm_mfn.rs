//! L2-SVM with Modified Finite Newton method
//!
//! Direct port of the L2_SVM_MFN algorithm from Percolator's ssl.cpp (lines 194-330).
//! This is the core SVM training algorithm that solves:
//! min_w 0.5*lambda*w'*w + 0.5*sum_i C[i] * max(0, 1 - Y[i]*w'*x[i])^2

use crate::svm::cgls::CglsSolver;
use crate::svm::data::{SvmData, SvmOptions, SvmVector};
use crate::svm::line_search::LineSearch;

/// Convergence criteria constants (from ssl.h)
const BIG_EPSILON: f64 = 0.01;
const RELATIVE_STOP_EPS: f64 = 1e-9;

/// L2-SVM trainer with Modified Finite Newton method
///
/// This implements the main L2-SVM training algorithm from Percolator
pub struct L2SvmMfn<'a> {
    data: &'a SvmData,
    options: &'a SvmOptions,
}

impl<'a> L2SvmMfn<'a> {
    /// Create new L2-SVM trainer
    pub fn new(data: &'a SvmData, options: &'a SvmOptions) -> Self {
        Self { data, options }
    }

    /// Train L2-SVM model using Modified Finite Newton method
    ///
    /// # Arguments
    /// * `cpos` - Cost parameter for positive examples
    /// * `cneg` - Cost parameter for negative examples
    ///
    /// # Returns
    /// Tuple of (weights, outputs) where:
    /// - weights: learned SVM weight vector (including bias)
    /// - outputs: SVM predictions for training examples
    pub fn train(
        &mut self,
        cpos: f64,
        cneg: f64,
    ) -> Result<(Vec<f64>, Vec<f64>), Box<dyn std::error::Error>> {
        let m = self.data.m;
        let n = self.data.n;

        // Initialize weight and output vectors
        let mut weights = SvmVector::new(n);
        let mut outputs = SvmVector::new(m);

        // Compute initial outputs and objective
        let mut f_old = 0.5 * self.options.lambda * weights.vec.iter().map(|&w| w * w).sum::<f64>();

        // Find initial active set (examples in the margin)
        let mut active_set = self.find_active_set(&outputs, cpos, cneg);
        let mut f = f_old + self.compute_loss(&active_set, &outputs, cpos, cneg);

        // Working vectors for MFN iterations
        let mut weights_bar = SvmVector::new(n);
        let mut outputs_bar = SvmVector::new(m);

        let cgls_solver = CglsSolver::new(
            self.data,
            self.options.lambda,
            self.options.epsilon,
            self.options.cgitermax,
        );
        let line_search = LineSearch::new(self.data);

        // Main MFN iterations
        for iter in 0..self.options.mfnitermax {
            println!(
                "L2_SVM_MFN Iteration {} ({} active examples, objective = {})",
                iter + 1,
                active_set.len(),
                f
            );

            // Copy current solution to working vectors
            weights_bar.vec.copy_from_slice(&weights.vec);
            outputs_bar.vec.copy_from_slice(&outputs.vec);

            // Solve CGLS subproblem on active set
            let mut epsilon = if iter == 0 {
                BIG_EPSILON
            } else {
                self.options.epsilon
            };
            let cgls_converged =
                cgls_solver.solve(&active_set, &mut weights_bar, &mut outputs_bar, cpos, cneg);

            // Compute outputs for all examples with new weights
            self.compute_all_outputs(&weights_bar, &mut outputs_bar);

            // Log target true positives at different FDR thresholds
            self.log_target_true_positives(&outputs_bar, iter + 1);

            // Check optimality conditions
            let mut optimal = cgls_converged;
            if optimal {
                optimal = self.check_optimality(&active_set, &outputs_bar, epsilon, cpos, cneg)?;
            }

            if optimal {
                if epsilon == BIG_EPSILON {
                    // First pass converged, use stricter tolerance
                    #[allow(unused_assignments)]
                    {
                        epsilon = self.options.epsilon;
                    }
                    println!("  epsilon = {} case converged (speedup heuristic). Continuing with epsilon = {}",
                            BIG_EPSILON, self.options.epsilon);
                    continue;
                } else {
                    // Final convergence
                    weights.vec.copy_from_slice(&weights_bar.vec);
                    outputs.vec.copy_from_slice(&outputs_bar.vec);
                    println!(
                        "L2_SVM_MFN converged (optimality) in {} iterations",
                        iter + 1
                    );
                    break;
                }
            }

            // Perform line search to find optimal step size
            let step_size = line_search.search(
                &weights.vec,
                &weights_bar.vec,
                self.options.lambda,
                &outputs.vec,
                &outputs_bar.vec,
                &self.data.labels,
                cpos,
                cneg,
            );

            f_old = f;

            // Update solution: w = (1-alpha)*w + alpha*w_bar
            for i in 0..n {
                weights.vec[i] =
                    (1.0 - step_size) * weights.vec[i] + step_size * weights_bar.vec[i];
            }

            // Recompute objective and active set
            f = 0.5 * self.options.lambda * weights.vec.iter().map(|&w| w * w).sum::<f64>();

            for i in 0..m {
                outputs.vec[i] =
                    (1.0 - step_size) * outputs.vec[i] + step_size * outputs_bar.vec[i];
            }

            active_set = self.find_active_set(&outputs, cpos, cneg);
            f += self.compute_loss(&active_set, &outputs, cpos, cneg);

            // Check relative stopping criterion
            if (f - f_old).abs() < RELATIVE_STOP_EPS * f_old.abs() {
                println!(
                    "L2_SVM_MFN converged (relative improvement) in {} iterations",
                    iter + 1
                );
                break;
            }
        }

        Ok((weights.vec, outputs.vec))
    }

    /// Find active set (examples with non-zero loss)
    fn find_active_set(&self, outputs: &SvmVector, _cpos: f64, _cneg: f64) -> Vec<usize> {
        let mut active = Vec::new();

        for i in 0..self.data.m {
            let label = self.data.get_label(i);
            let margin = 1.0 - label * outputs.vec[i];

            if margin > 0.0 {
                active.push(i);
            }
        }

        active
    }

    /// Compute loss function value for active examples
    fn compute_loss(&self, active_set: &[usize], outputs: &SvmVector, cpos: f64, cneg: f64) -> f64 {
        let mut loss = 0.0;

        for &i in active_set {
            let label = self.data.get_label(i);
            let cost = if label > 0.0 { cpos } else { cneg };
            let margin = 1.0 - label * outputs.vec[i];

            if margin > 0.0 {
                loss += 0.5 * cost * margin * margin;
            }
        }

        loss
    }

    /// Compute outputs for all examples using given weights
    fn compute_all_outputs(&self, weights: &SvmVector, outputs: &mut SvmVector) {
        for i in 0..self.data.m {
            let features = self.data.get_features(i);
            let mut output = 0.0;

            for j in 0..self.data.n {
                output += weights.vec[j] * features[j];
            }

            outputs.vec[i] = output;
        }
    }

    /// Check KKT optimality conditions
    fn check_optimality(
        &self,
        active_set: &[usize],
        outputs: &SvmVector,
        epsilon: f64,
        _cpos: f64,
        _cneg: f64,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        // Check that all active examples satisfy margin constraint
        for &i in active_set {
            let label = self.data.get_label(i);
            if label * outputs.vec[i] > 1.0 + epsilon {
                return Ok(false);
            }
        }

        // Check that all inactive examples satisfy complementary slackness
        let mut active_set_lookup = vec![false; self.data.m];
        for &i in active_set {
            active_set_lookup[i] = true;
        }

        for i in 0..self.data.m {
            if !active_set_lookup[i] {
                let label = self.data.get_label(i);
                if label * outputs.vec[i] < 1.0 - epsilon {
                    return Ok(false);
                }
            }
        }

        Ok(true)
    }

    /// Log target true positives at different FDR thresholds
    fn log_target_true_positives(&self, outputs: &SvmVector, iteration: usize) {
        use crate::svm::utils::count_positives_at_fdr;

        // Create score and label vectors for FDR calculation
        let mut scores = Vec::new();
        let mut labels = Vec::new();

        for i in 0..self.data.m {
            let label = self.data.get_label(i);
            let score = outputs.vec[i];

            scores.push(score);
            // Convert labels: positive examples (targets) = 1.0, negative (decoys) = -1.0 -> 1.0, 0.0
            labels.push(if label > 0.0 { 1.0 } else { 0.0 });
        }

        // Count target true positives at different FDR thresholds
        let tp_at_01 = count_positives_at_fdr(&scores, &labels, 0.01);
        let tp_at_05 = count_positives_at_fdr(&scores, &labels, 0.05);
        let tp_at_10 = count_positives_at_fdr(&scores, &labels, 0.10);
        let tp_at_50 = count_positives_at_fdr(&scores, &labels, 0.50);

        println!(
            "  Iteration {iteration}: Target true positives at FDR - 1%: {tp_at_01}, 5%: {tp_at_05}, 10%: {tp_at_10}, 50%: {tp_at_50}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::svm::data::SvmData;
    use approx::assert_relative_eq;

    #[test]
    fn test_l2_svm_mfn_separable_data() {
        // Create linearly separable 2D dataset
        let features = vec![
            vec![2.0, 1.0, 1.0], // positive examples
            vec![1.5, 2.0, 1.0],
            vec![-1.0, -2.0, 1.0], // negative examples
            vec![-2.0, -1.0, 1.0],
        ];
        let labels = vec![1.0, 1.0, -1.0, -1.0];

        let data = SvmData::new(features, labels);
        let options = SvmOptions::default();

        let mut trainer = L2SvmMfn::new(&data, &options);
        let (weights, outputs) = trainer.train(1.0, 1.0).expect("Training should succeed");

        // Check that learned weights are reasonable
        assert_eq!(weights.len(), 3);
        assert_eq!(outputs.len(), 4);

        // Check that outputs have correct signs
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

        // Check that margin constraints are satisfied (approximately)
        for i in 0..4 {
            let label = data.get_label(i);
            let margin = label * outputs[i];
            assert!(
                margin > 0.5,
                "Example {} should be well-classified (margin = {})",
                i,
                margin
            );
        }
    }

    #[test]
    fn test_l2_svm_mfn_noisy_data() {
        // Create dataset with some noise/overlapping classes
        let features = vec![
            vec![1.0, 1.0, 1.0],
            vec![1.2, 0.8, 1.0],
            vec![0.8, 1.2, 1.0], // positive examples
            vec![-1.0, -1.0, 1.0],
            vec![-0.8, -1.2, 1.0], // negative examples
            vec![0.1, -0.1, 1.0],  // noisy example (close to boundary)
        ];
        let labels = vec![1.0, 1.0, 1.0, -1.0, -1.0, -1.0];

        let data = SvmData::new(features, labels);
        let options = SvmOptions::default();

        let mut trainer = L2SvmMfn::new(&data, &options);
        let (weights, outputs) = trainer.train(1.0, 1.0).expect("Training should succeed");

        // Should still learn reasonable classifier
        assert_eq!(weights.len(), 3);
        assert_eq!(outputs.len(), 6);

        // Most examples should be classified correctly
        let mut correct = 0;
        for i in 0..6 {
            let label = data.get_label(i);
            if label * outputs[i] > 0.0 {
                correct += 1;
            }
        }

        assert!(
            correct >= 4,
            "Should classify at least 4/6 examples correctly"
        );
    }

    #[test]
    fn test_l2_svm_mfn_different_costs() {
        // Test with different cost parameters for positive/negative classes
        let features = vec![vec![1.0, 1.0], vec![-1.0, 1.0]];
        let labels = vec![1.0, -1.0];

        let data = SvmData::new(features, labels);
        let options = SvmOptions::default();

        let mut trainer = L2SvmMfn::new(&data, &options);

        // Train with high cost for positive examples
        let (weights_high_pos, _) = trainer.train(10.0, 1.0).expect("Training should succeed");

        // Train with high cost for negative examples
        let (weights_high_neg, _) = trainer.train(1.0, 10.0).expect("Training should succeed");

        // The learned classifiers should be different
        let weight_diff: f64 = weights_high_pos
            .iter()
            .zip(weights_high_neg.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        assert!(
            weight_diff > 0.1,
            "Different cost parameters should produce different models"
        );
    }

    #[test]
    fn test_l2_svm_mfn_single_example() {
        // Edge case: single training example
        let features = vec![vec![1.0, 2.0, 1.0]];
        let labels = vec![1.0];

        let data = SvmData::new(features, labels);
        let options = SvmOptions::default();

        let mut trainer = L2SvmMfn::new(&data, &options);
        let (weights, outputs) = trainer.train(1.0, 1.0).expect("Training should succeed");

        assert_eq!(weights.len(), 3);
        assert_eq!(outputs.len(), 1);

        // With single positive example, output should be positive
        assert!(outputs[0] > 0.0);
    }
}
