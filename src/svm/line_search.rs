//! Line search optimization
//!
//! Direct port of the line_search function from Percolator's ssl.cpp (lines 332-388).
//! This implements the line search algorithm for finding optimal step size in MFN iterations.

use crate::svm::data::SvmData;

/// Delta structure for line search algorithm
///
/// Corresponds to the Delta class from ssl.h
#[derive(Debug, Clone)]
struct Delta {
    delta: f64,
    index: usize,
    s: i32,
}

impl Delta {
    fn new(delta: f64, index: usize, s: i32) -> Self {
        Self { delta, index, s }
    }
}

/// Line search optimizer for Modified Finite Newton method
///
/// Finds optimal step size along search direction to minimize objective function
pub struct LineSearch<'a> {
    #[allow(dead_code)]
    data: &'a SvmData,
}

impl<'a> LineSearch<'a> {
    /// Create new line search optimizer
    pub fn new(data: &'a SvmData) -> Self {
        Self { data }
    }

    /// Perform line search to find optimal step size
    ///
    /// # Arguments
    /// * `w` - Current weight vector
    /// * `w_bar` - Search direction (new weight vector)
    /// * `lambda` - Regularization parameter
    /// * `o` - Current output vector
    /// * `o_bar` - New output vector
    /// * `labels` - Training labels
    /// * `cpos` - Cost parameter for positive examples
    /// * `cneg` - Cost parameter for negative examples
    ///
    /// # Returns
    /// Optimal step size alpha in [0, 1] such that w_new = (1-alpha)*w + alpha*w_bar
    #[allow(clippy::too_many_arguments)]
    pub fn search(
        &self,
        w: &[f64],
        w_bar: &[f64],
        lambda: f64,
        o: &[f64],
        o_bar: &[f64],
        labels: &[f64],
        cpos: f64,
        cneg: f64,
    ) -> f64 {
        let d = w.len(); // dimensionality
        let l = labels.len(); // number of examples

        // Compute directional derivatives of regularization term
        let mut omega_l = 0.0;
        let mut omega_r = 0.0;

        for i in 0..d {
            let diff = w_bar[i] - w[i];
            omega_l += w[i] * diff;
            omega_r += w_bar[i] * diff;
        }

        omega_l *= lambda;
        omega_r *= lambda;

        let mut left = omega_l; // f'(0-)
        let mut right = omega_r; // f'(1+)

        // Collect breakpoints where derivative changes
        let mut deltas = Vec::new();

        for i in 0..l {
            let label = labels[i];
            let cost = if label > 0.0 { cpos } else { cneg };
            let output_diff = label * (o_bar[i] - o[i]);

            if label * o[i] < 1.0 {
                // Point is in the margin (loss > 0)
                let d2 = cost * (o_bar[i] - o[i]);
                left += (o[i] - label) * d2;
                right += (o_bar[i] - label) * d2;

                if output_diff > 0.0 {
                    // Derivative will decrease at this point
                    let delta = (1.0 - label * o[i]) / output_diff;
                    deltas.push(Delta::new(delta, i, -1));
                }
            } else {
                // Point is outside margin (loss = 0)
                if output_diff < 0.0 {
                    // Derivative will increase at this point
                    let delta = (1.0 - label * o[i]) / output_diff;
                    deltas.push(Delta::new(delta, i, 1));
                }
            }
        }

        // Sort breakpoints by delta value
        deltas.sort_by(|a, b| a.delta.partial_cmp(&b.delta).unwrap());

        // Find optimal step size by checking derivative at each breakpoint
        for delta in &deltas {
            let delta_prime = left + delta.delta * (right - left);

            if delta_prime >= 0.0 {
                // Found sign change in derivative - optimal point
                break;
            }

            // Update derivatives for next interval
            let idx = delta.index;
            let label = labels[idx];
            let cost = if label > 0.0 { cpos } else { cneg };
            let diff = (delta.s as f64) * cost * (o_bar[idx] - o[idx]);

            left += diff * (o[idx] - label);
            right += diff * (o_bar[idx] - label);
        }

        // Compute final step size
        if (right - left).abs() < 1e-12 {
            // Avoid division by zero
            0.0
        } else {
            (-left / (right - left)).clamp(0.0, 1.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::svm::data::SvmData;
    use approx::assert_relative_eq;

    #[test]
    fn test_line_search_simple() {
        // Simple case where line search should find reasonable step size
        let features = vec![vec![1.0, 0.0, 1.0], vec![0.0, 1.0, 1.0]];
        let labels = vec![1.0, -1.0];
        let data = SvmData::new(features, labels.clone());

        let line_search = LineSearch::new(&data);

        // Current weights and outputs
        let w = vec![0.1, 0.1, 0.0];
        let o = vec![0.1, 0.1];

        // New weights and outputs (better separation)
        let w_bar = vec![1.0, -1.0, 0.0];
        let o_bar = vec![1.0, -1.0];

        let step_size = line_search.search(&w, &w_bar, 0.1, &o, &o_bar, &labels, 1.0, 1.0);

        // Should find a reasonable step size between 0 and 1
        assert!(step_size >= 0.0);
        assert!(step_size <= 1.0);

        // With good separation, should take a substantial step
        assert!(step_size > 0.1, "Should take a meaningful step size");
    }

    #[test]
    fn test_line_search_no_improvement() {
        // Case where new direction is worse - should take small step
        let features = vec![vec![1.0, 1.0], vec![-1.0, 1.0]];
        let labels = vec![1.0, -1.0];
        let data = SvmData::new(features, labels.clone());

        let line_search = LineSearch::new(&data);

        // Good current solution
        let w = vec![1.0, 0.0];
        let o = vec![1.0, -1.0];

        // Worse proposed solution
        let w_bar = vec![-1.0, 0.0];
        let o_bar = vec![-1.0, 1.0];

        let step_size = line_search.search(&w, &w_bar, 1.0, &o, &o_bar, &labels, 1.0, 1.0);

        // Should take very small step or no step
        assert!(step_size >= 0.0);
        assert!(step_size <= 1.0);
        assert!(
            step_size < 0.5,
            "Should not take large step toward worse solution"
        );
    }

    #[test]
    fn test_line_search_zero_direction() {
        // Case where search direction is zero
        let features = vec![vec![1.0, 1.0]];
        let labels = vec![1.0];
        let data = SvmData::new(features, labels.clone());

        let line_search = LineSearch::new(&data);

        let w = vec![0.5, 0.5];
        let o = vec![1.0];

        // Same weights and outputs (zero direction)
        let w_bar = vec![0.5, 0.5];
        let o_bar = vec![1.0];

        let step_size = line_search.search(&w, &w_bar, 1.0, &o, &o_bar, &labels, 1.0, 1.0);

        // Should handle zero direction gracefully
        assert!(step_size >= 0.0);
        assert!(step_size <= 1.0);
    }

    #[test]
    fn test_line_search_boundary_cases() {
        // Test various boundary conditions
        let features = vec![vec![2.0, 1.0], vec![-1.0, 1.0]];
        let labels = vec![1.0, -1.0];
        let data = SvmData::new(features, labels.clone());

        let line_search = LineSearch::new(&data);

        // Test with extreme cost parameters
        let w = vec![0.0, 0.0];
        let w_bar = vec![1.0, 1.0];
        let o = vec![0.0, 0.0];
        let o_bar = vec![2.0, -1.0];

        // Very high cost for positives
        let step_size_high_cpos =
            line_search.search(&w, &w_bar, 0.1, &o, &o_bar, &labels, 100.0, 1.0);

        // Very high cost for negatives
        let step_size_high_cneg =
            line_search.search(&w, &w_bar, 0.1, &o, &o_bar, &labels, 1.0, 100.0);

        // Both should be valid step sizes
        assert!(step_size_high_cpos >= 0.0 && step_size_high_cpos <= 1.0);
        assert!(step_size_high_cneg >= 0.0 && step_size_high_cneg <= 1.0);
    }
}
