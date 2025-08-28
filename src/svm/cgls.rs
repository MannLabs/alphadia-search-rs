//! Conjugate Gradient Least Squares solver
//!
//! Direct port of the CGLS algorithm from Percolator's ssl.cpp (lines 93-192).
//! This implements conjugate gradient method for solving sparse linear least squares problems.

use crate::svm::data::{SvmData, SvmVector};

/// CGLS solver for sparse linear least squares problems
///
/// Solves: min_w 0.5*lambda*w'*w + 0.5*sum_{i in Subset} C[i] * (Y[i] - w' * x_i)^2
/// over a subset of examples x_i specified by the active set
pub struct CglsSolver<'a> {
    data: &'a SvmData,
    lambda: f64,
    epsilon: f64,
    max_iterations: usize,
}

impl<'a> CglsSolver<'a> {
    /// Create new CGLS solver
    pub fn new(data: &'a SvmData, lambda: f64, epsilon: f64, max_iterations: usize) -> Self {
        Self {
            data,
            lambda,
            epsilon,
            max_iterations,
        }
    }

    /// Solve CGLS problem for given active set
    ///
    /// # Arguments
    /// * `active_set` - Indices of active training examples
    /// * `weights` - Input/output weight vector (modified in place)
    /// * `outputs` - Input/output predictions (modified in place)
    /// * `cpos` - Cost parameter for positive examples
    /// * `cneg` - Cost parameter for negative examples
    ///
    /// # Returns
    /// * `true` if converged to optimality, `false` if max iterations reached
    pub fn solve(
        &self,
        active_set: &[usize],
        weights: &mut SvmVector,
        outputs: &mut SvmVector,
        cpos: f64,
        cneg: f64,
    ) -> bool {
        let active = active_set.len();
        let n = self.data.n;

        if active == 0 {
            return true;
        }

        // Initialize working vectors
        let mut z = vec![0.0; active];
        let mut q = vec![0.0; active];
        let mut r = vec![0.0; n];
        let mut p = vec![0.0; n];

        // Create active feature matrix (transpose for efficiency)
        let mut feature_matrix = vec![vec![0.0; n]; active];
        for (i, &example_idx) in active_set.iter().enumerate() {
            let features = self.data.get_features(example_idx);
            feature_matrix[i].copy_from_slice(features);
        }

        // Initialize z vector: z[i] = C[i] * (Y[i] - o[i])
        for (i, &example_idx) in active_set.iter().enumerate() {
            let label = self.data.get_label(example_idx);
            let cost = if label > 0.0 { cpos } else { cneg };
            z[i] = cost * (label - outputs.vec[example_idx]);
        }

        // Initialize r = sum_i z[i] * x_i - lambda * w
        r.fill(0.0);
        for (i, &example_idx) in active_set.iter().enumerate() {
            let features = self.data.get_features(example_idx);
            for j in 0..n {
                r[j] += z[i] * features[j];
            }
        }
        for j in 0..n {
            r[j] -= self.lambda * weights.vec[j];
        }

        // p = r (initial search direction)
        p.copy_from_slice(&r);

        // omega1 = r^T * r
        let mut omega1 = r.iter().map(|&x| x * x).sum::<f64>();
        let mut omega_p = omega1;

        let epsilon2 = self.epsilon * self.epsilon;

        // Main CGLS iteration
        for _iter in 0..self.max_iterations {
            // Compute q = A^T * p (matrix-vector multiplication)
            self.compute_atx(&feature_matrix, &p, &mut q, active_set, cpos, cneg);

            // omega_q = sum_i C[i] * q[i]^2
            let omega_q = active_set
                .iter()
                .enumerate()
                .map(|(i, &example_idx)| {
                    let label = self.data.get_label(example_idx);
                    let cost = if label > 0.0 { cpos } else { cneg };
                    cost * q[i] * q[i]
                })
                .sum::<f64>();

            // Compute step size: gamma = omega1 / (lambda * omega_p + omega_q)
            let gamma = omega1 / (self.lambda * omega_p + omega_q);

            // Update weights: w = w + gamma * p
            for j in 0..n {
                weights.vec[j] += gamma * p[j];
            }

            // Update outputs and residual
            for (i, &example_idx) in active_set.iter().enumerate() {
                outputs.vec[example_idx] += gamma * q[i];
                let label = self.data.get_label(example_idx);
                let cost = if label > 0.0 { cpos } else { cneg };
                z[i] -= gamma * cost * q[i];
            }

            // Update residual: r = sum_i z[i] * x_i - lambda * w
            r.fill(0.0);
            for (i, &example_idx) in active_set.iter().enumerate() {
                let features = self.data.get_features(example_idx);
                for j in 0..n {
                    r[j] += z[i] * features[j];
                }
            }
            for j in 0..n {
                r[j] -= self.lambda * weights.vec[j];
            }

            let omega1_new = r.iter().map(|&x| x * x).sum::<f64>();

            // Check convergence: ||r||^2 < epsilon^2 * ||z||^2
            let omega_z = z.iter().map(|&x| x * x).sum::<f64>();
            if omega1_new < epsilon2 * omega_z {
                return true; // Converged
            }

            // Update search direction: p = r + beta * p
            let beta = omega1_new / omega1;
            for j in 0..n {
                p[j] = r[j] + beta * p[j];
            }

            omega1 = omega1_new;
            omega_p = p.iter().map(|&x| x * x).sum::<f64>();
        }

        false // Did not converge within max iterations
    }

    /// Compute matrix-vector product set2^T * p as in the C++ CGLS implementation
    /// C++ does: dgemv_('T', &n, &active, &alpha, set2, &n, p, &inc, &beta, q, &inc);
    /// This computes q = set2^T * p where set2 is active x n matrix
    fn compute_atx(
        &self,
        feature_matrix: &[Vec<f64>],
        x: &[f64],
        result: &mut [f64],
        active_set: &[usize],
        _cpos: f64,
        _cneg: f64,
    ) {
        let n = x.len();
        result.fill(0.0);

        // C++ stores set2 as row-major active x n matrix
        // dgemv with trans='T' computes: q = set2^T * p
        // So we need: result[i] = sum_j feature_matrix[i][j] * x[j]
        for (i, &_example_idx) in active_set.iter().enumerate() {
            let mut dot_product = 0.0;
            for j in 0..n {
                dot_product += feature_matrix[i][j] * x[j];
            }
            result[i] = dot_product;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::svm::data::SvmData;
    use approx::assert_relative_eq;

    #[test]
    fn test_cgls_simple_problem() {
        // Simple 2D problem: minimize ||Ax - b||^2 + lambda*||x||^2
        let features = vec![
            vec![1.0, 0.0, 1.0], // [1, 0] + bias
            vec![0.0, 1.0, 1.0], // [0, 1] + bias
            vec![1.0, 1.0, 1.0], // [1, 1] + bias
        ];
        let labels = vec![1.0, 1.0, 2.0]; // Target values

        let data = SvmData::new(features, labels);
        let solver = CglsSolver::new(&data, 0.1, 1e-6, 1000);

        let mut weights = SvmVector::new(3);
        let mut outputs = SvmVector::new(3);

        let active_set = vec![0, 1, 2]; // All examples active
        let converged = solver.solve(&active_set, &mut weights, &mut outputs, 1.0, 1.0);

        assert!(converged, "CGLS should converge");

        // Check that outputs are reasonable
        assert!(outputs.vec[0] > 0.0);
        assert!(outputs.vec[1] > 0.0);
        assert!(outputs.vec[2] > outputs.vec[0]); // Should be larger since label is 2.0
    }

    #[test]
    fn test_cgls_empty_active_set() {
        let features = vec![vec![1.0, 1.0]];
        let labels = vec![1.0];
        let data = SvmData::new(features, labels);

        let solver = CglsSolver::new(&data, 1.0, 1e-6, 100);
        let mut weights = SvmVector::new(2);
        let mut outputs = SvmVector::new(1);

        let active_set: Vec<usize> = vec![]; // Empty active set
        let converged = solver.solve(&active_set, &mut weights, &mut outputs, 1.0, 1.0);

        assert!(converged, "Empty active set should return immediately");
    }

    #[test]
    fn test_cgls_single_example() {
        let features = vec![vec![2.0, 1.0]]; // Single example
        let labels = vec![3.0];
        let data = SvmData::new(features, labels);

        let solver = CglsSolver::new(&data, 0.01, 1e-6, 1000);
        let mut weights = SvmVector::new(2);
        let mut outputs = SvmVector::new(1);

        let active_set = vec![0];
        let converged = solver.solve(&active_set, &mut weights, &mut outputs, 1.0, 1.0);

        assert!(converged, "Single example should converge");

        // With low regularization, output should be close to label
        assert_relative_eq!(outputs.vec[0], 3.0, epsilon = 0.1);
    }
}
