use numpy::ndarray::{Array1, Array2};

/// Scalar implementation of log-dot-product operation
pub fn axis_log_dot_product_scalar(array: &Array2<f32>, weights: &[f32]) -> Array1<f32> {
    let (n_rows, n_cols) = array.dim();

    // Check that the number of rows matches the number of weights
    assert_eq!(
        n_rows,
        weights.len(),
        "Number of rows in array must match the length of weights vector"
    );

    let mut result = Array1::zeros(n_cols);

    for i in 0..n_rows {
        for j in 0..n_cols {
            // Apply log transformation and then weighted sum
            let val = (array[[i, j]] + 1.0).ln();
            result[j] += val * weights[i];
        }
    }

    result
}

/// Scalar implementation of sqrt-dot-product operation
pub fn axis_sqrt_dot_product_scalar(array: &Array2<f32>, weights: &[f32]) -> Array1<f32> {
    let (n_rows, n_cols) = array.dim();

    // Check that the number of rows matches the number of weights
    assert_eq!(
        n_rows,
        weights.len(),
        "Number of rows in array must match the length of weights vector"
    );

    let mut result = Array1::zeros(n_cols);

    for i in 0..n_rows {
        for j in 0..n_cols {
            // Apply square root transformation and then weighted sum
            // Add small epsilon to avoid sqrt(negative)
            let val = (array[[i, j]].max(0.0)).sqrt();
            result[j] += val * weights[i];
        }
    }

    result
}
