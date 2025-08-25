use numpy::ndarray::{Array1, Array2};

#[cfg(test)]
mod tests;

/// Calculate observed fragment m/z values and mass errors from dense XIC/m/z observations
///
/// Returns (mz_observed, mass_error_observed) where:
/// - mz_observed: Weighted average m/z for each fragment (0 if no signal)
/// - mass_error_observed: Mass error in ppm (0 if no observation)
pub fn calculate_fragment_mz_and_errors(
    dense_mz: &Array2<f32>,
    intensity_observed: &Array2<f32>,
    mz_library: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let mz_observed_array = weighted_mean_nonzero_axis_1(dense_mz, intensity_observed);
    let num_fragments = mz_library.len();

    let mut mz_observed = vec![0.0f32; num_fragments];
    let mut mass_error_observed = vec![0.0f32; num_fragments];

    for fragment_idx in 0..num_fragments {
        let observed_mz = mz_observed_array[fragment_idx];
        mz_observed[fragment_idx] = observed_mz;

        // Calculate mass error in ppm (0 if no observation)
        if observed_mz > 0.0 {
            mass_error_observed[fragment_idx] =
                ((observed_mz - mz_library[fragment_idx]) / mz_library[fragment_idx]) * 1e6;
        } else {
            mass_error_observed[fragment_idx] = 0.0;
        }
    }

    (mz_observed, mass_error_observed)
}

/// Calculate weighted mean along axis 1, excluding zero weights
/// Returns a vector where each element is the weighted mean for that row
/// Values with zero weight are excluded from the calculation
pub fn weighted_mean_nonzero_axis_1(values: &Array2<f32>, weights: &Array2<f32>) -> Array1<f32> {
    let n_rows = values.nrows();
    let n_cols = values.ncols();
    let mut result = Array1::zeros(n_rows);

    for row_idx in 0..n_rows {
        let mut weighted_sum = 0.0f32;
        let mut weight_sum = 0.0f32;

        for col_idx in 0..n_cols {
            let weight = weights[[row_idx, col_idx]];

            // Only include values with non-zero weight
            if weight > 0.0 {
                let value = values[[row_idx, col_idx]];
                weighted_sum += value * weight;
                weight_sum += weight;
            }
        }

        // Return weighted average or 0 if no valid weights
        result[row_idx] = if weight_sum > 0.0 {
            weighted_sum / weight_sum
        } else {
            0.0
        };
    }

    result
}

/// Calculate weighted mean absolute mass error
///
/// Weights the absolute mass errors by the provided intensities.
/// Returns 0.0 if all weights are zero or if there are no valid mass errors.
///
/// Parameters:
/// - mass_errors: Array of mass errors (in ppm)
/// - intensity_weights: Array of intensity weights (typically library intensities)
///
/// Returns:
/// - Weighted mean of absolute mass errors
pub fn calculate_weighted_mean_absolute_error(
    mass_errors: &[f32],
    intensity_weights: &[f32],
) -> f32 {
    if mass_errors.len() != intensity_weights.len() || mass_errors.is_empty() {
        return 0.0;
    }

    let mut weighted_sum = 0.0f32;
    let mut weight_sum = 0.0f32;

    for i in 0..mass_errors.len() {
        let weight = intensity_weights[i];
        // Only include fragments with non-zero weight and valid mass error
        if weight > 0.0 && mass_errors[i] != 0.0 {
            weighted_sum += mass_errors[i].abs() * weight;
            weight_sum += weight;
        }
    }

    if weight_sum > 0.0 {
        weighted_sum / weight_sum
    } else {
        0.0
    }
}
