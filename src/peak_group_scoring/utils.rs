use crate::constants::FragmentType;
use numpy::ndarray::Array2;
use std::f32;

/// Calculate the median along axis 0 (first axis) of a 2D array
/// Similar to np.median(array, axis=0) in NumPy
pub fn median_axis_0(array: &Array2<f32>) -> Vec<f32> {
    let (rows, cols) = array.dim();
    let mut result = Vec::with_capacity(cols);

    for col in 0..cols {
        let mut column_values: Vec<f32> = Vec::with_capacity(rows);
        for row in 0..rows {
            column_values.push(array[[row, col]]);
        }

        // Sort the column values to find median
        column_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = if rows % 2 == 0 {
            // Even number of elements: average of two middle values
            let mid = rows / 2;
            (column_values[mid - 1] + column_values[mid]) / 2.0
        } else {
            // Odd number of elements: middle value
            column_values[rows / 2]
        };

        result.push(median);
    }

    result
}

/// Calculate normalized intensity profiles from dense array.
/// Similar to normalize_profiles in Python
pub fn normalize_profiles(intensity_slice: &Array2<f32>, center_dilations: usize) -> Array2<f32> {
    let (rows, cols) = intensity_slice.dim();
    let center_idx = cols / 2;

    // Calculate mean center intensity for each row
    let mut center_intensity = Vec::with_capacity(rows);

    for i in 0..rows {
        let start_idx = center_idx.saturating_sub(center_dilations);
        let end_idx = std::cmp::min(center_idx + center_dilations + 1, cols);

        let mut sum = 0.0;
        let mut count = 0;

        for j in start_idx..end_idx {
            sum += intensity_slice[[i, j]];
            count += 1;
        }

        let mean_intensity = if count > 0 { sum / count as f32 } else { 0.0 };
        center_intensity.push(mean_intensity);
    }

    // Create normalized output array, initialized to zeros
    let mut normalized = Array2::zeros((rows, cols));

    // Only normalize profiles where center intensity > 0
    for i in 0..rows {
        if center_intensity[i] > 0.0 {
            for j in 0..cols {
                normalized[[i, j]] = intensity_slice[[i, j]] / center_intensity[i];
            }
        }
    }

    normalized
}

/// Calculate correlation between median profile and each row of dense_xic
/// Returns zero where no safe correlation can be calculated
pub fn correlation_axis_0(median_profile: &[f32], dense_xic: &Array2<f32>) -> Vec<f32> {
    let (rows, _cols) = dense_xic.dim();
    let mut correlations = Vec::with_capacity(rows);

    for row in 0..rows {
        let row_data: Vec<f32> = dense_xic.row(row).to_vec();
        let correlation = calculate_correlation_safe(median_profile, &row_data);
        correlations.push(correlation);
    }

    correlations
}

/// Calculate correlation between two arrays safely
/// Returns 0.0 if correlation cannot be calculated safely
pub fn calculate_correlation_safe(x: &[f32], y: &[f32]) -> f32 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }

    // Check for all zeros or constant values
    let x_sum: f32 = x.iter().sum();
    let y_sum: f32 = y.iter().sum();

    if x_sum == 0.0 || y_sum == 0.0 {
        return 0.0;
    }

    // Check for constant values (zero variance)
    let x_mean = x_sum / x.len() as f32;
    let y_mean = y_sum / y.len() as f32;

    let mut x_variance = 0.0;
    let mut y_variance = 0.0;
    let mut covariance = 0.0;

    for i in 0..x.len() {
        let x_diff = x[i] - x_mean;
        let y_diff = y[i] - y_mean;

        x_variance += x_diff * x_diff;
        y_variance += y_diff * y_diff;
        covariance += x_diff * y_diff;
    }

    // Check for zero variance (constant values)
    if x_variance == 0.0 || y_variance == 0.0 {
        return 0.0;
    }

    // Calculate correlation coefficient
    let correlation = covariance / (f32::sqrt(x_variance) * f32::sqrt(y_variance));

    // Check for NaN or infinite values
    if correlation.is_nan() || correlation.is_infinite() {
        return 0.0;
    }

    // Clamp to valid range [-1, 1]
    correlation.clamp(-1.0, 1.0)
}

/// Calculate correlation between two f32 slices
/// Returns 0.0 if correlation cannot be calculated safely
pub fn correlation(x: &[f32], y: &[f32]) -> f32 {
    calculate_correlation_safe(x, y)
}

/// Calculate hyperscore similar to X! Tandem and MSFragger
///
/// hyperscore = log(Nb! * Ny! * sum(Ib,i) * sum(Iy,i))
/// where:
/// - Nb = number of matched b-ions
/// - Ny = number of matched y-ions
/// - Ib,i = intensities of matched b-ions
/// - Iy,i = intensities of matched y-ions
///
/// Fragment types are encoded as ASCII values:
/// - b-ion = 98 (ASCII 'b')
/// - y-ion = 121 (ASCII 'y')
///   (other fragment types exist but are not used in hyperscore)
pub fn calculate_hyperscore(
    fragment_types: &[u8],
    fragment_intensities: &[f32],
    matched_mask: &[bool],
) -> f32 {
    if fragment_types.len() != fragment_intensities.len()
        || fragment_types.len() != matched_mask.len()
    {
        return 0.0;
    }

    let mut n_b = 0u32;
    let mut n_y = 0u32;
    let mut sum_b_intensity = 0.0f32;
    let mut sum_y_intensity = 0.0f32;

    for i in 0..fragment_types.len() {
        if !matched_mask[i] || fragment_intensities[i] == 0.0 {
            continue;
        }

        match fragment_types[i] {
            FragmentType::B => {
                // b-ion
                n_b += 1;
                sum_b_intensity += fragment_intensities[i];
            }
            FragmentType::Y => {
                // y-ion
                n_y += 1;
                sum_y_intensity += fragment_intensities[i];
            }
            _ => {
                // Other fragment types not used in hyperscore
            }
        }
    }

    if n_b == 0 && n_y == 0 {
        return 0.0;
    }

    // Calculate factorial using gamma function: n! = Γ(n+1)
    let factorial_b = if n_b > 0 {
        gamma_ln(n_b as f32 + 1.0)
    } else {
        0.0
    };
    let factorial_y = if n_y > 0 {
        gamma_ln(n_y as f32 + 1.0)
    } else {
        0.0
    };

    // Calculate hyperscore: log(Nb! * Ny! * sum(Ib) * sum(Iy))
    let hyperscore =
        factorial_b + factorial_y + sum_b_intensity.ln().max(0.0) + sum_y_intensity.ln().max(0.0);

    if hyperscore.is_finite() && hyperscore > 0.0 {
        hyperscore
    } else {
        0.0
    }
}

/// Natural logarithm of gamma function using Stirling's approximation
/// For factorial calculation: ln(n!) = ln(Γ(n+1))
/// Always uses approximation as requested, except for special cases
fn gamma_ln(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }

    if (x - 1.0).abs() < 1e-6 {
        return 0.0; // ln(Γ(1)) = ln(0!) = ln(1) = 0
    }

    if (x - 2.0).abs() < 1e-6 {
        return 0.0; // ln(Γ(2)) = ln(1!) = ln(1) = 0
    }

    // Stirling's approximation: ln(Γ(x)) ≈ (x-0.5)*ln(x) - x + 0.5*ln(2π)
    let ln_2pi = 1.837_877_f32;
    (x - 0.5) * x.ln() - x + 0.5 * ln_2pi
}

/// Calculate longest continuous b and y ion series scores
/// Returns (longest_b_series, longest_y_series) based on fragment_number values
/// Handles fragment numbers in any order by sorting internally
pub fn calculate_longest_ion_series(
    fragment_types: &[u8],
    fragment_numbers: &[u8],
    matched_mask: &[bool],
) -> (u8, u8) {
    if fragment_types.len() != matched_mask.len() || fragment_types.len() != fragment_numbers.len()
    {
        return (0, 0);
    }

    // Collect matched b and y ions with their fragment numbers
    let mut b_ions: Vec<u8> = Vec::new();
    let mut y_ions: Vec<u8> = Vec::new();

    for i in 0..fragment_types.len() {
        if matched_mask[i] {
            match fragment_types[i] {
                FragmentType::B => b_ions.push(fragment_numbers[i]),
                FragmentType::Y => y_ions.push(fragment_numbers[i]),
                _ => {}
            }
        }
    }

    // Helper function to find longest continuous sequence
    let find_longest_sequence = |mut ions: Vec<u8>| -> u8 {
        if ions.is_empty() {
            return 0;
        }

        ions.sort_unstable();

        let mut max_length = 1u8;
        let mut current_length = 1u8;

        for i in 1..ions.len() {
            if ions[i] == ions[i - 1] + 1 {
                current_length += 1;
                max_length = max_length.max(current_length);
            } else {
                current_length = 1;
            }
        }

        max_length
    };

    let longest_b = find_longest_sequence(b_ions);
    let longest_y = find_longest_sequence(y_ions);

    (longest_b, longest_y)
}
