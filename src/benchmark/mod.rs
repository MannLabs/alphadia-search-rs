use crate::convolution::convolution;
use crate::kernel::GaussianKernel;
use numpy::ndarray::Array2;
use rand::prelude::*;
use std::cmp::min;
use std::time::Instant;
// Define a struct to hold benchmark results
pub struct BenchmarkResult {
    pub name: String,
    pub time_seconds: f64,
    pub speedup: f64,
}

// Define a type for convolution functions
type ConvolutionFunction = fn(&GaussianKernel, &Array2<f32>) -> Array2<f32>;

// Helper function to apply the padded convolution (similar to the one in PeakGroupScoring)
pub fn benchmark_padded_convolution(kernel: &GaussianKernel, xic: &Array2<f32>) -> Array2<f32> {
    let (n_fragments, n_points) = xic.dim();
    let kernel_size = kernel.kernel_array.len();
    let half_kernel = kernel_size / 2;

    // Create output array with same dimensions
    let mut convolved = Array2::zeros((n_fragments, n_points));

    // Process each fragment
    for f_idx in 0..n_fragments {
        let xic_row = xic.row(f_idx);
        let mut conv_row = convolved.row_mut(f_idx);

        // Apply convolution with padding for each point
        for i in 0..n_points {
            let mut sum = 0.0;

            for k in 0..kernel_size {
                let idx = i as isize + (k as isize - half_kernel as isize);
                let value = if idx < 0 {
                    // Left padding: mirror or use first value
                    xic_row[0]
                } else if idx >= n_points as isize {
                    // Right padding: mirror or use last value
                    xic_row[n_points - 1]
                } else {
                    xic_row[idx as usize]
                };

                sum += value * kernel.kernel_array[k];
            }

            conv_row[i] = sum;
        }
    }

    convolved
}

// Optimized convolution that reduces branching and minimizes allocations
pub fn benchmark_padded_convolution_branching(
    kernel: &GaussianKernel,
    xic: &Array2<f32>,
) -> Array2<f32> {
    let (n_fragments, n_points) = xic.dim();
    let kernel_size = kernel.kernel_array.len();
    let half_kernel = kernel_size / 2;

    // Create output array with same dimensions
    let mut convolved = Array2::zeros((n_fragments, n_points));

    // Process each fragment
    for f_idx in 0..n_fragments {
        let xic_row = xic.row(f_idx);
        let mut conv_row = convolved.row_mut(f_idx);

        // Handle edge cases separately to reduce branching in main loop

        // Handle left edge (where kernel overlaps the start of the array)
        for i in 0..min(half_kernel, n_points) {
            let mut sum = 0.0;
            let first_val = xic_row[0]; // Cache the first value

            for k in 0..kernel_size {
                let idx = i as isize + (k as isize - half_kernel as isize);
                let value = if idx < 0 {
                    first_val
                } else {
                    xic_row[idx as usize]
                };

                sum += value * kernel.kernel_array[k];
            }

            conv_row[i] = sum;
        }

        // Main loop - no boundary checks needed for most of the array
        // This section processes points where the full kernel fits within the array
        for i in half_kernel..n_points.saturating_sub(half_kernel) {
            let mut sum = 0.0;

            // Use direct indexing without branches for better performance
            for k in 0..kernel_size {
                let idx = (i as isize + (k as isize - half_kernel as isize)) as usize;
                sum += xic_row[idx] * kernel.kernel_array[k];
            }

            conv_row[i] = sum;
        }

        // Handle right edge (where kernel overlaps the end of the array)
        let last_val = xic_row[n_points - 1]; // Cache the last value
        for i in n_points.saturating_sub(half_kernel)..n_points {
            let mut sum = 0.0;

            for k in 0..kernel_size {
                let idx = i as isize + (k as isize - half_kernel as isize);
                let value = if idx >= n_points as isize {
                    last_val
                } else {
                    xic_row[idx as usize]
                };

                sum += value * kernel.kernel_array[k];
            }

            conv_row[i] = sum;
        }
    }

    convolved
}

/// Optimized convolution that combines both branch optimization and SIMD for the non-branching section
pub fn benchmark_padded_convolution_branching_simd(
    kernel: &GaussianKernel,
    xic: &Array2<f32>,
) -> Array2<f32> {
    #[cfg(target_arch = "aarch64")]
    use std::arch::aarch64::{vaddq_f32, vdupq_n_f32, vld1q_f32, vmulq_f32, vst1q_f32};

    let (n_fragments, n_points) = xic.dim();
    let kernel_size = kernel.kernel_array.len();
    let half_kernel = kernel_size / 2;

    // Create output array with same dimensions
    let mut convolved = Array2::zeros((n_fragments, n_points));

    // Process each fragment
    for f_idx in 0..n_fragments {
        let xic_row = xic.row(f_idx);
        let mut conv_row = convolved.row_mut(f_idx);

        // Handle left edge (where kernel overlaps the start of the array)
        for i in 0..min(half_kernel, n_points) {
            let mut sum = 0.0;
            let first_val = xic_row[0]; // Cache the first value

            for k in 0..kernel_size {
                let idx = i as isize + (k as isize - half_kernel as isize);
                let value = if idx < 0 {
                    first_val
                } else {
                    xic_row[idx as usize]
                };

                sum += value * kernel.kernel_array[k];
            }

            conv_row[i] = sum;
        }

        // Main loop - SIMD optimized without branching
        #[cfg(target_arch = "aarch64")]
        {
            const SIMD_WIDTH: usize = 4; // Process 4 points at a time

            // Determine range for the SIMD processing (where no bounds checking is needed)
            let start_idx = half_kernel;
            let end_idx = n_points.saturating_sub(half_kernel);
            let simd_end_idx = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;

            // Process points in SIMD chunks in the safe middle region
            for i in (start_idx..simd_end_idx).step_by(SIMD_WIDTH) {
                // Initialize vector accumulators for 4 points
                let mut sum_vec = unsafe { vdupq_n_f32(0.0) };

                // Apply convolution for each kernel position
                for k in 0..kernel_size {
                    let k_offset = k as isize - half_kernel as isize;
                    let kernel_val = kernel.kernel_array[k];
                    let kernel_vec = unsafe { vdupq_n_f32(kernel_val) };

                    // Load 4 consecutive points from xic with the kernel offset
                    // No bounds checking needed in this region
                    let base_idx = (i as isize + k_offset) as usize;
                    let input_vec = unsafe { vld1q_f32(xic_row.as_ptr().add(base_idx)) };

                    // Multiply and accumulate
                    sum_vec = unsafe { vaddq_f32(sum_vec, vmulq_f32(input_vec, kernel_vec)) };
                }

                // Store the results
                unsafe { vst1q_f32(conv_row.as_mut_ptr().add(i), sum_vec) };
            }

            // Handle remaining points in the middle section
            for i in simd_end_idx..end_idx {
                let mut sum = 0.0;

                // Use direct indexing without branches for better performance
                for k in 0..kernel_size {
                    let idx = (i as isize + (k as isize - half_kernel as isize)) as usize;
                    sum += xic_row[idx] * kernel.kernel_array[k];
                }

                conv_row[i] = sum;
            }
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            // Fallback for non-aarch64: use the branch-optimized version
            let start_idx = half_kernel;
            let end_idx = n_points.saturating_sub(half_kernel);

            for i in start_idx..end_idx {
                let mut sum = 0.0;

                // Use direct indexing without branches for better performance
                for k in 0..kernel_size {
                    let idx = (i as isize + (k as isize - half_kernel as isize)) as usize;
                    sum += xic_row[idx] * kernel.kernel_array[k];
                }

                conv_row[i] = sum;
            }
        }

        // Handle right edge (where kernel overlaps the end of the array)
        let last_val = xic_row[n_points - 1]; // Cache the last value
        for i in n_points.saturating_sub(half_kernel)..n_points {
            let mut sum = 0.0;

            for k in 0..kernel_size {
                let idx = i as isize + (k as isize - half_kernel as isize);
                let value = if idx >= n_points as isize {
                    last_val
                } else {
                    xic_row[idx as usize]
                };

                sum += value * kernel.kernel_array[k];
            }

            conv_row[i] = sum;
        }
    }

    convolved
}

// Implementation without padding that starts with first valid calculation
pub fn benchmark_nonpadded_convolution_simd(
    kernel: &GaussianKernel,
    xic: &Array2<f32>,
) -> Array2<f32> {
    #[cfg(target_arch = "aarch64")]
    use std::arch::aarch64::{vaddq_f32, vdupq_n_f32, vld1q_f32, vmulq_f32, vst1q_f32};

    let (n_fragments, n_points) = xic.dim();
    let kernel_size = kernel.kernel_array.len();
    let half_kernel = kernel_size / 2;

    // Create output array with same dimensions, initialized to zeros
    let mut convolved: Array2<f32> = Array2::zeros((n_fragments, n_points));

    // Process each fragment
    for f_idx in 0..n_fragments {
        let xic_row = xic.row(f_idx);
        let mut conv_row = convolved.row_mut(f_idx);

        // Skip left edge - no calculation needed as we're not padding
        // Start from the first valid index where the full kernel fits

        // SIMD optimized main loop
        #[cfg(target_arch = "aarch64")]
        {
            const SIMD_WIDTH: usize = 4; // Process 4 points at a time

            // We can start calculations from half_kernel (first valid point)
            // And end at n_points - half_kernel (last valid point)
            let start_idx = half_kernel;
            let end_idx = n_points.saturating_sub(half_kernel);
            let simd_end_idx = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;

            // Process points in SIMD chunks
            for i in (start_idx..simd_end_idx).step_by(SIMD_WIDTH) {
                // Initialize vector accumulators for 4 points
                let mut sum_vec = unsafe { vdupq_n_f32(0.0) };

                // Apply convolution for each kernel position
                for k in 0..kernel_size {
                    let k_offset = k as isize - half_kernel as isize;
                    let kernel_val = kernel.kernel_array[k];
                    let kernel_vec = unsafe { vdupq_n_f32(kernel_val) };

                    // Load 4 consecutive points from xic with the kernel offset
                    let base_idx = (i as isize + k_offset) as usize;
                    let input_vec = unsafe { vld1q_f32(xic_row.as_ptr().add(base_idx)) };

                    // Multiply and accumulate
                    sum_vec = unsafe { vaddq_f32(sum_vec, vmulq_f32(input_vec, kernel_vec)) };
                }

                // Store the results
                unsafe { vst1q_f32(conv_row.as_mut_ptr().add(i), sum_vec) };
            }

            // Handle remaining points in the valid region
            for i in simd_end_idx..end_idx {
                let mut sum = 0.0;

                for k in 0..kernel_size {
                    let idx = (i as isize + (k as isize - half_kernel as isize)) as usize;
                    sum += xic_row[idx] * kernel.kernel_array[k];
                }

                conv_row[i] = sum;
            }
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            // Fallback for non-aarch64 architectures
            // We only calculate points where the full kernel fits within the input array
            let start_idx = half_kernel;
            let end_idx = n_points.saturating_sub(half_kernel);

            for i in start_idx..end_idx {
                let mut sum = 0.0;

                for k in 0..kernel_size {
                    let idx = (i as isize + (k as isize - half_kernel as isize)) as usize;
                    sum += xic_row[idx] * kernel.kernel_array[k];
                }

                conv_row[i] = sum;
            }
        }

        // Skip right edge - no calculation needed as we're not padding
        // The output array is already initialized with zeros
    }

    convolved
}

// Implementation that leverages the symmetry of the Gaussian kernel
pub fn benchmark_symmetric_kernel_simd(kernel: &GaussianKernel, xic: &Array2<f32>) -> Array2<f32> {
    #[cfg(target_arch = "aarch64")]
    use std::arch::aarch64::{vaddq_f32, vdupq_n_f32, vld1q_f32, vmulq_f32, vst1q_f32};

    let (n_fragments, n_points) = xic.dim();
    let kernel_size = kernel.kernel_array.len();
    let half_kernel = kernel_size / 2;

    // Create output array with same dimensions
    let mut convolved: Array2<f32> = Array2::zeros((n_fragments, n_points));

    // Process each fragment
    for f_idx in 0..n_fragments {
        let xic_row = xic.row(f_idx);
        let mut conv_row = convolved.row_mut(f_idx);

        // Handle left edge (where kernel overlaps the start of the array)
        for i in 0..min(half_kernel, n_points) {
            let mut sum = 0.0;
            let first_val = xic_row[0]; // Cache the first value

            // Center element
            let center_idx = i as isize;
            let center_val = if center_idx >= 0 && center_idx < n_points as isize {
                xic_row[center_idx as usize]
            } else {
                first_val
            };
            sum += center_val * kernel.kernel_array[half_kernel];

            // Process symmetric pairs
            for k in 0..half_kernel {
                let kernel_val = kernel.kernel_array[k];

                // Left of center
                let left_idx = i as isize - (half_kernel - k) as isize;
                let left_val = if left_idx < 0 {
                    first_val
                } else {
                    xic_row[left_idx as usize]
                };

                // Right of center
                let right_idx = i as isize + (half_kernel - k) as isize;
                let right_val = if right_idx >= n_points as isize {
                    xic_row[n_points - 1]
                } else {
                    xic_row[right_idx as usize]
                };

                // Add symmetric inputs, then multiply by kernel value once
                sum += (left_val + right_val) * kernel_val;
            }

            conv_row[i] = sum;
        }

        // Main loop - SIMD optimized with symmetric kernel
        #[cfg(target_arch = "aarch64")]
        {
            const SIMD_WIDTH: usize = 4; // Process 4 points at a time

            // We can start calculations from half_kernel (first valid point)
            // And end at n_points - half_kernel (last valid point)
            let start_idx = half_kernel;
            let end_idx = n_points.saturating_sub(half_kernel);
            let simd_end_idx = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;

            // Process points in SIMD chunks in the safe middle region
            for i in (start_idx..simd_end_idx).step_by(SIMD_WIDTH) {
                // Accumulators for 4 points
                let mut sum_vec = unsafe { vdupq_n_f32(0.0) };

                // Process center element of kernel
                let center_kernel_val = kernel.kernel_array[half_kernel];
                let center_kernel_vec = unsafe { vdupq_n_f32(center_kernel_val) };
                let center_input_vec = unsafe { vld1q_f32(xic_row.as_ptr().add(i)) };
                sum_vec =
                    unsafe { vaddq_f32(sum_vec, vmulq_f32(center_input_vec, center_kernel_vec)) };

                // Process symmetric pairs
                for k in 0..half_kernel {
                    let kernel_val = kernel.kernel_array[k];
                    let kernel_val_vec = unsafe { vdupq_n_f32(kernel_val) };

                    let left_offset = i - (half_kernel - k);
                    let right_offset = i + (half_kernel - k);

                    let left_vec = unsafe { vld1q_f32(xic_row.as_ptr().add(left_offset)) };
                    let right_vec = unsafe { vld1q_f32(xic_row.as_ptr().add(right_offset)) };

                    // Add symmetric inputs, then multiply
                    let sum_inputs = unsafe { vaddq_f32(left_vec, right_vec) };
                    sum_vec = unsafe { vaddq_f32(sum_vec, vmulq_f32(sum_inputs, kernel_val_vec)) };
                }

                // Store the results
                unsafe { vst1q_f32(conv_row.as_mut_ptr().add(i), sum_vec) };
            }

            // Handle remaining points in the middle section
            for i in simd_end_idx..end_idx {
                let mut sum = 0.0;

                // Center element
                sum += xic_row[i] * kernel.kernel_array[half_kernel];

                // Process symmetric pairs
                for k in 0..half_kernel {
                    let kernel_val = kernel.kernel_array[k];

                    // Get pair of symmetric inputs
                    let left_val = xic_row[i - (half_kernel - k)];
                    let right_val = xic_row[i + (half_kernel - k)];

                    // Add symmetric inputs, then multiply by kernel value once
                    sum += (left_val + right_val) * kernel_val;
                }

                conv_row[i] = sum;
            }
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            // Fallback for non-aarch64: use the symmetric optimization
            let start_idx = half_kernel;
            let end_idx = n_points.saturating_sub(half_kernel);

            for i in start_idx..end_idx {
                let mut sum = 0.0;

                // Center element
                sum += xic_row[i] * kernel.kernel_array[half_kernel];

                // Process symmetric pairs
                for k in 0..half_kernel {
                    let kernel_val = kernel.kernel_array[k];

                    // Get pair of symmetric inputs
                    let left_val = xic_row[i - (half_kernel - k)];
                    let right_val = xic_row[i + (half_kernel - k)];

                    // Add symmetric inputs, then multiply by kernel value once
                    sum += (left_val + right_val) * kernel_val;
                }

                conv_row[i] = sum;
            }
        }

        // Handle right edge (where kernel overlaps the end of the array)
        let last_val = xic_row[n_points - 1]; // Cache the last value
        for i in n_points.saturating_sub(half_kernel)..n_points {
            let mut sum = 0.0;

            // Center element
            let center_idx = i as isize;
            let center_val = if center_idx >= 0 && center_idx < n_points as isize {
                xic_row[center_idx as usize]
            } else {
                last_val
            };
            sum += center_val * kernel.kernel_array[half_kernel];

            // Process symmetric pairs
            for k in 0..half_kernel {
                let kernel_val = kernel.kernel_array[k];

                // Left of center
                let left_idx = i as isize - (half_kernel - k) as isize;
                let left_val = if left_idx < 0 {
                    xic_row[0]
                } else {
                    xic_row[left_idx as usize]
                };

                // Right of center
                let right_idx = i as isize + (half_kernel - k) as isize;
                let right_val = if right_idx >= n_points as isize {
                    last_val
                } else {
                    xic_row[right_idx as usize]
                };

                // Add symmetric inputs, then multiply by kernel value once
                sum += (left_val + right_val) * kernel_val;
            }

            conv_row[i] = sum;
        }
    }

    convolved
}

// Function to generate random test data
fn generate_test_data(num_arrays: usize, n_fragments: usize, n_points: usize) -> Vec<Array2<f32>> {
    let mut rng = rand::rng();
    let mut arrays = Vec::with_capacity(num_arrays);

    for _ in 0..num_arrays {
        let mut arr = Array2::<f32>::zeros((n_fragments, n_points));
        for i in 0..n_fragments {
            for j in 0..n_points {
                arr[[i, j]] = rng.random_range(0.0..1.0);
            }
        }
        arrays.push(arr);
    }

    arrays
}

// Main benchmarking function
pub fn run_convolution_benchmark() -> Vec<BenchmarkResult> {
    // Configuration
    let num_arrays = 1000;
    let n_points = 1000;
    let n_fragments = 12;

    let kernel_width = 20;

    println!("Running convolution benchmark in Rust...");
    println!("Testing with {num_arrays} arrays of shape {n_fragments}x{n_points}");

    // Create a kernel with shape 20 and fwhm_rt = 2
    let kernel = GaussianKernel::new(
        2.0,          // fwhm_rt
        1.0,          // sigma_scale_rt
        kernel_width, // kernel_width
        1.0,          // rt_resolution
    );

    println!("Kernel width: {kernel_width}");

    // Generate test data
    println!("Generating random test data...");
    let arrays = generate_test_data(num_arrays, n_fragments, n_points);

    // Define the implementations to test
    let implementations: Vec<(String, ConvolutionFunction)> = vec![
        ("Original".to_string(), benchmark_padded_convolution),
        (
            "Branching".to_string(),
            benchmark_padded_convolution_branching,
        ),
        (
            "Branching+SIMD".to_string(),
            benchmark_padded_convolution_branching_simd,
        ),
        (
            "Nonpadded+SIMD".to_string(),
            benchmark_nonpadded_convolution_simd,
        ),
        (
            "Symmetric+SIMD".to_string(),
            benchmark_symmetric_kernel_simd,
        ),
        ("Nonpadded+Symmetric".to_string(), convolution),
    ];

    // Run benchmarks
    let mut results = Vec::new();
    let mut baseline_time = 0.0;

    for (i, (name, implementation)) in implementations.iter().enumerate() {
        println!("Benchmarking {name} implementation...");
        let start_time = Instant::now();

        // Apply convolution to each array
        for arr in &arrays {
            let _result = implementation(&kernel, arr);
        }

        let end_time = Instant::now();
        let duration = end_time.duration_since(start_time);
        let time_seconds = duration.as_secs_f64();

        // Save baseline time from first implementation
        if i == 0 {
            baseline_time = time_seconds;
        }

        // Calculate speedup relative to baseline
        let speedup = baseline_time / time_seconds;

        // Store results
        results.push(BenchmarkResult {
            name: name.clone(),
            time_seconds,
            speedup,
        });
    }

    // Print the results
    println!("\nBenchmark Results:");
    println!("------------------");
    for result in &results {
        println!(
            "{} implementation: {:.4} seconds (speedup: {:.2}x)",
            result.name, result.time_seconds, result.speedup
        );
    }
    println!("------------------");

    results
}

// Function to test numerical similarity between different implementations
#[allow(dead_code)]
pub fn test_convolution_implementations() -> bool {
    // Use a smaller dataset for testing
    let n_points = 100;
    let n_fragments = 6;
    let num_arrays = 10;

    let kernel_width = 20;

    println!("Testing numerical similarity of convolution implementations...");

    // Create a kernel
    let kernel = GaussianKernel::new(
        2.0,          // fwhm_rt
        1.0,          // sigma_scale_rt
        kernel_width, // kernel_width
        1.0,          // rt_resolution
    );

    println!("Kernel width: {kernel_width}");

    // Generate test data
    let arrays = generate_test_data(num_arrays, n_fragments, n_points);

    // Define implementations to test
    let implementations: Vec<(String, ConvolutionFunction)> = vec![
        ("Original".to_string(), benchmark_padded_convolution),
        (
            "Branching".to_string(),
            benchmark_padded_convolution_branching,
        ),
        (
            "Branching+SIMD".to_string(),
            benchmark_padded_convolution_branching_simd,
        ),
        (
            "Symmetric+SIMD".to_string(),
            benchmark_symmetric_kernel_simd,
        ),
    ];

    // Define tolerance for numerical similarity
    let tolerance = 1e-5;
    let mut all_tests_passed = true;

    for arr in &arrays {
        // Get reference result from original implementation
        let reference_result = benchmark_padded_convolution(&kernel, arr);

        // Test each optimized implementation
        for (_i, (name, implementation)) in implementations.iter().enumerate().skip(1) {
            let optimized_result = implementation(&kernel, arr);

            // Compare results
            let mut max_diff = 0.0f32;
            let (n_fragments, n_points) = reference_result.dim();

            for f in 0..n_fragments {
                for p in 0..n_points {
                    let diff = (reference_result[[f, p]] - optimized_result[[f, p]]).abs();
                    max_diff = max_diff.max(diff);
                }
            }

            // Check if difference is within tolerance
            let passed = max_diff <= tolerance;
            if !passed {
                println!("❌ {name} implementation differs from Original by {max_diff}, exceeding tolerance {tolerance}");
                all_tests_passed = false;
            } else {
                println!("✅ {name} implementation matches Original within tolerance (max diff: {max_diff})");
            }
        }
    }

    if all_tests_passed {
        println!("All implementations produced numerically similar results within tolerance!");
    } else {
        println!("Some implementations failed the numerical similarity test!");
    }

    all_tests_passed
}

#[cfg(test)]
mod tests;
