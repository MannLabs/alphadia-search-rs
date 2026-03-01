#[cfg(target_arch = "aarch64")]
use crate::kernel::GaussianKernel;
#[cfg(target_arch = "aarch64")]
use numpy::ndarray::Array2;
#[cfg(target_arch = "aarch64")]
use std::cmp::min;

/// NEON-optimized convolution implementation for aarch64
#[cfg(target_arch = "aarch64")]
pub fn convolution_neon(kernel: &GaussianKernel, xic: &Array2<f32>) -> Array2<f32> {
    use std::arch::aarch64::{vaddq_f32, vdupq_n_f32, vld1q_f32, vmulq_f32, vst1q_f32};

    let (n_fragments, n_points) = xic.dim();
    let kernel_size = kernel.kernel_array.len();
    let half_kernel = kernel_size / 2;

    // Create output array with same dimensions, initialized to zeros
    let mut convolved: Array2<f32> = Array2::zeros((n_fragments, n_points));

    // Early return for empty inputs or when kernel is too large - we'll keep this branch
    // as it's executed only once at the beginning of the function
    if n_fragments == 0 || n_points == 0 || n_points < kernel_size {
        return convolved;
    }

    // Pre-compute valid region boundaries - this avoids branching in the inner loops
    let start_idx = half_kernel;
    let end_idx = n_points.saturating_sub(half_kernel);

    // Process each fragment
    for f_idx in 0..n_fragments {
        let xic_row = xic.row(f_idx);
        let mut conv_row = convolved.row_mut(f_idx);

        // Don't process if there's no valid region (start_idx >= end_idx)
        if start_idx < end_idx {
            // SIMD optimized main loop with symmetric kernel
            const SIMD_WIDTH: usize = 4; // Process 4 points at a time

            // Calculate SIMD-aligned end point
            let simd_end_idx = start_idx + ((end_idx - start_idx) / SIMD_WIDTH) * SIMD_WIDTH;

            // SIMD-optimized processing for aligned blocks
            let mut i = start_idx;
            while i < min(simd_end_idx, end_idx) {
                // Ensure we don't go out of bounds with SIMD loads
                if i + SIMD_WIDTH <= n_points {
                    // Accumulators for 4 points
                    let mut sum_vec = unsafe { vdupq_n_f32(0.0) };

                    // Process center element of kernel once
                    let center_kernel_val = kernel.kernel_array[half_kernel];
                    let center_kernel_vec = unsafe { vdupq_n_f32(center_kernel_val) };
                    let center_input_vec = unsafe { vld1q_f32(xic_row.as_ptr().add(i)) };
                    sum_vec = unsafe {
                        vaddq_f32(sum_vec, vmulq_f32(center_input_vec, center_kernel_vec))
                    };

                    // Process symmetric pairs
                    for k in 0..half_kernel {
                        let kernel_val = kernel.kernel_array[k];
                        let kernel_val_vec = unsafe { vdupq_n_f32(kernel_val) };

                        let left_offset = i - (half_kernel - k);
                        let right_offset = i + (half_kernel - k);

                        // We're guaranteed both indices are in bounds because:
                        // - i >= half_kernel (due to start_idx = half_kernel)
                        // - i + half_kernel < n_points (due to end_idx = n_points - half_kernel)
                        // - left_offset is between i-half_kernel and i
                        // - right_offset is between i and i+half_kernel
                        let left_vec = unsafe { vld1q_f32(xic_row.as_ptr().add(left_offset)) };
                        let right_vec = unsafe { vld1q_f32(xic_row.as_ptr().add(right_offset)) };

                        // Add symmetric inputs, then multiply
                        let sum_inputs = unsafe { vaddq_f32(left_vec, right_vec) };
                        sum_vec =
                            unsafe { vaddq_f32(sum_vec, vmulq_f32(sum_inputs, kernel_val_vec)) };
                    }

                    // Store the results
                    unsafe { vst1q_f32(conv_row.as_mut_ptr().add(i), sum_vec) };
                    i += SIMD_WIDTH;
                } else {
                    // Not enough space for full SIMD, switch to scalar processing
                    break;
                }
            }

            // Handle remaining points in the valid region using scalar code
            for i in i..end_idx {
                let mut sum = 0.0;

                // Center element
                sum += xic_row[i] * kernel.kernel_array[half_kernel];

                // Process symmetric pairs
                for k in 0..half_kernel {
                    let kernel_val = kernel.kernel_array[k];

                    // Get pair of symmetric inputs - no bounds check needed in valid region
                    let left_val = xic_row[i - (half_kernel - k)];
                    let right_val = xic_row[i + (half_kernel - k)];

                    // Add symmetric inputs, then multiply by kernel value once
                    sum += (left_val + right_val) * kernel_val;
                }

                conv_row[i] = sum;
            }
        }

        // Zeros remain at the edges (no padding)
    }

    convolved
}

/// NEON-v2 convolution with FMA and 4x loop unrolling for better ILP.
/// Exploits symmetric kernel and processes 16 output points per iteration.
#[cfg(target_arch = "aarch64")]
pub fn convolution_neon_v2(kernel: &GaussianKernel, xic: &Array2<f32>) -> Array2<f32> {
    use std::arch::aarch64::{vaddq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32, vst1q_f32};

    let (n_fragments, n_points) = xic.dim();
    let kernel_size = kernel.kernel_array.len();
    let half_kernel = kernel_size / 2;

    let mut convolved: Array2<f32> = Array2::zeros((n_fragments, n_points));

    if n_fragments == 0 || n_points == 0 || n_points < kernel_size {
        return convolved;
    }

    let start_idx = half_kernel;
    let end_idx = n_points.saturating_sub(half_kernel);

    if start_idx >= end_idx {
        return convolved;
    }

    const SIMD_WIDTH: usize = 4;
    const UNROLL: usize = 4;
    const BLOCK: usize = SIMD_WIDTH * UNROLL; // 16 points per iteration

    let valid_len = end_idx - start_idx;
    let block_end = start_idx + (valid_len / BLOCK) * BLOCK;
    let simd_end = start_idx + (valid_len / SIMD_WIDTH) * SIMD_WIDTH;

    for f_idx in 0..n_fragments {
        let row_ptr = xic.as_ptr();
        let row_base = f_idx * n_points;
        let out_ptr = convolved.as_mut_ptr();

        // 4x unrolled SIMD path
        unsafe {
            let center_kval = vdupq_n_f32(kernel.kernel_array[half_kernel]);

            let mut i = start_idx;
            while i < block_end {
                let mut acc0 = vdupq_n_f32(0.0);
                let mut acc1 = vdupq_n_f32(0.0);
                let mut acc2 = vdupq_n_f32(0.0);
                let mut acc3 = vdupq_n_f32(0.0);

                // Center element
                let c0 = vld1q_f32(row_ptr.add(row_base + i));
                let c1 = vld1q_f32(row_ptr.add(row_base + i + SIMD_WIDTH));
                let c2 = vld1q_f32(row_ptr.add(row_base + i + SIMD_WIDTH * 2));
                let c3 = vld1q_f32(row_ptr.add(row_base + i + SIMD_WIDTH * 3));
                acc0 = vfmaq_f32(acc0, c0, center_kval);
                acc1 = vfmaq_f32(acc1, c1, center_kval);
                acc2 = vfmaq_f32(acc2, c2, center_kval);
                acc3 = vfmaq_f32(acc3, c3, center_kval);

                // Symmetric pairs
                for k in 0..half_kernel {
                    let kval = vdupq_n_f32(kernel.kernel_array[k]);
                    let offset = half_kernel - k;

                    let l0 = vld1q_f32(row_ptr.add(row_base + i - offset));
                    let r0 = vld1q_f32(row_ptr.add(row_base + i + offset));
                    let s0 = vaddq_f32(l0, r0);
                    acc0 = vfmaq_f32(acc0, s0, kval);

                    let l1 = vld1q_f32(row_ptr.add(row_base + i + SIMD_WIDTH - offset));
                    let r1 = vld1q_f32(row_ptr.add(row_base + i + SIMD_WIDTH + offset));
                    let s1 = vaddq_f32(l1, r1);
                    acc1 = vfmaq_f32(acc1, s1, kval);

                    let l2 = vld1q_f32(row_ptr.add(row_base + i + SIMD_WIDTH * 2 - offset));
                    let r2 = vld1q_f32(row_ptr.add(row_base + i + SIMD_WIDTH * 2 + offset));
                    let s2 = vaddq_f32(l2, r2);
                    acc2 = vfmaq_f32(acc2, s2, kval);

                    let l3 = vld1q_f32(row_ptr.add(row_base + i + SIMD_WIDTH * 3 - offset));
                    let r3 = vld1q_f32(row_ptr.add(row_base + i + SIMD_WIDTH * 3 + offset));
                    let s3 = vaddq_f32(l3, r3);
                    acc3 = vfmaq_f32(acc3, s3, kval);
                }

                vst1q_f32(out_ptr.add(row_base + i), acc0);
                vst1q_f32(out_ptr.add(row_base + i + SIMD_WIDTH), acc1);
                vst1q_f32(out_ptr.add(row_base + i + SIMD_WIDTH * 2), acc2);
                vst1q_f32(out_ptr.add(row_base + i + SIMD_WIDTH * 3), acc3);

                i += BLOCK;
            }

            // Remaining SIMD-width blocks
            while i < simd_end {
                let mut acc = vdupq_n_f32(0.0);

                let c = vld1q_f32(row_ptr.add(row_base + i));
                acc = vfmaq_f32(acc, c, center_kval);

                for k in 0..half_kernel {
                    let kval = vdupq_n_f32(kernel.kernel_array[k]);
                    let offset = half_kernel - k;

                    let l = vld1q_f32(row_ptr.add(row_base + i - offset));
                    let r = vld1q_f32(row_ptr.add(row_base + i + offset));
                    let s = vaddq_f32(l, r);
                    acc = vfmaq_f32(acc, s, kval);
                }

                vst1q_f32(out_ptr.add(row_base + i), acc);
                i += SIMD_WIDTH;
            }

            // Scalar tail
            let xic_row = xic.row(f_idx);
            let mut conv_row = convolved.row_mut(f_idx);
            for i in simd_end..end_idx {
                let mut sum = xic_row[i] * kernel.kernel_array[half_kernel];
                for k in 0..half_kernel {
                    let left_val = xic_row[i - (half_kernel - k)];
                    let right_val = xic_row[i + (half_kernel - k)];
                    sum += (left_val + right_val) * kernel.kernel_array[k];
                }
                conv_row[i] = sum;
            }
        }
    }

    convolved
}
