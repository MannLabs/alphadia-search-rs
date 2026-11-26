//! AVX-512F port of the NEON convolution kernel.

#[cfg(target_arch = "x86_64")]
use crate::kernel::GaussianKernel;
#[cfg(target_arch = "x86_64")]
use numpy::ndarray::Array2;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "x86_64")]
use std::cmp::min;

#[cfg(target_arch = "x86_64")]
const LANES: usize = 16;

/// # Safety
/// The caller must ensure the CPU supports the `avx512f` target feature.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn convolution_avx512(kernel: &GaussianKernel, xic: &Array2<f32>) -> Array2<f32> {
    let (n_fragments, n_points) = xic.dim();
    let kernel_size = kernel.kernel_array.len();
    let half_kernel = kernel_size / 2;

    let mut convolved: Array2<f32> = Array2::zeros((n_fragments, n_points));

    if n_fragments == 0 || n_points == 0 || n_points < kernel_size {
        super::scalar::fill_edges(kernel, xic, &mut convolved);
        return convolved;
    }

    let start_idx = half_kernel;
    let end_idx = n_points.saturating_sub(half_kernel);

    for f_idx in 0..n_fragments {
        let xic_row = xic.row(f_idx);
        let mut conv_row = convolved.row_mut(f_idx);

        if start_idx < end_idx {
            let simd_end_idx = start_idx + ((end_idx - start_idx) / LANES) * LANES;

            let mut i = start_idx;
            while i < min(simd_end_idx, end_idx) {
                if i + LANES <= n_points {
                    let center_kernel_vec = _mm512_set1_ps(kernel.kernel_array[half_kernel]);
                    let center_input_vec = _mm512_loadu_ps(xic_row.as_ptr().add(i));
                    let mut sum_vec = _mm512_mul_ps(center_input_vec, center_kernel_vec);

                    for k in 0..half_kernel {
                        let kernel_val_vec = _mm512_set1_ps(kernel.kernel_array[k]);
                        let left_offset = i - (half_kernel - k);
                        let right_offset = i + (half_kernel - k);
                        let left_vec = _mm512_loadu_ps(xic_row.as_ptr().add(left_offset));
                        let right_vec = _mm512_loadu_ps(xic_row.as_ptr().add(right_offset));
                        let sum_inputs = _mm512_add_ps(left_vec, right_vec);
                        sum_vec = _mm512_fmadd_ps(sum_inputs, kernel_val_vec, sum_vec);
                    }

                    _mm512_storeu_ps(conv_row.as_mut_ptr().add(i), sum_vec);
                    i += LANES;
                } else {
                    break;
                }
            }

            for i in i..end_idx {
                let mut sum = 0.0;
                sum += xic_row[i] * kernel.kernel_array[half_kernel];
                for k in 0..half_kernel {
                    let kernel_val = kernel.kernel_array[k];
                    let left_val = xic_row[i - (half_kernel - k)];
                    let right_val = xic_row[i + (half_kernel - k)];
                    sum += (left_val + right_val) * kernel_val;
                }
                conv_row[i] = sum;
            }
        }
    }

    super::scalar::fill_edges(kernel, xic, &mut convolved);
    convolved
}
