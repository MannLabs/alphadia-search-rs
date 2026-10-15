use crate::kernel::GaussianKernel;
use numpy::ndarray::Array2;

// Module declarations
pub mod neon;
pub mod scalar;

/// Main convolution function that uses the SIMD backend for optimization
/// Returns a convolved array with same dimensions as input. Within half a kernel of either end the kernel is
/// truncated to the points that exist and rescaled to the full kernel's weight.
pub fn convolution(kernel: &GaussianKernel, xic: &Array2<f32>) -> Array2<f32> {
    let mut convolved = crate::simd::get_backend().convolution(kernel, xic);
    fill_edges(kernel, xic, &mut convolved);
    convolved
}

/// Fills the points the SIMD backends leave at zero.
///
/// Zeros there meant that no peak within half a kernel of the run start or end could be selected. On a 2 min
/// gradient the RT window spans the whole run, and peptides still elute in its last 10 cycles.
fn fill_edges(kernel: &GaussianKernel, xic: &Array2<f32>, convolved: &mut Array2<f32>) {
    let (n_fragments, n_points) = xic.dim();
    let kernel_size = kernel.kernel_array.len();
    let half_kernel = kernel_size / 2;
    if n_fragments == 0 || n_points == 0 {
        return;
    }

    // The backends apply weight kernel_array[half_kernel - |d|] at offset d, for |d| <= half_kernel.
    let weight = |d: usize| kernel.kernel_array[half_kernel - d];
    let full_weight: f32 = weight(0) + 2.0 * (1..=half_kernel).map(weight).sum::<f32>();

    // The backends compute nothing when the input is shorter than the kernel.
    let interior = if n_points >= kernel_size {
        half_kernel..n_points - half_kernel
    } else {
        0..0
    };

    for i in (0..n_points).filter(|i| !interior.contains(i)) {
        let lo = i.saturating_sub(half_kernel);
        let hi = (i + half_kernel).min(n_points - 1);
        let used_weight: f32 = (lo..=hi).map(|j| weight(i.abs_diff(j))).sum();
        let scale = full_weight / used_weight;
        for f_idx in 0..n_fragments {
            let sum: f32 = (lo..=hi)
                .map(|j| xic[[f_idx, j]] * weight(i.abs_diff(j)))
                .sum();
            convolved[[f_idx, i]] = sum * scale;
        }
    }
}

#[cfg(test)]
mod tests;
