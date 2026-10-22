use crate::kernel::GaussianKernel;
use approx::assert_relative_eq;

#[test]
fn test_gaussian_kernel_dimensions() {
    let kernel = GaussianKernel::default();
    assert_eq!(kernel.kernel_width, 30);
}

#[test]
fn test_kernel_creation() {
    let kernel = GaussianKernel::new(10.0, 1.0, 30, 60.0);
    let rt_kernel = kernel.kernel_array;

    // Check dimensions
    assert_eq!(rt_kernel.len(), 30);

    // Check sum is approximately 1.0 (since weights are normalized)
    assert_relative_eq!(rt_kernel.sum(), 1.0, epsilon = 1e-6);

    // Center should have highest value
    let center_val = rt_kernel[15];
    for i in 0..30 {
        assert!(rt_kernel[i] <= center_val);
    }
}

#[test]
fn test_kernel_fwhm_is_scaled_by_rt_resolution() {
    // A 3 s FWHM on a 1.5 s cycle is a 2-cycle FWHM.
    let seconds = GaussianKernel::new(3.0, 1.0, 20, 1.5);
    let cycles = GaussianKernel::new(2.0, 1.0, 20, 1.0);

    for (a, b) in seconds.kernel_array.iter().zip(cycles.kernel_array.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-6);
    }
}
