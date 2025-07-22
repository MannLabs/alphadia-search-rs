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
