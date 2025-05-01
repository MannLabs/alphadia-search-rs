use numpy::ndarray::{Array1, ArrayView1};
use std::f32::consts::PI;

#[derive(Clone)]
pub struct GaussianKernel {
    pub fwhm_rt: f32,
    pub sigma_scale_rt: f32,
    pub kernel_width: usize,
    pub kernel_array: Array1<f32>,
}

impl GaussianKernel {
    pub fn new(
        fwhm_rt: f32,
        sigma_scale_rt: f32,
        kernel_width: usize,
        rt_resolution: f32,
    ) -> Self {
        // Ensure kernel dimension is even
        let kernel_width = (kernel_width as f32 / 2.0).ceil() as usize * 2;
        
        // Calculate RT sigma
        let sigma = fwhm_rt / 2.3548;
        let rt_sigma = sigma * sigma_scale_rt / rt_resolution;
        
        // Generate kernel
        let kernel_array = Self::gaussian_kernel_1d(kernel_width, rt_sigma);
        
        Self {
            fwhm_rt,
            sigma_scale_rt,
            kernel_width,
            kernel_array,
        }
    }


    fn gaussian_kernel_1d(size: usize, sigma: f32) -> Array1<f32> {
        let half_size = (size / 2) as i32;
        
        let mut weights = Array1::<f32>::zeros(size);
        
        // Calculate normalization factor
        let normalization = 1.0 / (sigma * (2.0 * PI).sqrt());
        
        // Fill weights array
        for x in -half_size..half_size {
            // Calculate weight using univariate normal distribution formula
            let exponent = -0.5 * (x as f32).powi(2) / sigma.powi(2);
            let weight = normalization * exponent.exp();
            
            // Convert to array index
            let idx = (x + half_size) as usize;
            weights[idx] = weight;
        }
        
        // Normalize weights so they sum to 1
        let sum = weights.sum();
        weights /= sum;
        
        weights
    }
}

// Default implementation with reasonable defaults
impl Default for GaussianKernel {
    fn default() -> Self {
        Self::new(
            10.0,   // fwhm_rt
            1.0,    // sigma_scale_rt
            30,     // kernel_width
            60.0,   // default rt_resolution (seconds)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
}