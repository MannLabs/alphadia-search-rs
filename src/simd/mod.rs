use std::sync::atomic::{AtomicUsize, Ordering};

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Rank {
    Scalar = 0,
    #[allow(dead_code)]
    Neon = 2,
}

/// The operations that can have SIMD kernels. The default of every operation is the scalar
/// kernel. A backend overrides only the operations that it has its own kernels for. Callers get
/// the backend from [`get_backend`].
pub trait SimdBackend: Send + Sync {
    // Dummy function to track which backend implementation is called
    #[allow(dead_code)]
    fn test_backend(&self) -> String;

    // Score module functions
    fn axis_log_dot_product(
        &self,
        array: &numpy::ndarray::Array2<f32>,
        weights: &[f32],
    ) -> numpy::ndarray::Array1<f32> {
        crate::score::scalar::axis_log_dot_product_scalar(array, weights)
    }
    fn axis_sqrt_dot_product(
        &self,
        array: &numpy::ndarray::Array2<f32>,
        weights: &[f32],
    ) -> numpy::ndarray::Array1<f32> {
        crate::score::scalar::axis_sqrt_dot_product_scalar(array, weights)
    }

    // Convolution module functions
    fn convolution(
        &self,
        kernel: &crate::kernel::GaussianKernel,
        xic: &numpy::ndarray::Array2<f32>,
    ) -> numpy::ndarray::Array2<f32> {
        crate::convolution::scalar::convolution_scalar(kernel, xic)
    }

    // Quantile transform of the FDR classifier: the normal scores of one feature column
    fn quantile_transform_column(
        &self,
        map: &crate::fdr_classifier::quantile::FeatureMap,
        x: &[f32],
        out: &mut [f32],
    ) {
        map.apply_column(x, out)
    }

    // Layer kernels of the FDR network
    /// `out[rows x n] = activation(bias + x[rows x k] @ wt[k x n])`
    #[allow(clippy::too_many_arguments)]
    fn mlp_dense_forward(
        &self,
        x: &[f32],
        rows: usize,
        k: usize,
        wt: &[f32],
        n: usize,
        bias: &[f32],
        activation: crate::mlp::Activation,
        out: &mut [f32],
    ) {
        crate::mlp::scalar::dense_forward_scalar(x, rows, k, wt, n, bias, activation, out)
    }
    /// `d_prev[rows x k] = (dz[rows x n] @ w[n x k]) * (a_prev > 0)`
    #[allow(clippy::too_many_arguments)]
    fn mlp_dense_backward_input(
        &self,
        dz: &[f32],
        rows: usize,
        n: usize,
        w: &[f32],
        k: usize,
        a_prev: &[f32],
        d_prev: &mut [f32],
    ) {
        crate::mlp::scalar::dense_backward_input_scalar(dz, rows, n, w, k, a_prev, d_prev)
    }
    /// `grad_wt[k x n] += a_prev[rows x k]^T @ dz[rows x n]`, `grad_b[n] += colsum(dz)`
    #[allow(clippy::too_many_arguments)]
    fn mlp_dense_weight_grad(
        &self,
        a_prev: &[f32],
        rows: usize,
        k: usize,
        dz: &[f32],
        n: usize,
        grad_wt: &mut [f32],
        grad_b: &mut [f32],
    ) {
        crate::mlp::scalar::dense_weight_grad_scalar(a_prev, rows, k, dz, n, grad_wt, grad_b)
    }
    /// The single-logit output layer and the BCE-with-logits loss in one pass:
    /// `logit = a[r] . w + b`, `d = (sigmoid(logit) - y[r]) * loss_scale`,
    /// `dz_prev[r] = d * w * (a[r] > 0)`, `grad_w += d * a[r]`, `grad_b += d`
    #[allow(clippy::too_many_arguments)]
    fn mlp_output_layer_train(
        &self,
        a: &[f32],
        rows: usize,
        k: usize,
        w: &[f32],
        b: f32,
        y: &[f32],
        loss_scale: f32,
        dz_prev: &mut [f32],
        grad_w: &mut [f32],
        grad_b: &mut f32,
    ) {
        crate::mlp::scalar::output_layer_train_scalar(
            a, rows, k, w, b, y, loss_scale, dz_prev, grad_w, grad_b,
        )
    }
    /// `logits[r] = a[r] . w + b`
    fn mlp_output_layer_logits(
        &self,
        a: &[f32],
        rows: usize,
        k: usize,
        w: &[f32],
        b: f32,
        logits: &mut [f32],
    ) {
        crate::mlp::scalar::output_layer_logits_scalar(a, rows, k, w, b, logits)
    }

    // Backend metadata
    fn name(&self) -> &'static str;
    fn is_available(&self) -> bool;
    fn priority(&self) -> Rank;
}

// Import backend implementations
mod neon;
mod scalar;

#[cfg(target_arch = "aarch64")]
use neon::NeonBackend;
use scalar::ScalarBackend;

// Static backend instances
static SCALAR: ScalarBackend = ScalarBackend;
#[cfg(target_arch = "aarch64")]
static NEON: NeonBackend = NeonBackend;

// Backend registry
static BACKENDS: &[&dyn SimdBackend] = &[
    &SCALAR,
    #[cfg(target_arch = "aarch64")]
    &NEON,
];

// Simple backend override: usize::MAX means no override
static BACKEND_OVERRIDE: AtomicUsize = AtomicUsize::new(usize::MAX);

fn select_best_backend() -> &'static dyn SimdBackend {
    BACKENDS
        .iter()
        .copied()
        .filter(|b| b.is_available())
        .max_by_key(|b| b.priority())
        .unwrap_or(&SCALAR)
}

// Public API - dummy function to track backend usage
#[allow(dead_code)]
pub fn test_backend() -> String {
    get_backend().test_backend()
}

// Internal API for score module
pub(crate) fn get_backend() -> &'static dyn SimdBackend {
    let override_idx = BACKEND_OVERRIDE.load(Ordering::Relaxed);

    if override_idx != usize::MAX {
        // Use the overridden backend if it's still available
        if let Some(backend) = BACKENDS.get(override_idx) {
            if backend.is_available() {
                return *backend;
            }
        }
    }

    // Natural selection based on availability and priority
    select_best_backend()
}

// Public API - set a specific backend by name
pub fn set_backend(backend_name: &str) -> Result<(), String> {
    // Find the backend by name and get its index
    if let Some((idx, backend)) = BACKENDS
        .iter()
        .enumerate()
        .find(|(_, b)| b.name() == backend_name)
    {
        if backend.is_available() {
            // Set the backend override by index
            BACKEND_OVERRIDE.store(idx, Ordering::Relaxed);
            Ok(())
        } else {
            Err(format!(
                "Backend '{backend_name}' is not available on this system"
            ))
        }
    } else {
        let available: Vec<_> = BACKENDS.iter().map(|b| b.name()).collect();
        Err(format!(
            "Unknown backend '{backend_name}'. Available backends: {available:?}"
        ))
    }
}

// Public API - clear any backend override (use natural selection)
pub fn clear_backend() {
    BACKEND_OVERRIDE.store(usize::MAX, Ordering::Relaxed);
}

// Public API - get currently selected backend name
pub fn get_current_backend() -> String {
    get_backend().name().to_string()
}

// Utility function for Python integration
pub fn get_optimal_simd_backend() -> String {
    select_best_backend().name().to_string()
}

// Tests module
#[cfg(test)]
mod tests;
