use std::sync::atomic::{AtomicUsize, Ordering};

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Rank {
    Scalar = 0,
    #[allow(dead_code)]
    Neon = 2,
}

pub trait SimdBackend: Send + Sync {
    // Dummy function to track which backend implementation is called
    #[allow(dead_code)]
    fn test_backend(&self) -> String;

    // Score module functions (actually used)
    fn axis_log_dot_product(
        &self,
        array: &numpy::ndarray::Array2<f32>,
        weights: &[f32],
    ) -> numpy::ndarray::Array1<f32>;
    fn axis_sqrt_dot_product(
        &self,
        array: &numpy::ndarray::Array2<f32>,
        weights: &[f32],
    ) -> numpy::ndarray::Array1<f32>;

    // Convolution module functions
    fn convolution(
        &self,
        kernel: &crate::kernel::GaussianKernel,
        xic: &numpy::ndarray::Array2<f32>,
    ) -> numpy::ndarray::Array2<f32>;

    // Dense-layer kernels of the mlp module
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
    );
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
    );
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
    );
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
    ) -> f32;
    fn mlp_output_layer_logits(
        &self,
        a: &[f32],
        rows: usize,
        k: usize,
        w: &[f32],
        b: f32,
        logits: &mut [f32],
    );

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
