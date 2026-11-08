//! Dense-layer operations of the FDR network on two matmul backends.
//!
//! - `Accelerate`: Apple's BLAS, which runs on the AMX unit (macOS only, the default there)
//! - `Simd`: the fused kernels of [`crate::mlp`] through the SIMD backend registry: NEON on
//!   aarch64, AVX-512 or AVX2 on x86_64, scalar otherwise (the default outside macOS)
//!
//! The bias/ReLU/mask passes of the BLAS backend are branch-free selects: a conditional store
//! does not vectorise and mispredicts on the random ReLU pattern.

use crate::mlp::{self, Activation};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    #[cfg(target_os = "macos")]
    Accelerate,
    Simd,
}

impl Backend {
    pub fn default_for_target() -> Self {
        #[cfg(target_os = "macos")]
        return Backend::Accelerate;
        #[cfg(not(target_os = "macos"))]
        return Backend::Simd;
    }

    pub fn from_name(name: &str) -> Result<Self, String> {
        Self::from_name_unchecked(name).ok_or_else(|| {
            format!(
                "unknown or unavailable matmul backend '{name}', expected one of {:?}",
                Self::available_names()
            )
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            #[cfg(target_os = "macos")]
            Backend::Accelerate => "accelerate",
            Backend::Simd => "simd",
        }
    }

    /// Backends available on this platform, the default first.
    pub fn available_names() -> Vec<&'static str> {
        let default = Self::default_for_target();
        let mut names = vec![default.name()];
        for name in ["accelerate", "simd"] {
            if name != default.name() && Self::from_name_unchecked(name).is_some() {
                names.push(name);
            }
        }
        names
    }

    fn from_name_unchecked(name: &str) -> Option<Self> {
        match name {
            #[cfg(target_os = "macos")]
            "accelerate" => Some(Backend::Accelerate),
            "simd" => Some(Backend::Simd),
            _ => None,
        }
    }
}

/// out[rows x n] = max(bias + x[rows x k] @ wt[k x n], 0)
#[allow(clippy::too_many_arguments)]
pub fn dense_forward_relu(
    backend: Backend,
    x: &[f32],
    rows: usize,
    k: usize,
    wt: &[f32],
    n: usize,
    bias: &[f32],
    out: &mut [f32],
) {
    match backend {
        Backend::Simd => mlp::dense_forward(x, rows, k, wt, n, bias, Activation::Relu, out),
        #[cfg(target_os = "macos")]
        Backend::Accelerate => {
            accelerate::sgemm(false, false, rows, n, k, x, k, wt, n, 0.0, out, n);
            for row in out[..rows * n].chunks_exact_mut(n) {
                for (v, &b) in row.iter_mut().zip(bias) {
                    *v = (*v + b).max(0.0);
                }
            }
        }
    }
}

/// The per-layer kernels of a training step on Accelerate; the output layer stays on the
/// SIMD kernels (a matrix-vector product).
#[cfg(target_os = "macos")]
pub struct Accelerate;

#[cfg(target_os = "macos")]
impl mlp::LayerKernels for Accelerate {
    fn dense_forward(
        &self,
        x: &[f32],
        rows: usize,
        k: usize,
        wt: &[f32],
        n: usize,
        bias: &[f32],
        activation: Activation,
        out: &mut [f32],
    ) {
        accelerate::sgemm(false, false, rows, n, k, x, k, wt, n, 0.0, out, n);
        for row in out[..rows * n].chunks_exact_mut(n) {
            for (v, &b) in row.iter_mut().zip(bias) {
                *v += b;
                if activation == Activation::Relu {
                    *v = v.max(0.0);
                }
            }
        }
    }

    /// d_prev[rows x k] = (dz[rows x n] @ W) * (a_prev > 0)
    fn dense_backward_input(
        &self,
        dz: &[f32],
        rows: usize,
        n: usize,
        _w: &[f32],
        wt: &[f32],
        k: usize,
        a_prev: &[f32],
        d_prev: &mut [f32],
    ) {
        accelerate::sgemm(false, true, rows, k, n, dz, n, wt, n, 0.0, d_prev, k);
        for (d, &a) in d_prev[..rows * k].iter_mut().zip(&a_prev[..rows * k]) {
            *d = if a > 0.0 { *d } else { 0.0 };
        }
    }

    /// grad_wt[k x n] += a_prev^T @ dz, grad_b[n] += colsum(dz)
    fn dense_weight_grad(
        &self,
        a_prev: &[f32],
        rows: usize,
        k: usize,
        dz: &[f32],
        n: usize,
        grad_wt: &mut [f32],
        grad_b: &mut [f32],
    ) {
        accelerate::sgemm(true, false, k, n, rows, a_prev, k, dz, n, 1.0, grad_wt, n);
        for row in dz[..rows * n].chunks_exact(n) {
            for (g, &d) in grad_b.iter_mut().zip(row) {
                *g += d;
            }
        }
    }

    fn output_layer_train(
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
        mlp::output_layer_train(a, rows, k, w, b, y, loss_scale, dz_prev, grad_w, grad_b);
    }
}

#[cfg(target_os = "macos")]
mod accelerate {
    const ROW_MAJOR: i32 = 101;
    const NO_TRANS: i32 = 111;
    const TRANS: i32 = 112;

    #[link(name = "Accelerate", kind = "framework")]
    extern "C" {
        fn cblas_sgemm(
            order: i32,
            trans_a: i32,
            trans_b: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sgemm(
        trans_a: bool,
        trans_b: bool,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        lda: usize,
        b: &[f32],
        ldb: usize,
        beta: f32,
        c: &mut [f32],
        ldc: usize,
    ) {
        assert!(c.len() >= (m - 1) * ldc + n, "output too small");
        // SAFETY: BLAS reads op(A), op(B) and writes C within the shapes and strides passed,
        // which the callers derive from the same layer dimensions as the slice lengths.
        unsafe {
            cblas_sgemm(
                ROW_MAJOR,
                if trans_a { TRANS } else { NO_TRANS },
                if trans_b { TRANS } else { NO_TRANS },
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a.as_ptr(),
                lda as i32,
                b.as_ptr(),
                ldb as i32,
                beta,
                c.as_mut_ptr(),
                ldc as i32,
            );
        }
    }
}
