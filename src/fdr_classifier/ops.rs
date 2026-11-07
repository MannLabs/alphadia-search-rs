//! Dense-layer operations of the FDR network on three matmul backends.
//!
//! - `Accelerate`: Apple's BLAS, which runs on the AMX unit (macOS only, the default there)
//! - `Simd`: the fused kernels of [`crate::mlp`] (NEON on aarch64, the default on other ARM)
//! - `Gemm`: the `gemm` crate with runtime AVX2/AVX-512 dispatch (the default on x86)
//!
//! The bias/ReLU/mask passes of the BLAS backends are branch-free selects: a conditional store
//! does not vectorise and mispredicts on the random ReLU pattern.

use crate::mlp::{self, Activation};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    #[cfg(target_os = "macos")]
    Accelerate,
    Simd,
    Gemm,
}

impl Backend {
    pub fn default_for_target() -> Self {
        #[cfg(target_os = "macos")]
        return Backend::Accelerate;
        #[cfg(all(not(target_os = "macos"), target_arch = "aarch64"))]
        return Backend::Simd;
        #[cfg(all(not(target_os = "macos"), not(target_arch = "aarch64")))]
        return Backend::Gemm;
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
            Backend::Gemm => "gemm",
        }
    }

    /// Backends available on this platform, the default first.
    pub fn available_names() -> Vec<&'static str> {
        let default = Self::default_for_target();
        let mut names = vec![default.name()];
        for name in ["accelerate", "simd", "gemm"] {
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
            "gemm" => Some(Backend::Gemm),
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
        _ => {
            sgemm(backend, false, false, rows, n, k, x, k, wt, n, 0.0, out, n);
            for row in out[..rows * n].chunks_exact_mut(n) {
                for (v, &b) in row.iter_mut().zip(bias) {
                    *v = (*v + b).max(0.0);
                }
            }
        }
    }
}

/// d_prev[rows x k] = (dz[rows x n] @ W) * (a_prev > 0); `w` is [n x k], `wt` the same as [k x n].
#[allow(clippy::too_many_arguments)]
pub fn dense_backward_input(
    backend: Backend,
    dz: &[f32],
    rows: usize,
    n: usize,
    w: &[f32],
    wt: &[f32],
    k: usize,
    a_prev: &[f32],
    d_prev: &mut [f32],
) {
    match backend {
        Backend::Simd => mlp::dense_backward_input(dz, rows, n, w, k, a_prev, d_prev),
        _ => {
            sgemm(
                backend, false, true, rows, k, n, dz, n, wt, n, 0.0, d_prev, k,
            );
            for (d, &a) in d_prev[..rows * k].iter_mut().zip(&a_prev[..rows * k]) {
                *d = if a > 0.0 { *d } else { 0.0 };
            }
        }
    }
}

/// grad_wt[k x n] += a_prev^T @ dz, grad_b[n] += colsum(dz)
#[allow(clippy::too_many_arguments)]
pub fn dense_weight_grad(
    backend: Backend,
    a_prev: &[f32],
    rows: usize,
    k: usize,
    dz: &[f32],
    n: usize,
    grad_wt: &mut [f32],
    grad_b: &mut [f32],
) {
    match backend {
        Backend::Simd => mlp::dense_weight_grad(a_prev, rows, k, dz, n, grad_wt, grad_b),
        _ => {
            sgemm(
                backend, true, false, k, n, rows, a_prev, k, dz, n, 1.0, grad_wt, n,
            );
            for row in dz[..rows * n].chunks_exact(n) {
                for (g, &d) in grad_b.iter_mut().zip(row) {
                    *g += d;
                }
            }
        }
    }
}

/// Row-major C[m x n] = op(A) @ op(B) + beta * C; `lda`/`ldb` are the stored row strides.
#[allow(clippy::too_many_arguments)]
fn sgemm(
    backend: Backend,
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
    match backend {
        #[cfg(target_os = "macos")]
        Backend::Accelerate => {
            accelerate::sgemm(trans_a, trans_b, m, n, k, a, lda, b, ldb, beta, c, ldc)
        }
        _ => gemm_crate(trans_a, trans_b, m, n, k, a, lda, b, ldb, beta, c, ldc),
    }
}

#[allow(clippy::too_many_arguments)]
fn gemm_crate(
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
    // (row stride, column stride) of op(X)
    let (a_rs, a_cs) = if trans_a {
        (1, lda as isize)
    } else {
        (lda as isize, 1)
    };
    let (b_rs, b_cs) = if trans_b {
        (1, ldb as isize)
    } else {
        (ldb as isize, 1)
    };
    assert!(c.len() >= (m - 1) * ldc + n, "output too small");
    // SAFETY: the strides above stay inside the slices for the given shapes, which the
    // callers derive from the same layer dimensions as the slice lengths.
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            c.as_mut_ptr(),
            1,
            ldc as isize,
            beta != 0.0,
            a.as_ptr(),
            a_cs,
            a_rs,
            b.as_ptr(),
            b_cs,
            b_rs,
            beta,
            1.0,
            false,
            false,
            false,
            gemm::Parallelism::None,
        );
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
        // SAFETY: as for the gemm crate; BLAS reads op(A), op(B) and writes C within the
        // shapes and strides passed.
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
