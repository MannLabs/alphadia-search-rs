# SIMD Dynamic Dispatch Strategy for Rust Package

## Current Implementation Status

**Implemented Core Architecture**
- Runtime backend selection with `SimdBackend` trait
- Scalar backend (universal fallback)
- NEON backend for ARM64 with custom optimizations
- Dynamic dispatch using `AtomicUsize` for backend override
- Score module functions: `axis_log_dot_product`, `axis_sqrt_dot_product`
- Convolution module functions with NEON optimizations
- Python integration with backend selection APIs

## Problem Solved

The implemented runtime backend selection addresses:
- Single wheel deployment (currently Scalar + NEON)
- Runtime optimization selection
- Cross-platform compatibility (ARM64 + x86_64 scalar fallback)

### Current Architecture Implementation

```rust
pub trait SimdBackend: Send + Sync {
    // Testing function
    fn test_backend(&self) -> String;

    // Score module functions (currently implemented)
    fn axis_log_dot_product(&self, array: &Array2<f32>, weights: &[f32]) -> Array1<f32>;
    fn axis_sqrt_dot_product(&self, array: &Array2<f32>, weights: &[f32]) -> Array1<f32>;

    // Convolution module functions (currently implemented)
    fn convolution(&self, kernel: &GaussianKernel, xic: &Array2<f32>) -> Array2<f32>;

    // Backend metadata
    fn name(&self) -> &'static str;
    fn is_available(&self) -> bool;
    fn priority(&self) -> Rank;
}

// Current rank system
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Rank {
    Scalar = 0,
    #[allow(dead_code)]
    Neon = 2,
}
```

**Performance Rankings:**
- **Scalar (0)**: Universal fallback, no SIMD acceleration
- **NEON (2)**: 128-bit ARM SIMD, standard on all AArch64 processors

### Backend Implementations

**Scalar Backend** (universal fallback):
```rust
pub struct ScalarBackend;
impl SimdBackend for ScalarBackend {
    fn is_available(&self) -> bool { true }
    fn priority(&self) -> Rank { Rank::Scalar }
    fn name(&self) -> &'static str { "scalar" }

    fn axis_log_dot_product(&self, array: &Array2<f32>, weights: &[f32]) -> Array1<f32> {
        crate::score::scalar::axis_log_dot_product_scalar(array, weights)
    }

    fn axis_sqrt_dot_product(&self, array: &Array2<f32>, weights: &[f32]) -> Array1<f32> {
        crate::score::scalar::axis_sqrt_dot_product_scalar(array, weights)
    }

    fn convolution(&self, kernel: &GaussianKernel, xic: &Array2<f32>) -> Array2<f32> {
        crate::convolution::scalar::convolution_scalar(kernel, xic)
    }
}
```

**ARM NEON Backend** (with custom optimizations):
```rust
#[cfg(target_arch = "aarch64")]
pub struct NeonBackend;

#[cfg(target_arch = "aarch64")]
impl SimdBackend for NeonBackend {
    fn is_available(&self) -> bool {
        neon_check::get()  // Using cpufeatures crate
    }
    fn priority(&self) -> Rank { Rank::Neon }
    fn name(&self) -> &'static str { "neon" }

    fn axis_log_dot_product(&self, array: &Array2<f32>, weights: &[f32]) -> Array1<f32> {
        crate::score::neon::axis_log_dot_product_neon(array, weights)
    }

    fn axis_sqrt_dot_product(&self, array: &Array2<f32>, weights: &[f32]) -> Array1<f32> {
        crate::score::neon::axis_sqrt_dot_product_neon(array, weights)
    }

    fn convolution(&self, kernel: &GaussianKernel, xic: &Array2<f32>) -> Array2<f32> {
        crate::convolution::neon::convolution_neon(kernel, xic)
    }
}
```

### Current Global Dispatcher Implementation

```rust
use std::sync::atomic::{AtomicUsize, Ordering};

// Current backend instances
static SCALAR: ScalarBackend = ScalarBackend;
#[cfg(target_arch = "aarch64")]
static NEON: NeonBackend = NeonBackend;

// Current backend registry
static BACKENDS: &[&dyn SimdBackend] = &[
    &SCALAR,
    #[cfg(target_arch = "aarch64")]
    &NEON,
];

// Current backend override system using AtomicUsize
static BACKEND_OVERRIDE: AtomicUsize = AtomicUsize::new(usize::MAX);

fn select_best_backend() -> &'static dyn SimdBackend {
    BACKENDS
        .iter()
        .copied()
        .filter(|b| b.is_available())
        .max_by_key(|b| b.priority())
        .unwrap_or(&SCALAR)
}

pub(crate) fn get_backend() -> &'static dyn SimdBackend {
    let override_idx = BACKEND_OVERRIDE.load(Ordering::Relaxed);

    if override_idx != usize::MAX {
        if let Some(backend) = BACKENDS.get(override_idx) {
            if backend.is_available() {
                return *backend;
            }
        }
    }

    // Natural selection based on availability and priority
    select_best_backend()
}

// Current public API functions
pub fn set_backend(backend_name: &str) -> Result<(), String> { /* implemented */ }
pub fn clear_backend() { /* implemented */ }
pub fn get_current_backend() -> String { /* implemented */ }
pub fn get_optimal_simd_backend() -> String { /* implemented */ }
```

### Current Python Integration

**Implemented Python Functions:**
```rust
#[pyfunction]
fn get_optimal_simd_backend() -> PyResult<String> {
    Ok(simd::get_optimal_simd_backend())
}

#[pyfunction]
fn set_simd_backend(backend_name: String) -> PyResult<()> {
    simd::set_backend(&backend_name).map_err(PyErr::new::<PyValueError, _>)
}

#[pyfunction]
fn clear_simd_backend() -> PyResult<()> {
    simd::clear_backend();
    Ok(())
}

#[pyfunction]
fn get_current_simd_backend() -> PyResult<String> {
    Ok(simd::get_current_backend())
}
```

These functions provide complete backend management from Python, allowing users to:
- Query the optimal backend for their system
- Override backend selection for testing/debugging
- Clear overrides to return to automatic selection
- Check which backend is currently active

## Current Benefits Achieved

- **Runtime optimization selection** - Automatic best backend selection
- **Backward compatible** - Existing API unchanged, drop-in replacement
- **Allocation-free** - Static backend registry eliminates heap allocations
- **Cross-platform** - Works on ARM64 (with NEON) and x86_64 (scalar fallback)
- **Testable** - Backend override API enables consistent testing
- **Extensible** - Clean architecture for adding new SIMD variants
- **Python integration** - Complete backend management from Python

## Current File Organization

**Implemented Structure**: Clean separation between dispatcher and algorithm-specific optimizations.

### Current Implementation
```
src/
├── simd/                      // Backend dispatcher and traits
│   ├── mod.rs                 // Dispatcher + trait + get_backend()
│   ├── scalar.rs              // ScalarBackend implementation
│   ├── neon.rs                // NeonBackend implementation
│   └── tests.rs               // Backend integration tests
├── score/                     // Score computation algorithms
│   ├── mod.rs                 // Public API using get_backend()
│   ├── scalar.rs              // Scalar score implementations
│   ├── neon.rs                // NEON-optimized score functions with custom log/sqrt
│   ├── tests.rs               // Cross-backend validation tests
│   └── tests_neon.rs          // NEON-specific tests
└── convolution/               // Convolution algorithms
    ├── mod.rs                 // Public API using get_backend()
    ├── scalar.rs              // Scalar convolution implementation
    ├── neon.rs                // NEON-optimized convolution with SIMD loops
    └── tests.rs               // Cross-backend validation tests
```

### Current Integration Pattern

The transition from static dispatch to runtime dispatch has been completed:

```rust
// Current implementation in src/score/mod.rs:
pub fn axis_log_dot_product(array: &Array2<f32>, weights: &[f32]) -> Array1<f32> {
    crate::simd::get_backend().axis_log_dot_product(array, weights)
}

pub fn axis_sqrt_dot_product(array: &Array2<f32>, weights: &[f32]) -> Array1<f32> {
    crate::simd::get_backend().axis_sqrt_dot_product(array, weights)
}
```

```rust
// Current implementation in src/convolution/mod.rs:
pub fn convolution(kernel: &GaussianKernel, xic: &Array2<f32>) -> Array2<f32> {
    crate::simd::get_backend().convolution(kernel, xic)
}
```

### Current Backend Implementation Pattern

```rust
// In src/simd/scalar.rs (implemented)
impl SimdBackend for ScalarBackend {
    fn axis_log_dot_product(&self, array: &Array2<f32>, weights: &[f32]) -> Array1<f32> {
        crate::score::scalar::axis_log_dot_product_scalar(array, weights)
    }

    fn convolution(&self, kernel: &GaussianKernel, xic: &Array2<f32>) -> Array2<f32> {
        crate::convolution::scalar::convolution_scalar(kernel, xic)
    }
}

// In src/simd/neon.rs (implemented)
impl SimdBackend for NeonBackend {
    fn axis_log_dot_product(&self, array: &Array2<f32>, weights: &[f32]) -> Array1<f32> {
        crate::score::neon::axis_log_dot_product_neon(array, weights)
    }

    fn convolution(&self, kernel: &GaussianKernel, xic: &Array2<f32>) -> Array2<f32> {
        crate::convolution::neon::convolution_neon(kernel, xic)
    }
}
```

## Implementation Summary

**Completed Architecture:**
- Runtime backend selection with trait-based dispatch
- Static backend registry with allocation-free dispatch
- Backend override system for testing and debugging
- Comprehensive Python integration
- Cross-backend validation testing
- Clean separation between dispatch and algorithm implementations
- Minimal refactoring approach preserving existing optimizations

The current implementation provides a solid foundation for SIMD acceleration with ARM64 NEON optimizations and universal scalar fallback, using a clean, extensible architecture.