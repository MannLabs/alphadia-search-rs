# SIMD Dynamic Dispatch Strategy for Rust Package

## Problem

Current static dispatch limits deployment flexibility:
- Separate wheels needed for each CPU variant
- No runtime optimization selection
- Limited platform coverage

## Solution: Runtime Backend Selection

The proposed architecture provides clear separation with a `SimdBackend` trait that captures the four kernel families plus metadata, while per-ISA structs (Scalar, NEON, AVX2) remain simple and orthogonal. The `OnceLock` ensures the CPUID probe runs just once, providing zero cost after first call.

### Core Architecture

```rust
pub trait SimdBackend: Send + Sync {
    // Score module functions
    fn axis_dot_product(&self, array: &Array2<f32>, weights: &Vec<f32>) -> Array1<f32>;
    fn axis_log_dot_product(&self, array: &Array2<f32>, weights: &Vec<f32>) -> Array1<f32>;
    fn axis_sqrt_dot_product(&self, array: &Array2<f32>, weights: &Vec<f32>) -> Array1<f32>;
    fn axis_log_sum(&self, array: &Array2<f32>) -> Array1<f32>;

    // Kernel module functions
    fn gaussian_convolve_1d(&self, data: &Array1<f32>, kernel: &Array1<f32>) -> Array1<f32>;

    // Convolution module functions
    fn convolution_1d(&self, signal: &Array1<f32>, kernel: &Array1<f32>) -> Array1<f32>;
    fn convolution_2d(&self, signal: &Array2<f32>, kernel: &Array2<f32>) -> Array2<f32>;

    // Backend metadata
    fn name(&self) -> &'static str;
    fn is_available(&self) -> bool;
    fn priority(&self) -> Rank;
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Rank {
    Scalar = 0,
    Sse42 = 1,
    Neon = 2,
    Avx2 = 3,
    Avx512 = 4,
    Sve2 = 5,
}
```

**Improvement rationale:** Using a `Rank` enum expresses intent (ordering) without magic numbers, while `max_by_key` still works because `Rank` is `Copy + Ord`. Rankings are ordered by performance potential:
- **Scalar (0)**: Universal fallback, no SIMD acceleration
- **SSE4.2 (1)**: Basic 128-bit x86 SIMD, wide compatibility (95+ % of x86_64 CPUs)
- **NEON (2)**: 128-bit ARM SIMD, standard on all AArch64 processors
- **AVX2 (3)**: 256-bit x86 SIMD, excellent performance/compatibility balance
- **AVX-512 (4)**: 512-bit x86 SIMD, maximum throughput for compute-intensive workloads
- **SVE2 (5)**: Scalable ARM SIMD (128-2048 bits), future high-performance server architecture

### Backend Implementations

**Scalar Backend** (universal fallback):
```rust
pub struct ScalarBackend;
impl SimdBackend for ScalarBackend {
    fn is_available(&self) -> bool { true }
    fn priority(&self) -> Rank { Rank::Scalar }
    fn name(&self) -> &'static str { "scalar" }
    // ... implement functions with current scalar code
}
```

**ARM NEON Backend**:
```rust
#[cfg(target_arch = "aarch64")]
pub struct NeonBackend;

#[cfg(target_arch = "aarch64")]
impl SimdBackend for NeonBackend {
    fn is_available(&self) -> bool {
        cpufeatures::new!(neon, "neon");
        neon::get()  // Improved: use cpufeatures crate
    }
    fn priority(&self) -> Rank { Rank::Neon }
    fn name(&self) -> &'static str { "neon" }
    fn axis_log_dot_product(&self, array: &Array2<f32>, weights: &Vec<f32>) -> Array1<f32> {
        axis_log_dot_product_simd(array, weights) // Use existing implementation
    }
    // ... other functions
}
```

**x86_64 SSE4.2 Backend**:
```rust
#[cfg(target_arch = "x86_64")]
pub struct Sse42Backend;

#[cfg(target_arch = "x86_64")]
impl SimdBackend for Sse42Backend {
    fn is_available(&self) -> bool {
        cpufeatures::new!(sse42, "sse4.2");
        sse42::get()
    }
    fn priority(&self) -> Rank { Rank::Sse42 }
    fn name(&self) -> &'static str { "sse4.2" }
    // ... implement SSE4.2 versions
}
```

**x86_64 AVX2 Backend**:
```rust
#[cfg(target_arch = "x86_64")]
pub struct Avx2Backend;

#[cfg(target_arch = "x86_64")]
impl SimdBackend for Avx2Backend {
    fn is_available(&self) -> bool {
        cpufeatures::new!(avx2, "avx2");
        avx2::get()
    }
    fn priority(&self) -> Rank { Rank::Avx2 }
    fn name(&self) -> &'static str { "avx2" }
    // ... implement AVX2 versions
}
```

**x86_64 AVX-512 Backend**:
```rust
#[cfg(target_arch = "x86_64")]
pub struct Avx512Backend;

#[cfg(target_arch = "x86_64")]
impl SimdBackend for Avx512Backend {
    fn is_available(&self) -> bool {
        // Check for AVX-512F (foundation) + commonly available extensions
        cpufeatures::new!(avx512f, "avx512f");
        cpufeatures::new!(avx512vl, "avx512vl");
        cpufeatures::new!(avx512dq, "avx512dq");
        avx512f::get() && avx512vl::get() && avx512dq::get()
    }
    fn priority(&self) -> Rank { Rank::Avx512 }
    fn name(&self) -> &'static str { "avx512" }
    // ... implement AVX-512 versions
}
```

**ARM SVE2 Backend**:
```rust
#[cfg(target_arch = "aarch64")]
pub struct Sve2Backend;

#[cfg(target_arch = "aarch64")]
impl SimdBackend for Sve2Backend {
    fn is_available(&self) -> bool {
        cpufeatures::new!(sve2, "sve2");
        sve2::get()
    }
    fn priority(&self) -> Rank { Rank::Sve2 }
    fn name(&self) -> &'static str { "sve2" }
    // ... implement SVE2 versions with scalable vector lengths
}
```

**Improvement rationale:** The `cpufeatures` crate avoids nightly/macro hygiene issues and simplifies tests by allowing compile-time mocks.

### Global Dispatcher

```rust
use std::sync::OnceLock;

static SCALAR: ScalarBackend = ScalarBackend;
#[cfg(target_arch = "aarch64")]
static NEON: NeonBackend = NeonBackend;
#[cfg(target_arch = "aarch64")]
static SVE2: Sve2Backend = Sve2Backend;
#[cfg(target_arch = "x86_64")]
static SSE42: Sse42Backend = Sse42Backend;
#[cfg(target_arch = "x86_64")]
static AVX2: Avx2Backend = Avx2Backend;
#[cfg(target_arch = "x86_64")]
static AVX512: Avx512Backend = Avx512Backend;

static BACKENDS: &[&dyn SimdBackend] = &[
    &SCALAR,
    #[cfg(target_arch = "x86_64")]
    &SSE42,
    #[cfg(target_arch = "aarch64")]
    &NEON,
    #[cfg(target_arch = "x86_64")]
    &AVX2,
    #[cfg(target_arch = "x86_64")]
    &AVX512,
    #[cfg(target_arch = "aarch64")]
    &SVE2,
];

static BACKEND: OnceLock<&'static dyn SimdBackend> = OnceLock::new();

fn get_backend() -> &'static dyn SimdBackend {
    *BACKEND.get_or_init(|| select_best_backend())
}

fn select_best_backend() -> &'static dyn SimdBackend {
    // Environment override for reproducibility (testing/debugging)
    if let Ok(force) = std::env::var("ALPHA_RS_BACKEND") {
        if let Some(backend) = BACKENDS.iter()
            .find(|b| b.name() == force && b.is_available()) {
            return *backend;
        }
    }

    BACKENDS.iter()
        .copied()
        .filter(|b| b.is_available())
        .max_by_key(|b| b.priority())
        .unwrap_or(&SCALAR)
}

// Maintain existing API with thin wrappers
pub fn axis_dot_product(array: &Array2<f32>, weights: &[f32]) -> Array1<f32> {
    get_backend().axis_dot_product(array, weights)
}

pub fn axis_log_dot_product(array: &Array2<f32>, weights: &[f32]) -> Array1<f32> {
    get_backend().axis_log_dot_product(array, weights)
}

pub fn axis_sqrt_dot_product(array: &Array2<f32>, weights: &[f32]) -> Array1<f32> {
    get_backend().axis_sqrt_dot_product(array, weights)
}

pub fn axis_log_sum(array: &Array2<f32>) -> Array1<f32> {
    get_backend().axis_log_sum(array)
}
```

**Improvements integrated:**
- **Static backend registry:** Removes heap allocation and dynamic drop; polymorphic call cost remains a single indirection
- **Environment override:** Lets users force Scalar (e.g. CI on Rosetta) or compare results across ISAs
- **Thin wrappers:** Avoid boxing returns while maintaining API compatibility

### Python Integration

```rust
#[pyfunction]
fn get_simd_info() -> (String, bool) {
    let backend = get_backend();
    (backend.name().to_string(), backend.priority() > Rank::Scalar)
}
```

The `get_simd_info()` function provides a handy debug hook while API-compatibility is preserved because all public helpers (`axis_*`) proxy through `get_backend()`.

## Migration Plan

### Phase 1: Setup
1. Create `src/simd/` folder with `mod.rs` (dispatcher), `scalar.rs`, and backend files
2. Integrate `cpufeatures` crate for runtime feature detection
3. Replace existing conditional compilation with `get_backend()` calls
4. Keep existing SIMD implementations in place (score module functions)
5. Add environment variable override support (`ALPHA_RS_BACKEND`)

### Phase 2: SIMD Integration
1. **Score module**: Add `avx2.rs`, `avx512.rs`, `sse42.rs` alongside existing code
2. **Kernel module**: Add `neon.rs`, `avx2.rs`, `avx512.rs` for convolution ops
3. **Convolution module**: Add optimized implementations as separate files
4. **Backend wiring**: Connect new implementations to dispatcher in `simd.rs`
5. **Testing**: Validate consistency across backends with existing test suite
6. **Performance**: Benchmark and tune each implementation

### Phase 3: Distribution
1. Update build configuration for universal wheels
2. Build universal-2 wheels on macOS (`arm64` + `x86_64`)—runtime dispatch picks the right path
3. On Linux, ship manylinux2014 wheels with both AVX2 and scalar backends
4. Add CI/CD for multi-platform builds
5. Document `ALPHA_RS_BACKEND` env var and `get_simd_info()` helper in Python README
6. Validate deployment across targets

## Key Benefits

- **Single wheel per platform** - runtime optimization selection
- **Backward compatible** - existing API unchanged
- **Allocation-free** - static backend registry eliminates heap allocations
- **Reproducible** - environment override enables testing and debugging
- **Extensible** - easy to add new SIMD variants
- **Testable** - consistent results across backends with feature-gated testing

## File Organization Strategy

**Simplified Approach**: Keep SIMD implementations close to their algorithms and use minimal central infrastructure.

### Minimal Structure
```
src/
├── simd/
│   ├── mod.rs                 // Dispatcher + trait + get_backend() (~100 lines)
│   ├── scalar.rs              // ScalarBackend implementation
│   ├── neon.rs                // NeonBackend implementation
│   ├── sse42.rs               // Sse42Backend implementation
│   ├── avx2.rs                // Avx2Backend implementation
│   └── avx512.rs              // Avx512Backend implementation
├── score/
│   ├── mod.rs                 // Public API + current NEON function
│   ├── avx2.rs                // AVX2-optimized score functions
│   ├── avx512.rs              // AVX-512-optimized score functions
│   └── sse42.rs               // SSE4.2-optimized score functions
├── kernel/
│   ├── mod.rs                 // Current code + scalar implementations
│   ├── neon.rs                // NEON-optimized kernel functions
│   ├── avx2.rs                // AVX2-optimized kernel functions
│   └── avx512.rs              // AVX-512-optimized kernel functions
└── convolution/
    ├── mod.rs                 // Public API + scalar implementations
    ├── neon.rs                // NEON-optimized convolution
    ├── avx2.rs                // AVX2-optimized convolution
    └── avx512.rs              // AVX-512-optimized convolution
```

### Simple Integration Pattern
Replace existing conditional compilation with runtime dispatch:

```rust
// In src/score/mod.rs - Replace this:
pub fn axis_log_dot_product(array: &Array2<f32>, weights: &Vec<f32>) -> Array1<f32> {
    #[cfg(target_arch = "aarch64")]
    { axis_log_dot_product_simd(array, weights) }
    #[cfg(not(target_arch = "aarch64"))]
    { axis_log_dot_product_scalar(array, weights) }
}

// With this:
pub fn axis_log_dot_product(array: &Array2<f32>, weights: &Vec<f32>) -> Array1<f32> {
    crate::simd::get_backend().axis_log_dot_product(array, weights)
}
```

### Backend Implementation Pattern
```rust
// In src/simd/neon.rs
impl SimdBackend for NeonBackend {
    fn axis_log_dot_product(&self, array: &Array2<f32>, weights: &Vec<f32>) -> Array1<f32> {
        crate::score::axis_log_dot_product_simd(array, weights) // Use existing function
    }

    fn gaussian_convolve_1d(&self, data: &Array1<f32>, kernel: &Array1<f32>) -> Array1<f32> {
        crate::kernel::neon::gaussian_convolve_1d_neon(data, kernel) // Call module function
    }

    fn convolution_1d(&self, signal: &Array1<f32>, kernel: &Array1<f32>) -> Array1<f32> {
        crate::convolution::neon::convolution_1d_neon(signal, kernel) // Call module function
    }
}

// In src/simd/avx2.rs
impl SimdBackend for Avx2Backend {
    fn axis_log_dot_product(&self, array: &Array2<f32>, weights: &Vec<f32>) -> Array1<f32> {
        crate::score::avx2::axis_log_dot_product_avx2(array, weights) // Call module function
    }

    fn gaussian_convolve_1d(&self, data: &Array1<f32>, kernel: &Array1<f32>) -> Array1<f32> {
        crate::kernel::avx2::gaussian_convolve_1d_avx2(data, kernel) // Call module function
    }
}
```

**Why this is simpler:**
- **Dedicated simd/ folder** keeps all backend implementations organized in one place
- **Keep existing function implementations** where they are (score module already has SIMD)
- **Backends call module functions** - clean separation between dispatch and implementation
- **Minimal refactoring** of current codebase - existing functions can be reused directly
- **Easy to understand** - backend selection in `simd/mod.rs`, implementations stay local to modules

## Implementation Notes

- Use `OnceLock` for zero-cost dispatch after initialization
- Existing SIMD functions (`axis_log_dot_product_simd`) wrap directly into new backends
- `Rank` enum-based selection ensures best available backend with clear intent
- Static backend registry provides single indirection with no allocations
- Scalar fallback guarantees universal compatibility
- `cpufeatures` crate provides robust, testable feature detection
- Environment override (`ALPHA_RS_BACKEND`) enables reproducible testing across ISAs
- **Simplified Architecture:**
  - Dedicated `simd/` folder with clean backend separation
  - SIMD implementations stay co-located with their algorithms in modules
  - Minimal refactoring of existing codebase (score module already has NEON)
  - Backends delegate to existing module functions for maximum code reuse
- **Backend Coverage:** SSE4.2/AVX2/AVX-512 for x86_64, NEON/SVE2 for ARM64
- **Zero Configuration:** Automatically selects best available backend at runtime

This design is clean, practical, and ready for production deployment with simple allocation-free runtime dispatch that maintains full API compatibility.