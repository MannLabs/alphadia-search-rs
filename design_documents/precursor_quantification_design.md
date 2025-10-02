# Design Document: Precursor Quantification from SpecLibFlat

## Executive Summary

This document outlines design options for quantifying precursors from `SpecLibFlat` and returning fragment quantification data, with focus on minimal memory footprint and multithreading compatibility. Two primary approaches are analyzed: storing quantification data back in the original `SpecLibFlat` vs. returning a new `SpecLibFlatQuantified` structure.

## Current Architecture Analysis

### Memory Management Patterns
- **>99.9% memory overhead reduction** achieved via flat array storage
- **Zero-copy access patterns** using slices (`&[f32]`) instead of `Vec<f32>`
- **Pre-allocation strategies** with exact capacity estimation
- **Memory footprint tracking** via `memory_footprint_bytes()` trait implementation
- **Consolidated storage** with index ranges instead of individual fragment allocations

### Threading Architecture
- **Rayon-based parallelization** for candidate scoring (`par_iter()`)
- **Thread-safe SIMD backend selection** using atomic operations
- **Work stealing** through Rayon's parallel iterators
- **Zero-allocation hot paths** to avoid thread contention

### Current Quantification Pipeline
```
SpecLibFlat → Precursor → DenseXICObservation → Feature Calculation → CandidateFeature
```

Current quantification features include:
- Fragment correlations (mean, median, std)
- Intensity correlations
- Hyperscore calculations
- Ion series analysis (longest b/y series)
- Mass error calculations
- RT deviation metrics

## Design Options

### Option 1: In-Place Modification with Quantified Variants

**Concept**: Extend existing `SpecLibFlat` with optional quantification data storage.

```rust
#[pyclass]
pub struct SpecLibFlat {
    // Existing fields...

    // Optional quantification data (None = not quantified)
    fragment_mz_observed: Option<Vec<f32>>,
    fragment_intensity_observed: Option<Vec<f32>>,
    fragment_correlation_observed: Option<Vec<f32>>,
    fragment_mass_error_observed: Option<Vec<f32>>,
}

impl SpecLibFlat {
    pub fn quantify_precursors<T: DIADataTrait + Sync>(
        &mut self,
        dia_data: &T,
        candidates: &CandidateCollection,
        params: &QuantificationParameters,
    ) -> Result<(), QuantificationError> {
        // Initialize quantification arrays if not present
        self.initialize_quantification_storage();

        // Parallel quantification
        candidates.par_iter()
            .for_each(|candidate| {
                self.quantify_single_precursor(dia_data, candidate, params);
            });

        Ok(())
    }

    fn initialize_quantification_storage(&mut self) {
        if self.fragment_mz_observed.is_none() {
            let total_fragments = self.fragment_mz.len();
            self.fragment_mz_observed = Some(vec![0.0; total_fragments]);
            self.fragment_intensity_observed = Some(vec![0.0; total_fragments]);
            self.fragment_correlation_observed = Some(vec![0.0; total_fragments]);
            self.fragment_mass_error_observed = Some(vec![0.0; total_fragments]);
        }
    }
}
```

**Pros**:
- **Minimal memory overhead**: Single structure, no duplication
- **Zero-copy quantification access**: Direct slice access to quantified data
- **Preserves existing API**: Minimal breaking changes
- **Thread-safe updates**: Atomic writes to disjoint array ranges per precursor

**Cons**:
- **Mutable state complexity**: Requires `&mut self` for quantification
- **Memory waste for partial quantification**: Full arrays allocated even if only subset quantified
- **API breaking changes**: Optional fields change memory layout and serialization
- **Thread safety concerns**: Requires careful coordination for concurrent quantification

### Option 2: Separate SpecLibFlatQuantified Structure

**Concept**: Return new quantified structure, preserving original immutable `SpecLibFlat`.

```rust
pub struct QuantificationEngine {
    params: QuantificationParameters,
}

impl QuantificationEngine {
    pub fn quantify_precursors<T: DIADataTrait + Sync>(
        &self,
        lib: &SpecLibFlat,
        dia_data: &T,
        candidates: &CandidateCollection,
    ) -> SpecLibFlatQuantified {
        // Build quantified precursors in parallel
        let quantified_precursors: Vec<PrecursorQuantified> = candidates
            .par_iter()
            .filter_map(|candidate| {
                lib.get_precursor_by_idx(candidate.precursor_idx)
                    .map(|precursor| self.quantify_precursor(&precursor, dia_data, candidate))
            })
            .collect();

        // Convert to flat storage
        SpecLibFlatQuantified::from_precursor_quantified_vec(quantified_precursors)
    }

    fn quantify_precursor<T: DIADataTrait>(
        &self,
        precursor: &Precursor,
        dia_data: &T,
        candidate: &Candidate,
    ) -> PrecursorQuantified {
        // Extract XIC observation
        let xic_obs = DenseXICObservation::new(
            dia_data,
            candidate.cycle_start,
            candidate.cycle_stop,
            &precursor.fragment_mz,
            self.params.mz_tolerance,
        );

        // Calculate quantification metrics per fragment
        let fragment_mz_observed = self.calculate_observed_mz(&xic_obs, precursor);
        let fragment_intensity_observed = self.calculate_observed_intensity(&xic_obs);
        let fragment_correlation_observed = self.calculate_fragment_correlations(&xic_obs);
        let fragment_mass_error_observed = self.calculate_mass_errors(&fragment_mz_observed, &precursor.fragment_mz);

        PrecursorQuantified {
            precursor_idx: precursor.precursor_idx,
            mz: precursor.mz,
            rt: precursor.rt,
            naa: precursor.naa,
            fragment_mz: precursor.fragment_mz.clone(),
            fragment_intensity: precursor.fragment_intensity.clone(),
            fragment_cardinality: precursor.fragment_cardinality.clone(),
            fragment_charge: precursor.fragment_charge.clone(),
            fragment_loss_type: precursor.fragment_loss_type.clone(),
            fragment_number: precursor.fragment_number.clone(),
            fragment_position: precursor.fragment_position.clone(),
            fragment_type: precursor.fragment_type.clone(),
            fragment_mz_observed,
            fragment_intensity_observed,
            fragment_correlation_observed,
            fragment_mass_error_observed,
        }
    }
}
```

**Pros**:
- **Immutable original data**: No modification of existing `SpecLibFlat`
- **Memory-efficient for partial quantification**: Only stores quantified precursors
- **Clean separation of concerns**: Quantification logic isolated in separate module
- **Perfect thread safety**: No shared mutable state
- **Flexible data retention**: Can store subset of precursors without memory waste

**Cons**:
- **Memory duplication**: Library data duplicated in quantified structure
- **Additional structure complexity**: Two similar but different data structures
- **Potential cache locality issues**: Quantified data separate from library data

### Option 3: Hybrid Lazy Quantification Cache

**Concept**: On-demand quantification with memory-efficient caching.

```rust
pub struct QuantificationCache {
    quantified_fragments: HashMap<usize, FragmentQuantification>, // precursor_idx -> quantification
    memory_limit: usize,
    lru_tracker: LinkedHashMap<usize, ()>, // LRU eviction
}

pub struct FragmentQuantification {
    fragment_mz_observed: Vec<f32>,
    fragment_intensity_observed: Vec<f32>,
    fragment_correlation_observed: Vec<f32>,
    fragment_mass_error_observed: Vec<f32>,
}

#[pyclass]
pub struct SpecLibFlatWithQuantification {
    speclib: SpecLibFlat,
    quantification_cache: Arc<RwLock<QuantificationCache>>,
    quantification_engine: QuantificationEngine,
}

impl SpecLibFlatWithQuantification {
    pub fn get_precursor_quantified<T: DIADataTrait + Sync>(
        &self,
        precursor_idx: usize,
        dia_data: &T,
        candidate: &Candidate,
    ) -> Option<PrecursorQuantified> {
        // Check cache first
        {
            let cache = self.quantification_cache.read().unwrap();
            if let Some(quant) = cache.quantified_fragments.get(&precursor_idx) {
                if let Some(precursor) = self.speclib.get_precursor_by_idx(precursor_idx) {
                    return Some(self.combine_precursor_with_quantification(precursor, quant));
                }
            }
        }

        // Quantify on-demand
        if let Some(precursor) = self.speclib.get_precursor_by_idx(precursor_idx) {
            let quantification = self.quantification_engine.quantify_precursor(&precursor, dia_data, candidate);

            // Update cache
            {
                let mut cache = self.quantification_cache.write().unwrap();
                cache.insert_with_eviction(precursor_idx, &quantification);
            }

            Some(quantification)
        } else {
            None
        }
    }
}
```

**Pros**:
- **Memory-bounded quantification**: LRU cache prevents unbounded memory growth
- **Lazy computation**: Only quantifies requested precursors
- **Thread-safe caching**: RwLock provides concurrent read access
- **Preserves original data**: No modification of base `SpecLibFlat`

**Cons**:
- **Cache complexity**: HashMap overhead and LRU management
- **Lock contention**: RwLock can become bottleneck under high concurrency
- **Non-deterministic memory usage**: Cache size varies with access patterns

## Memory Footprint Analysis

### Current SpecLibFlat Memory (Example):
- 10,000 precursors, average 20 fragments each = 200,000 fragments
- Per fragment: 8 arrays × 4 bytes = 32 bytes
- Total fragment data: 200,000 × 32 = 6.4 MB
- Precursor metadata: 10,000 × 24 = 240 KB
- **Total: ~6.6 MB**

### Option 1 (In-Place):
- Additional quantification arrays: 4 × 200,000 × 4 = 3.2 MB
- **Total: 6.6 + 3.2 = 9.8 MB (+48% overhead)**

### Option 2 (Separate Structure):
- Original library: 6.6 MB (unchanged)
- Quantified subset (50% quantified): 5 MB library data + 1.6 MB quantification = 6.6 MB
- **Total: 6.6 + 6.6 = 13.2 MB (+100% overhead for full quantification)**
- **Partial quantification advantage**: Only 6.6 + 3.3 = 9.9 MB for 50% quantified

### Option 3 (Hybrid Cache):
- Original library: 6.6 MB (unchanged)
- Cache overhead: HashMap + LRU = ~10% of cached data
- Cache for 1,000 precursors: ~660 KB + 66 KB overhead = 726 KB
- **Total: 6.6 + 0.7 = 7.3 MB (+11% overhead with bounded cache)**

## Threading Performance Analysis

### Option 1 Concurrency Model:
```rust
// Parallel quantification with range-based partitioning
candidates.chunks(chunk_size)
    .par_iter()
    .for_each(|chunk| {
        for candidate in chunk {
            // Each thread writes to disjoint fragment ranges
            let start_idx = speclib.precursor_start_idx[array_index];
            let stop_idx = speclib.precursor_stop_idx[array_index];

            // Safe: no overlap between precursor fragment ranges
            unsafe {
                let mz_slice = &mut speclib.fragment_mz_observed.as_mut_slice()[start_idx..stop_idx];
                // Quantify into slice
            }
        }
    });
```

### Option 2 Concurrency Model:
```rust
// Embarrassingly parallel - no shared state
let quantified: Vec<PrecursorQuantified> = candidates
    .par_iter()
    .filter_map(|candidate| {
        // Each thread works independently
        speclib.get_precursor_by_idx(candidate.precursor_idx)
            .map(|precursor| quantify_precursor(&precursor, dia_data, candidate))
    })
    .collect();
```

### Threading Performance Expectations:
- **Option 1**: Moderate parallelism due to mutable shared state, requires synchronization
- **Option 2**: Excellent parallelism, embarrassingly parallel workload
- **Option 3**: Limited by RwLock contention, especially for cache misses

## Recommendation

**Option 2 (Separate SpecLibFlatQuantified Structure)** is recommended based on:

### Primary Benefits:
1. **Excellent thread safety**: No shared mutable state, perfect parallelization
2. **Memory efficiency for partial quantification**: Common use case of quantifying subsets
3. **Clean architecture**: Separation of concerns between library and quantification
4. **API stability**: No breaking changes to existing `SpecLibFlat`

### Implementation Strategy:
```rust
pub struct QuantificationWorkflow {
    engine: QuantificationEngine,
    params: QuantificationParameters,
}

impl QuantificationWorkflow {
    pub fn quantify_candidates<T: DIADataTrait + Sync>(
        &self,
        lib: &SpecLibFlat,
        dia_data: &T,
        candidates: &CandidateCollection,
    ) -> QuantificationResult {
        let start = Instant::now();

        let quantified = self.engine.quantify_precursors(lib, dia_data, candidates);

        let duration = start.elapsed();
        let throughput = candidates.len() as f64 / duration.as_secs_f64();

        QuantificationResult {
            quantified_library: quantified,
            processing_stats: ProcessingStats {
                candidates_processed: candidates.len(),
                processing_time: duration,
                throughput_per_second: throughput,
            },
        }
    }
}
```

### Performance Optimizations:
1. **Batch XIC extraction**: Group candidates by RT window to reuse observations
2. **SIMD-optimized correlation**: Leverage existing SIMD infrastructure
3. **Memory pool allocation**: Pre-allocate fragment arrays to avoid allocation overhead
4. **Streaming quantification**: Process candidates in chunks to bound memory usage

This approach provides the best balance of memory efficiency, thread safety, and maintainability while leveraging the existing high-performance infrastructure in alphadia-search-rs.

## Implementation Roadmap

### Phase 1: Core Infrastructure
1. Implement `QuantificationEngine` with basic quantification methods
2. Add quantification parameter configuration
3. Create unit tests for individual quantification functions

### Phase 2: Integration
1. Integrate with existing `DenseXICObservation` workflow
2. Add parallel processing with Rayon
3. Implement performance monitoring and throughput reporting

### Phase 3: Optimization
1. Add SIMD-optimized correlation calculations
2. Implement batch XIC extraction for efficiency
3. Add memory pool allocation for fragment arrays

### Phase 4: Python Integration
1. Expose `QuantificationEngine` to Python via PyO3
2. Add Python bindings for `SpecLibFlatQuantified`
3. Create comprehensive Python API documentation

This roadmap ensures a systematic implementation that maintains the high-performance characteristics of the existing alphadia-search-rs codebase.