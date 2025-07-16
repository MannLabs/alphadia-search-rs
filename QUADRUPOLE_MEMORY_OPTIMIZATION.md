# QuadrupoleObservation Memory Optimization Design

## Design Overview

This document outlines an optimized builder strategy for `QuadrupoleObservation` that achieves >99.9% memory overhead reduction while providing dramatic performance improvements through better algorithms and multithreading.

**Key Innovation**: Pre-bin peaks by mz_index in single pass, then build optimal structures directly with exact pre-allocation and full parallelization.

## Problem Statement

The current `QuadrupoleObservation` structure causes substantial memory overhead due to:
- ~300 QuadrupoleObservations × ~4 million XICSlice objects = ~1.2 billion XICSlice objects
- Each XICSlice contains two Vec allocations (cycle_index: Vec<u16>, intensity: Vec<f32>)
- Vec overhead includes capacity, length, and pointer storage for each allocation
- Memory fragmentation from millions of small allocations

## Current Structure Analysis

```rust
// Current inefficient structure
pub struct QuadrupoleObservation {
    pub isolation_window: [f32; 2],
    pub num_cycles: usize,
    pub xic_slices: Vec<XICSlice>,  // ~4M objects per observation
}

pub struct XICSlice {
    pub cycle_index: Vec<u16>,     // Individual allocation + overhead
    pub intensity: Vec<f32>,       // Individual allocation + overhead
}
```

**Memory overhead per XICSlice:**
- Vec overhead: 24 bytes × 2 = 48 bytes per XICSlice
- Plus heap allocation overhead and fragmentation
- Total: ~48+ bytes overhead per XICSlice (before actual data)

## Optimized Builder Strategy

### Analysis of Current Algorithm Performance Issues

**Current Algorithm Problems:**
1. **Expensive repeated lookups**: Every peak calls `find_closest_index()` (binary search on ~3M elements)
2. **Multiple data passes**: Full spectrum scan for each delta_scan_idx (poor cache utilization)  
3. **Random memory access**: Processing by spectrum order creates poor cache locality
4. **Two-phase optimization needed**: Incremental building creates suboptimal temporary structures

**Current Data Flow:**
```
For each delta_scan_idx:                     // 300 iterations
  For each spectrum in data:                 // ~millions of spectra scanned 300x
    If spectrum.delta_scan_idx matches:
      For each peak in spectrum:             // Millions of peaks
        mz_index = find_closest_index(mz)    // Expensive binary search
        add_peak(mz_index, ...)              // Random Vec appends
```

### Core Innovation

Pre-bin all peaks by (delta_scan_idx, mz_index) in single pass, then build optimal structures directly with full parallelization.

### Optimized Structure

```rust
pub struct QuadrupoleObservation {
    pub isolation_window: [f32; 2],
    pub num_cycles: usize,
    
    // Start indices only - stop[i] = start[i+1] (last stop = total length)
    pub slice_starts: Vec<u32>,             // Length = mz_index.len() + 1
    
    // Consolidated data arrays
    pub cycle_indices: Vec<u16>,            // All cycle indices concatenated
    pub intensities: Vec<f32>,              // All intensities concatenated
}
```

### Memory Layout

```
slice_starts:     [start0, start1, start2, start3, total_length]
                     ↓       ↓       ↓       ↓
cycle_indices:    [slice0_data][slice1_data][slice2_data]...
intensities:      [slice0_data][slice1_data][slice2_data]...
```

### Implementation with Multithreading

```rust
use rayon::prelude::*;
use dashmap::DashMap;
use std::collections::HashSet;

pub struct OptimizedDIADataBuilder;

impl OptimizedDIADataBuilder {
    pub fn from_alpha_raw(alpha_raw_view: &AlphaRawView) -> DIAData {
        let mz_index = MZIndex::new();
        let rt_index = RTIndex::from_alpha_raw(alpha_raw_view);
        
        // Phase 1: Single-pass binning
        let binned_peaks = Self::bin_peaks_by_mz(alpha_raw_view, &mz_index);
        
        // Phase 2: Fully parallel observation building
        let quadrupole_observations = Self::build_observations_parallel(
            &binned_peaks, &mz_index, alpha_raw_view
        );
        
        DIAData { mz_index, rt_index, quadrupole_observations }
    }
    
    fn bin_peaks_by_mz(
        alpha_raw_view: &AlphaRawView, 
        mz_index: &MZIndex
    ) -> DashMap<(i64, usize), Vec<(u16, f32)>> {
        let binned_peaks = DashMap::new();
        
        // Single pass through all spectra - O(n) instead of O(n*m)
        for spectrum_idx in 0..alpha_raw_view.spectrum_delta_scan_idx.len() {
            let delta_scan_idx = alpha_raw_view.spectrum_delta_scan_idx[spectrum_idx];
            let cycle_idx = alpha_raw_view.spectrum_cycle_idx[spectrum_idx] as u16;
            
            let peak_start = alpha_raw_view.spectrum_peak_start_idx[spectrum_idx] as usize;
            let peak_stop = alpha_raw_view.spectrum_peak_stop_idx[spectrum_idx] as usize;
            
            // Process all peaks in this spectrum
            for peak_idx in peak_start..peak_stop {
                let mz = alpha_raw_view.peak_mz[peak_idx];
                let intensity = alpha_raw_view.peak_intensity[peak_idx];
                let mz_idx = mz_index.find_closest_index(mz);
                
                binned_peaks.entry((delta_scan_idx, mz_idx))
                           .or_insert_with(Vec::new)
                           .push((cycle_idx, intensity));
            }
        }
        
        binned_peaks
    }
    
    fn build_observations_parallel(
        binned_peaks: &DashMap<(i64, usize), Vec<(u16, f32)>>,
        mz_index: &MZIndex,
        alpha_raw_view: &AlphaRawView,
    ) -> Vec<QuadrupoleObservation> {
        let num_observations = binned_peaks.iter()
            .map(|entry| entry.key().0)
            .max().unwrap_or(0) + 1;
        
        // Extract metadata once
        let metadata = Self::extract_metadata(alpha_raw_view);
            
        // FULLY PARALLEL: Each observation is completely independent!
        (0..num_observations).into_par_iter().map(|delta_scan_idx| {
            Self::build_single_observation(binned_peaks, mz_index, delta_scan_idx, &metadata)
        }).collect()
    }
    
    fn build_single_observation(
        binned_peaks: &DashMap<(i64, usize), Vec<(u16, f32)>>,
        mz_index: &MZIndex,
        delta_scan_idx: i64,
        metadata: &HashMap<i64, (f32, f32, usize)>, // (lower_mz, upper_mz, num_cycles)
    ) -> QuadrupoleObservation {
        // Calculate exact sizes needed - no reallocation!
        let total_peaks: usize = (0..mz_index.len())
            .map(|mz_idx| {
                binned_peaks.get(&(delta_scan_idx, mz_idx))
                           .map_or(0, |entry| entry.value().len())
            })
            .sum();
        
        // Pre-allocate with exact capacity
        let mut slice_starts = Vec::with_capacity(mz_index.len() + 1);
        let mut cycle_indices = Vec::with_capacity(total_peaks);
        let mut intensities = Vec::with_capacity(total_peaks);
        
        // Build in mz_index order for optimal cache locality
        for mz_idx in 0..mz_index.len() {
            slice_starts.push(cycle_indices.len() as u32);
            
            if let Some(entry) = binned_peaks.get(&(delta_scan_idx, mz_idx)) {
                for &(cycle_idx, intensity) in entry.value() {
                    cycle_indices.push(cycle_idx);
                    intensities.push(intensity);
                }
            }
        }
        
        slice_starts.push(cycle_indices.len() as u32);
        
        let (isolation_lower, isolation_upper, num_cycles) = 
            metadata.get(&delta_scan_idx).unwrap_or(&(0.0, 0.0, 0));
        
        QuadrupoleObservation {
            isolation_window: [*isolation_lower, *isolation_upper],
            num_cycles: *num_cycles,
            slice_starts,
            cycle_indices,
            intensities,
        }
    }
    
    fn extract_metadata(alpha_raw_view: &AlphaRawView) -> HashMap<i64, (f32, f32, usize)> {
        let mut metadata = HashMap::new();
        
        for i in 0..alpha_raw_view.spectrum_delta_scan_idx.len() {
            let delta_scan_idx = alpha_raw_view.spectrum_delta_scan_idx[i];
            
            if !metadata.contains_key(&delta_scan_idx) {
                let isolation_lower = alpha_raw_view.isolation_lower_mz[i];
                let isolation_upper = alpha_raw_view.isolation_upper_mz[i];
                
                // Count unique cycles for this delta_scan_idx
                let unique_cycles: HashSet<_> = alpha_raw_view.spectrum_delta_scan_idx
                    .iter()
                    .zip(alpha_raw_view.spectrum_cycle_idx.iter())
                    .filter(|(&ds, _)| ds == delta_scan_idx)
                    .map(|(_, &cycle)| cycle)
                    .collect();
                
                metadata.insert(delta_scan_idx, (isolation_lower, isolation_upper, unique_cycles.len()));
            }
        }
        
        metadata
    }
}
```

### Key Benefits

1. **Algorithmic Improvement**: O(n*m) → O(n log k) complexity
2. **Memory Reduction**: >99.9% overhead reduction (3 allocations vs 8M+)
3. **Cache Optimization**: Sequential access patterns vs random access
4. **Exact Pre-allocation**: Zero reallocations during construction
5. **Full Parallelization**: Near-linear scaling with CPU cores

### Performance Analysis

**Before (Current Implementation):**
- O(n*m) complexity: Scan all spectra for each observation
- Millions of binary searches: find_closest_index() per peak
- Random memory access: Poor cache locality
- Millions of Vec reallocations

**After (Optimized Builder):**
- O(n log k) complexity: Single pass + binary search per peak
- Sequential memory access: Optimal cache locality per thread
- Exact pre-allocation: Zero reallocations
- Full parallelization: Each observation independent

**Expected Performance Improvements:**
- **Memory**: >99.9% overhead reduction
- **Single-threaded**: 5-10x construction speedup
- **Multi-threaded**: Near-linear scaling (30-150x total improvement)

### Multithreading Strategy

**Phase 1: Sequential Binning**
- Single pass through all spectra
- O(n) complexity vs current O(n*m)
- Use DashMap for thread-safe access

**Phase 2: Parallel Observation Building**
- Each observation completely independent
- Near-linear scaling with CPU cores
- Perfect for rayon parallel iterators

**Expected Scaling:**
- **8-core system**: 6-7x speedup for observation building
- **16-core system**: 12-15x speedup for observation building

## Implementation Plan

### Integration with Existing Code

The optimized builder requires updating `DIAData::from_arrays()` to use the new builder:

```rust
#[pymethods]
impl DIAData {
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn from_arrays<'py>(
        spectrum_delta_scan_idx: PyReadonlyArray1<'py, i64>,
        isolation_lower_mz: PyReadonlyArray1<'py, f32>,
        isolation_upper_mz: PyReadonlyArray1<'py, f32>,
        spectrum_peak_start_idx: PyReadonlyArray1<'py, i64>,
        spectrum_peak_stop_idx: PyReadonlyArray1<'py, i64>,
        spectrum_cycle_idx: PyReadonlyArray1<'py, i64>,
        spectrum_rt: PyReadonlyArray1<'py, f32>,
        peak_mz: PyReadonlyArray1<'py, f32>,
        peak_intensity: PyReadonlyArray1<'py, f32>,
        _py: Python<'py>,
    ) -> PyResult<Self> {
        let alpha_raw_view = AlphaRawView::new(
            spectrum_delta_scan_idx.as_array(),
            isolation_lower_mz.as_array(),
            isolation_upper_mz.as_array(),
            spectrum_peak_start_idx.as_array(),
            spectrum_peak_stop_idx.as_array(),
            spectrum_cycle_idx.as_array(),
            spectrum_rt.as_array(),
            peak_mz.as_array(),
            peak_intensity.as_array(),
        );

        // Use optimized builder instead of DIADataBuilder
        let dia_data = OptimizedDIADataBuilder::from_alpha_raw(&alpha_raw_view);
        Ok(dia_data)
    }
}
```

### Updated fill_xic_slice Method

```rust
impl QuadrupoleObservation {
    pub fn fill_xic_slice(
        &self,
        mz_index: &MZIndex,
        dense_xic: &mut ArrayViewMut1<f32>,
        cycle_start_idx: usize,
        cycle_stop_idx: usize,
        mass_tolerance: f32,
        mz: f32,
    ) {
        let delta_mz = mz * mass_tolerance * 1e-6;
        let lower_mz = mz - delta_mz;
        let upper_mz = mz + delta_mz;

        for mz_idx in mz_index.mz_range_indices(lower_mz, upper_mz) {
            // Direct slice access using optimized indexing
            let start = self.slice_starts[mz_idx] as usize;
            let stop = self.slice_starts[mz_idx + 1] as usize;
            
            let cycle_indices = &self.cycle_indices[start..stop];
            let intensities = &self.intensities[start..stop];

            // Binary search for start position
            let start_pos = cycle_indices
                .binary_search(&(cycle_start_idx as u16))
                .unwrap_or_else(|idx| idx);

            // Process cycles within range
            for i in start_pos..cycle_indices.len() {
                let cycle_idx = cycle_indices[i] as usize;
                
                if cycle_idx >= cycle_stop_idx {
                    break;
                }
                
                dense_xic[cycle_idx - cycle_start_idx] += intensities[i];
            }
        }
    }
    
    pub fn get_xic_slice_data(&self, mz_idx: usize) -> (&[u16], &[f32]) {
        let start = self.slice_starts[mz_idx] as usize;
        let stop = self.slice_starts[mz_idx + 1] as usize;
        
        (&self.cycle_indices[start..stop], &self.intensities[start..stop])
    }
}
```

### Updated Memory Footprint Calculation

```rust
impl DIAData {
    pub fn memory_footprint_bytes(&self) -> usize {
        let mut total_size = 0;

        // Size of MZIndex and RTIndex remain the same
        total_size += self.mz_index.mz.len() * std::mem::size_of::<f32>();
        total_size += self.rt_index.rt.len() * std::mem::size_of::<f32>();

        // Size of quadrupole_observations Vec overhead
        total_size += std::mem::size_of::<Vec<QuadrupoleObservation>>();

        // Size of each optimized QuadrupoleObservation
        for obs in &self.quadrupole_observations {
            // Fixed size components
            total_size += std::mem::size_of::<[f32; 2]>(); // isolation_window
            total_size += std::mem::size_of::<usize>(); // num_cycles
            
            // Optimized storage - only 3 Vec overheads total
            total_size += std::mem::size_of::<Vec<u32>>(); // slice_starts Vec overhead
            total_size += std::mem::size_of::<Vec<u16>>(); // cycle_indices Vec overhead  
            total_size += std::mem::size_of::<Vec<f32>>(); // intensities Vec overhead
            
            // Actual data in the optimized arrays
            total_size += obs.slice_starts.len() * std::mem::size_of::<u32>();
            total_size += obs.cycle_indices.len() * std::mem::size_of::<u16>();
            total_size += obs.intensities.len() * std::mem::size_of::<f32>();
        }

        total_size
    }
}
```

## Memory Impact Analysis

### Before Optimization
- **Per QuadrupoleObservation**: ~4M XICSlice objects × 48+ bytes overhead = ~192+ MB overhead
- **Total for 300 observations**: ~57.6+ GB overhead

### After Optimization  
- **Per QuadrupoleObservation**: 3 Vec allocations × 8 bytes = 24 bytes overhead
- **Total for 300 observations**: ~7.2 KB overhead

### **Expected Memory Reduction: >99.9%**

## Implementation Checklist

### Core Implementation
- [ ] Implement `OptimizedDIADataBuilder::bin_peaks_by_mz()` (sequential version)
- [ ] Implement `OptimizedDIADataBuilder::bin_peaks_parallel()` with DashMap
- [ ] Implement metadata extraction for isolation_window and num_cycles
- [ ] Implement `build_observations_parallel()` with rayon parallel iterator
- [ ] Implement `build_single_observation_parallel()` with exact pre-allocation
- [ ] Update QuadrupoleObservation to work with direct construction
- [ ] Create XICSliceView compatibility layer

### Performance & Memory
- [ ] Verify slice_starts bounds checking for mz_idx + 1 access
- [ ] Benchmark construction speed improvement vs current implementation
- [ ] Benchmark single-threaded vs parallel binning performance
- [ ] Benchmark parallel observation building scaling (2, 4, 8, 16 cores)
- [ ] Benchmark memory usage improvements (>99.9% reduction)
- [ ] Profile cache miss rates to confirm improved locality
- [ ] Measure memory bandwidth utilization improvement

### Testing & Validation
- [ ] Add unit tests for binning algorithm
- [ ] Add integration tests comparing both builders' outputs
- [ ] Validate identical results between old and new builders
- [ ] Test with real data files using scripts/test_search.py
- [ ] Ensure no breaking changes to existing Python API

### Production Readiness
- [ ] Handle edge cases (empty observations, missing data)
- [ ] Add comprehensive error handling
- [ ] Performance benchmarking and documentation
- [ ] Migration documentation and examples

## API Compatibility

### Python API Compatibility ✅
- `DIAData.from_arrays()` continues to work (just different internal builder)
- `memory_footprint_bytes()` and `memory_footprint_mb()` remain available
- `get_valid_observations()` functionality preserved
- No breaking changes to PyO3 interface

### Migration Strategy
1. **Implement alongside existing**: Create `OptimizedDIADataBuilder` without touching current code
2. **Feature flag**: Allow switching between builders for comparison
3. **Validation**: Ensure identical outputs between both approaches
4. **Gradual rollout**: Default to optimized builder once validated

## Testing Strategy

1. **Unit Tests**: 
   - Test binning algorithm correctness
   - Verify metadata extraction (isolation_window, num_cycles)
   - Test optimized QuadrupoleObservation structure
2. **Memory Tests**: 
   - Confirm >99.9% memory reduction
   - Benchmark memory usage vs current implementation
3. **Performance Tests**: 
   - Benchmark single-threaded vs multithreaded performance
   - Measure construction speed improvement
   - Profile cache miss rates and memory bandwidth
4. **Integration Tests**: 
   - Validate identical outputs vs current DIADataBuilder
   - Test with real data files using scripts/test_search.py
   - Ensure Python API compatibility

## Risks and Mitigation

- **Risk**: Implementation complexity
  - **Mitigation**: Phased implementation with comprehensive testing
- **Risk**: DashMap memory overhead during binning
  - **Mitigation**: Profile and optimize; consider streaming approach if needed
- **Risk**: Index bounds errors in slice_starts access
  - **Mitigation**: Comprehensive bounds checking and testing
- **Risk**: Performance regression in edge cases
  - **Mitigation**: Extensive benchmarking with real-world data 