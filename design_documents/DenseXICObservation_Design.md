# DenseXICObservation Design Document

## Overview

This document outlines the design for a `DenseXICObservation` struct to encapsulate the common pattern of creating dense XIC (Extracted Ion Chromatogram) matrices with associated observation indices found throughout the alphadia-search-rs codebase.

## Current Pattern Analysis

### Identified Locations

The pattern of creating `dense_xic` matrices with associated `obs_idx` collections appears in multiple locations:

1. **Peak Group Scoring** (`src/peak_group_scoring/mod.rs:120-139`)
2. **Peak Group Selection** (`src/peak_group_selection/mod.rs:157-173`)
3. **Integration Tests** (`src/integration_tests.rs:38-92`)

### Common Pattern

```rust
// Current repeated pattern:
let mut dense_xic: Array2<f32> =
    Array2::zeros((filtered_fragment_mz.len(), cycle_stop_idx - cycle_start_idx));

let valid_obs_idxs = dia_data.get_valid_observations(precursor.mz);

for &obs_idx in &valid_obs_idxs {
    let obs = &dia_data.quadrupole_observations()[obs_idx];

    for (f_idx, f_mz) in filtered_fragment_mz.iter().enumerate() {
        obs.fill_xic_slice(
            dia_data.mz_index(),
            &mut dense_xic.row_mut(f_idx),
            cycle_start_idx,
            cycle_stop_idx,
            mass_tolerance,
            *f_mz,
        );
    }
}
```

## Proposed Design

### Core Struct with Constructor Pattern

```rust
use numpy::ndarray::Array2;
use crate::traits::DIADataTrait;

/// Encapsulates a dense XIC matrix with metadata about its construction
pub struct DenseXICObservation {
    /// Dense XIC matrix: [fragment_index, cycle_index] -> intensity
    pub dense_xic: Array2<f32>,

    /// Indices of observations that contributed to this dense XIC
    pub contributing_obs_indices: Vec<usize>,

    /// Cycle range metadata
    pub cycle_start_idx: usize,
    pub cycle_stop_idx: usize,

    /// Mass tolerance used for extraction
    pub mass_tolerance: f32,
}

impl DenseXICObservation {
    /// Create a new DenseXICObservation from DIA data and parameters
    ///
    /// This constructor pattern allows for zero-cost abstractions and full
    /// compiler optimization through monomorphization.
    #[inline]
    pub fn new<T: DIADataTrait>(
        dia_data: &T,
        precursor_mz: f32,
        cycle_start_idx: usize,
        cycle_stop_idx: usize,
        mass_tolerance: f32,
        fragment_mz: &[f32],  // Use slice for better performance
    ) -> Self {
        let mut dense_xic = Array2::zeros((
            fragment_mz.len(),
            cycle_stop_idx - cycle_start_idx
        ));

        let valid_obs_idxs = dia_data.get_valid_observations(precursor_mz);

        for &obs_idx in &valid_obs_idxs {
            let obs = &dia_data.quadrupole_observations()[obs_idx];

            for (f_idx, &f_mz) in fragment_mz.iter().enumerate() {
                obs.fill_xic_slice(
                    dia_data.mz_index(),
                    &mut dense_xic.row_mut(f_idx),
                    cycle_start_idx,
                    cycle_stop_idx,
                    mass_tolerance,
                    f_mz,
                );
            }
        }

        Self {
            dense_xic,
            contributing_obs_indices: valid_obs_idxs,
            cycle_start_idx,
            cycle_stop_idx,
            mass_tolerance,
        }
    }
}
```

## Usage Examples

### Peak Group Scoring (Refactored)

```rust
// Before:
let mut dense_xic: Array2<f32> =
    Array2::zeros((filtered_fragment_mz.len(), cycle_stop_idx - cycle_start_idx));
let valid_obs_idxs = dia_data.get_valid_observations(precursor.mz);
// ... repetitive filling logic ...

// After (High-Performance Constructor Pattern):
let dense_xic_obs = DenseXICObservation::new(
    &dia_data,
    precursor.mz,
    cycle_start_idx,
    cycle_stop_idx,
    mass_tolerance,
    &filtered_fragment_mz,  // Use slice reference for performance
);

let normalized_xic = normalize_profiles(&dense_xic_obs.dense_xic, 1);
let observation_intensities = dense_xic_obs.dense_xic.sum_axis(Axis(1));
```

### Peak Group Selection (Refactored)

```rust
// Before:
let mut dense_xic: Array2<f32> =
    Array2::zeros((filtered_fragment_mz.len(), cycle_stop_idx - cycle_start_idx));
// ... repetitive filling logic ...
let convolved_xic = convolution(&self.kernel, &dense_xic);

// After (High-Performance Constructor Pattern):
let dense_xic_obs = DenseXICObservation::new(
    &dia_data,
    precursor.mz,
    cycle_start_idx,
    cycle_stop_idx,
    mass_tolerance,
    &filtered_fragment_mz,  // Use slice reference for performance
);

let convolved_xic = convolution(&self.kernel, &dense_xic_obs.dense_xic);
```

## File Structure

```
src/
├── dense_xic_observation/
│   ├── mod.rs                 # Main struct and implementation
│   └── tests.rs               # Unit tests
└── traits.rs                  # Updated DIADataTrait
```