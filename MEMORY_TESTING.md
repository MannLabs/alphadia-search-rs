# Memory Footprint Testing Guide

This guide explains how to run, test, and analyze memory usage in the alphadia-ng package.

## Prerequisites

```bash
# Activate the conda environment
conda activate alphadia-ng

```

## Building the Package

### Release Build (Optimized)
```bash
maturin develop --release
```

**Note:** Always use release builds for memory footprint analysis as they provide optimized performance and more accurate memory usage patterns.

## Running Tests

### Unit Tests
```bash
# Run Rust unit tests
cargo test
```

### Integration Tests with Memory Monitoring
```bash
# Run the main integration test with memory logging
python ./scripts/test_search.py
```

This will output:
- DIAData memory footprint in MB and bytes
- Number of quadrupole observations
- Processing performance metrics
