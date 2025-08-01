# AlphaDIA-NG

High-performance alphaDIA backend

## Development Setup

### Prerequisites

- **Rust 1.88.0**
- **Python 3.11+**

### Quick Start

1. **Clone and enter the repository:**
   ```bash
   git clone <repository-url>
   cd alphadia-ng
   ```

2. **Set up pre-commit hooks (recommended):**
   ```bash
   # Install pre-commit
   pip install pre-commit
   # or: conda install -c conda-forge pre-commit
   # or: brew install pre-commit

   # Install the git hook scripts
   pre-commit install
   ```

3. **Install Python dependencies:**
   ```bash
   conda activate alphadia-ng  # or create environment if it doesn't exist
   pip install maturin
   ```

4. **Build the Rust extension:**
   ```bash
   maturin develop
   ```

5. **Run tests:**
   ```bash
   cargo test                    # Rust tests
   python ./scripts/test_search.py  # Python integration test
   ```

## Testing

### Integration Test

The `scripts/test_search.py` script provides a comprehensive integration test.
```bash
# Run the integration test
python ./scripts/test_search.py
```

The script will automatically:
1. Use existing test data in `/Users/georgwallmann/Documents/data/alphadia-ng` if available
2. Otherwise create a temporary directory and download required files:
   - `spectrum_df.parquet` - Mass spectrometry spectra data
   - `peak_df.parquet` - Peak detection results
   - `precursor_df.parquet` - Precursor ion information
   - `fragment_df.parquet` - Fragment ion data

**Expected output:**
- Processing speed: ~200k+ precursors per second
- Results: ~11M candidates found

## CLI Benchmarking

### Score Benchmark Tool

The `score-benchmark` CLI tool benchmarks multiple implementations of `axis_log_dot_product` to compare performance and verify numerical accuracy.

```bash
# Run the benchmark
cargo run --bin score-benchmark
```

### Troubleshooting

**Library Loading Error on macOS:**
If you encounter the error `dyld[xxxxx]: Library not loaded: @rpath/libpython3.11.dylib` when running `cargo test`, set the library path:

```bash
export DYLD_LIBRARY_PATH=$(realpath $(which python)/../../lib)
cargo test
```

## Development Workflow

### Code Quality Standards

This project enforces strict code quality standards via automated tooling:

- **Formatting**: All code must be formatted with `rustfmt`
- **Linting**: All code must pass `clippy` with no warnings
- **Consistency**: Same toolchain used locally and in CI (Rust 1.88.0)

### Pre-Commit Hooks

We use the [pre-commit](https://pre-commit.com/) framework for automated code quality checks:

```bash
# Install pre-commit (one-time setup)
pip install pre-commit

# Install hooks (one-time setup)
pre-commit install
```

### Manual Code Quality Checks

You can run the same checks manually:

```bash
# Format code
cargo fmt

# Check formatting (without modifying files)
cargo fmt --all -- --check

# Run linting
cargo clippy -- -D warnings

# Run all pre-commit hooks manually
pre-commit run --all-files
```
