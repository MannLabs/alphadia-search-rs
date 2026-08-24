# alphadia-search-rs

High-performance alphaDIA backend.

## Notes for users
This repository contains the high-performance backend for alphaDIA. This
code should to used as part of [alphaDIA](https://github.com/MannLabs/alphadia).

### Controlling Thread Count

Control the number of threads used for parallel operations:

**Option 1: Direct call (must be first call after import)**
```python
import alphadia_search_rs
alphadia_search_rs.set_num_threads(4)
```

**Option 2: Environment variable (recommended)**
> ℹ️ **Note:** The environment variable `RAYON_NUM_THREADS` must be set *before* starting Python, or at least before importing `alphadia_search_rs`. Setting it at runtime will not affect the thread pool.

```bash
export RAYON_NUM_THREADS=4
python your_script.py
```

## Development Setup

---

**📌 Versioning Policy**

This repository strictly follows [**Semantic Versioning 2.0.0**](https://semver.org/). Given a version number `MAJOR.MINOR.PATCH`:
- **MAJOR** version for incompatible API changes
- **MINOR** version for backwards-compatible functionality additions
- **PATCH** version for backwards-compatible bug fixes

---

### Prerequisites

- **Rust 1.88.0**
- **Python 3.11+**

### Quick Start

1. **Clone and enter the repository:**
   ```bash
   git clone <repository-url>
   cd alphadia-search-rs
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
   conda activate alphadia-search-rs  # or create environment if it doesn't exist
   pip install maturin
   ```

4. **Build the Rust extension:**
   ```bash
   maturin develop --release
   ```
Omit the `--release` extension for a developer build.

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
python ./scripts/test_search.py --path ./test_data
```

The script will automatically:
1. Use existing test data in `./test_data` if available (using a temporary directory if `--path` not specified).
2. Otherwise download required files:
   - `spectrum_df.parquet` - Mass spectrometry spectra data
   - `peak_df.parquet` - Peak detection results
   - `precursor_df.parquet` - Precursor ion information
   - `fragment_df.parquet` - Fragment ion data

**Expected output:**
- Processing speed: ~200k+ precursors per second
- Results: ~11M candidates found

## Scripts

The `scripts/` directory contains analysis pipelines for processing DIA-MS data:


### Key Scripts


1. **Candidate Selection**: Takes a calibrated speclib as an input and an AlphaRaw hdf. Performs candidate selection and saves the candidates.
   ```bash
   python scripts/candidate_selection.py --ms_data_path data.hdf --spec_lib_path lib.hdf --output_folder ./output
   ```

2. **Candidate Scoring**: Performs scoring following selection. Takes input from previous step and save precursor at 1% FDR.
   ```bash
   python scripts/candidate_scoring.py --ms_data_path data.hdf --spec_lib_path lib.hdf --candidates_path candidates.parquet --fdr --quantify
   ```
   - option to perform quantification with `--quantify`
   - option to perform FDR adn filter @1% with `--fdr`
   - option to add diagnosis plot for all features with `--diagnosis`


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

Mac:
```bash
export DYLD_LIBRARY_PATH=$(realpath $(which python)/../../lib)
cargo test
```

Linux:
```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
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
