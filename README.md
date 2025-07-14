# Alpha-RS

High-performance mass spectrometry data analysis library with SIMD optimizations for aarch64 and x86_64 architectures.

## Features

- **SIMD-optimized algorithms** for convolution and scoring operations
- **Cross-platform support** for Linux (x86_64, aarch64), Windows (x86_64), and macOS (aarch64)
- **Runtime backend selection** automatically chooses the best available SIMD implementation
- **Python bindings** via PyO3/maturin for easy integration

## Development Setup

### Prerequisites

- **Rust 1.88.0** (automatically managed via `rust-toolchain.toml`)
- **Python 3.8+** (for testing Python bindings)
- **conda** (recommended for managing Python dependencies)

### Quick Start

1. **Clone and enter the repository:**
   ```bash
   git clone <repository-url>
   cd alpha-rs
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
   conda activate alpha-rs  # or create environment if it doesn't exist
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

# Test hooks on all files
pre-commit run --all-files

# Hooks will automatically run on every commit
git commit -m "your changes"

# To bypass hooks in emergencies only:
git commit --no-verify -m "emergency fix"

# Update hooks to latest versions:
pre-commit autoupdate
```

#### Available Hooks

- **cargo fmt**: Rust code formatting
- **cargo clippy**: Rust linting with strict warnings
- **trailing-whitespace**: Remove trailing whitespace
- **end-of-file-fixer**: Ensure files end with newline
- **check-yaml**: Validate YAML files
- **check-toml**: Validate TOML files  
- **check-merge-conflict**: Detect merge conflict markers
- **check-added-large-files**: Prevent large files (>1MB)

### Manual Code Quality Checks

You can run the same checks manually:

```bash
# Format code
cargo fmt

# Check formatting (without modifying files)
cargo fmt --all -- --check

# Run linting
cargo clippy -- -D warnings

# Run both checks (same as CI)
cargo clippy -- -D warnings && cargo fmt --all -- --check

# Run all pre-commit hooks manually
pre-commit run --all-files
```

### Working with SIMD Code

This project includes SIMD optimizations for performance-critical operations:

- **NEON** (aarch64): Optimized for Apple Silicon and ARM64 servers
- **Scalar fallback**: Works on all architectures
- **Future**: AVX2/AVX-512 support planned for x86_64

The SIMD backend is selected automatically at runtime based on CPU capabilities.

#### Testing SIMD Implementations

```bash
# Run benchmarks to compare implementations
cargo run --release --bin benchmark

# Test specific backend (for debugging)
ALPHA_RS_BACKEND=scalar cargo test
ALPHA_RS_BACKEND=neon cargo test    # aarch64 only
```

## Project Structure

```
alpha-rs/
├── src/
│   ├── simd/              # SIMD backend system
│   ├── convolution/       # Convolution algorithms (NEON + scalar)
│   ├── score/             # Scoring functions (NEON + scalar)
│   ├── kernel/            # Gaussian kernel operations
│   └── ...                # Other modules
├── scripts/               # Development and test scripts
├── .pre-commit-config.yaml # Pre-commit hooks configuration
├── rust-toolchain.toml    # Pinned Rust version
├── clippy.toml            # Clippy configuration
└── pyproject.toml         # Python packaging
```

## Performance

The library includes extensive benchmarking and optimization:

- **Convolution operations**: ~2-4x speedup with NEON on aarch64
- **Scoring functions**: Optimized for large arrays with SIMD
- **Memory usage**: Minimized allocations in hot paths

Run benchmarks:
```bash
cargo run --release --bin benchmark
```

## Platform Support

| Platform | Architecture | Status | SIMD Support |
|----------|-------------|---------|--------------|
| Linux    | x86_64      | ✅      | Scalar (AVX2 planned) |
| Linux    | aarch64     | ✅      | NEON |
| Windows  | x86_64      | ✅      | Scalar (AVX2 planned) |
| macOS    | aarch64     | ✅      | NEON |

## Contributing

1. **Fork and clone** the repository
2. **Set up development environment** (see Development Setup)
3. **Install pre-commit hooks**:
   ```bash
   pip install pre-commit
   pre-commit install
   ```
4. **Make your changes** with tests
5. **Ensure all checks pass**: `pre-commit run --all-files`
6. **Submit a pull request**

### Code Style

- Follow standard Rust formatting (`cargo fmt`)
- Address all clippy warnings
- Add tests for new functionality
- Document public APIs
- Use descriptive commit messages

## CI/CD

The project uses GitHub Actions for continuous integration:

- **Linting**: `clippy` with strict warnings
- **Formatting**: `rustfmt` consistency checks
- **Testing**: Full test suite on multiple platforms
- **Consistent toolchain**: Same Rust version (1.88.0) everywhere

## License

[Add your license information here]

## Acknowledgments

- Built with [PyO3](https://pyo3.rs/) for Python bindings
- Uses [maturin](https://maturin.rs/) for packaging
- SIMD optimizations inspired by scientific computing best practices
- Code quality maintained with [pre-commit](https://pre-commit.com/) 