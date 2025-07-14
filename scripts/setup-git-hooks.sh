#!/bin/bash
# Setup script for alpha-rs git hooks using pre-commit

set -e

echo "🔧 Setting up pre-commit hooks for alpha-rs..."

# Check if we're in the project root
if [ ! -f "Cargo.toml" ] || [ ! -f ".pre-commit-config.yaml" ]; then
    echo "❌ Please run this script from the alpha-rs project root"
    exit 1
fi

# Check if pre-commit is installed
if ! command -v pre-commit &> /dev/null; then
    echo "📦 Installing pre-commit..."
    
    # Try different installation methods
    if command -v pip &> /dev/null; then
        pip install pre-commit
    elif command -v conda &> /dev/null; then
        conda install -c conda-forge pre-commit
    elif command -v brew &> /dev/null; then
        brew install pre-commit
    else
        echo "❌ Could not find pip, conda, or brew to install pre-commit"
        echo "💡 Please install pre-commit manually: https://pre-commit.com/#install"
        exit 1
    fi
fi

# Install the git hook scripts
echo "🔗 Installing pre-commit hooks..."
pre-commit install

# Run hooks on all files to verify setup
echo "🧪 Testing hooks on all files..."
if pre-commit run --all-files; then
    echo "✅ Pre-commit hooks configured successfully!"
else
    echo "⚠️  Some hooks failed - this is normal if code needs formatting"
    echo "💡 Run 'cargo fmt' to fix formatting, then commit again"
fi

echo ""
echo "📋 Installed hooks:"
echo "  - cargo fmt: Rust code formatting"
echo "  - cargo clippy: Rust linting"
echo "  - trailing-whitespace: Remove trailing whitespace"
echo "  - end-of-file-fixer: Ensure files end with newline"
echo "  - check-yaml: Validate YAML files"
echo "  - check-toml: Validate TOML files"
echo "  - check-merge-conflict: Detect merge conflict markers"
echo "  - check-added-large-files: Prevent large files"
echo ""
echo "💡 To bypass hooks for a single commit (emergency only):"
echo "   git commit --no-verify -m \"your message\""
echo ""
echo "💡 To update hooks to latest versions:"
echo "   pre-commit autoupdate" 