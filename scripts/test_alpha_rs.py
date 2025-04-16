"""Test script for alpha_rs module functions.

This script tests the main functionality of the alpha_rs Rust module.
"""

import numpy as np
import sys
import os

# Add the parent directory to sys.path so we can import the built module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

try:
    import alpha_rs
except ImportError:
    print("Failed to import alpha_rs module. Make sure it's built with 'maturin develop'")
    sys.exit(1)


def test_sum_as_string():
    """Test the sum_as_string function."""
    result = alpha_rs.sum_as_string(5, 7)
    expected = "12"
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ sum_as_string test passed")


def test_sum_array():
    """Test the sum_array function."""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = alpha_rs.sum_array(arr)
    expected = "15"
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ sum_array test passed")


def test_raw_class():
    """Test the Raw class."""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    raw = alpha_rs.Raw(arr)
    result = raw.sum()
    expected = 15.0
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ Raw class test passed")


def main():
    """Run all tests."""
    print("Running alpha_rs tests...")
    test_sum_as_string()
    test_sum_array()
    test_raw_class()
    print("All tests passed!")


if __name__ == "__main__":
    main()