"""Test module for alpha_rs functions using pytest."""

import numpy as np
import pytest
import sys
import os

# Add the parent directory to sys.path so we can import the built module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

try:
    import alpha_rs
except ImportError:
    pytest.skip("Failed to import alpha_rs module. Make sure it's built with 'maturin develop'", allow_module_level=True)


def test_sum_as_string():
    """Test the sum_as_string function."""
    result = alpha_rs.sum_as_string(5, 7)
    assert result == "12"


def test_sum_array():
    """Test the sum_array function."""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = alpha_rs.sum_array(arr)
    assert result == "15"


def test_raw_class():
    """Test the Raw class."""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    raw = alpha_rs.Raw(arr)
    result = raw.sum()
    assert result == 15.0


def test_sum_container_class():
    """Test the SumContainer class."""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    container = alpha_rs.SumContainer(arr)
    result = container.sum_array()
    assert result == 15.0
