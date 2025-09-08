import pytest
import sys
import os

# Add the parent directory to the path to import from core_services
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core_services.math_helpers import dot


class TestDot:
    """Test cases for the dot product function."""
    
    def test_dot_identical_vectors(self):
        """Test dot product of identical vectors."""
        a = [1.0, 2.0, 3.0]
        b = [1.0, 2.0, 3.0]
        result = dot(a, b)
        expected = 1.0 + 4.0 + 9.0  # 14.0
        assert result == expected
    
    def test_dot_orthogonal_vectors(self):
        """Test dot product of orthogonal vectors (should be 0)."""
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        result = dot(a, b)
        assert result == 0.0
    
    def test_dot_opposite_vectors(self):
        """Test dot product of opposite vectors."""
        a = [1.0, 2.0, 3.0]
        b = [-1.0, -2.0, -3.0]
        result = dot(a, b)
        expected = -1.0 - 4.0 - 9.0  # -14.0
        assert result == expected
    
    def test_dot_mixed_signs(self):
        """Test dot product with mixed positive and negative values."""
        a = [1.0, -2.0, 3.0]
        b = [4.0, 5.0, -6.0]
        result = dot(a, b)
        expected = 4.0 - 10.0 - 18.0  # -24.0
        assert result == expected
    
    def test_dot_zero_vector(self):
        """Test dot product with zero vector."""
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        result = dot(a, b)
        assert result == 0.0
    
    def test_dot_single_element(self):
        """Test dot product of single-element vectors."""
        a = [5.0]
        b = [3.0]
        result = dot(a, b)
        assert result == 15.0
    
    def test_dot_large_vectors(self):
        """Test dot product of larger vectors."""
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [2.0, 3.0, 4.0, 5.0, 6.0]
        result = dot(a, b)
        expected = 2.0 + 6.0 + 12.0 + 20.0 + 30.0  # 70.0
        assert result == expected
    
    def test_dot_float_precision(self):
        """Test dot product with floating point precision."""
        a = [0.1, 0.2, 0.3]
        b = [0.4, 0.5, 0.6]
        result = dot(a, b)
        expected = 0.04 + 0.10 + 0.18  # 0.32
        assert abs(result - expected) < 1e-10
    
    def test_dot_empty_vectors(self):
        """Test dot product of empty vectors."""
        a = []
        b = []
        result = dot(a, b)
        assert result == 0.0
    
    def test_dot_different_lengths_uses_shorter(self):
        """Test that dot product uses the shorter vector length."""
        a = [1.0, 2.0]
        b = [1.0, 2.0, 3.0]
        result = dot(a, b)
        # Should only use first 2 elements: 1*1 + 2*2 = 5
        expected = 5.0
        assert result == expected
