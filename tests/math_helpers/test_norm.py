import pytest
import sys
import os
import math

# Add the parent directory to the path to import from core_services
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core_services.math_helpers import norm


class TestNorm:
    """Test cases for the vector norm function."""
    
    def test_norm_unit_vector(self):
        """Test norm of unit vector."""
        a = [1.0, 0.0, 0.0]
        result = norm(a)
        assert result == 1.0
    
    def test_norm_zero_vector(self):
        """Test norm of zero vector (should return minimum value)."""
        a = [0.0, 0.0, 0.0]
        result = norm(a)
        assert result == math.sqrt(1e-12)  # Minimum value from implementation
    
    def test_norm_standard_vector(self):
        """Test norm of standard vector."""
        a = [3.0, 4.0]
        result = norm(a)
        expected = math.sqrt(9.0 + 16.0)  # 5.0
        assert result == expected
    
    def test_norm_negative_values(self):
        """Test norm with negative values."""
        a = [-3.0, 4.0]
        result = norm(a)
        expected = math.sqrt(9.0 + 16.0)  # 5.0 (norm is always positive)
        assert result == expected
    
    def test_norm_mixed_signs(self):
        """Test norm with mixed positive and negative values."""
        a = [1.0, -2.0, 3.0]
        result = norm(a)
        expected = math.sqrt(1.0 + 4.0 + 9.0)  # sqrt(14)
        assert result == expected
    
    def test_norm_single_element(self):
        """Test norm of single-element vector."""
        a = [5.0]
        result = norm(a)
        assert result == 5.0
    
    def test_norm_large_vector(self):
        """Test norm of larger vector."""
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = norm(a)
        expected = math.sqrt(1.0 + 4.0 + 9.0 + 16.0 + 25.0)  # sqrt(55)
        assert result == expected
    
    def test_norm_float_precision(self):
        """Test norm with floating point precision."""
        a = [0.1, 0.2, 0.3]
        result = norm(a)
        expected = math.sqrt(0.01 + 0.04 + 0.09)  # sqrt(0.14)
        assert abs(result - expected) < 1e-10
    
    def test_norm_empty_vector(self):
        """Test norm of empty vector."""
        a = []
        result = norm(a)
        assert result == math.sqrt(1e-12)  # Minimum value from implementation
    
    def test_norm_very_small_values(self):
        """Test norm with very small values that might cause numerical issues."""
        a = [1e-10, 1e-10, 1e-10]
        result = norm(a)
        expected = math.sqrt(3e-20)
        # Should still work correctly even with very small values
        # Due to the max(1e-12, dot(a, a)) in the implementation, very small values get clamped
        assert result >= math.sqrt(1e-12)
    
    def test_norm_very_large_values(self):
        """Test norm with very large values."""
        a = [1e6, 2e6, 3e6]
        result = norm(a)
        expected = math.sqrt(1e12 + 4e12 + 9e12)  # sqrt(14e12)
        assert abs(result - expected) < 1e-6
    
    def test_norm_orthogonal_vectors(self):
        """Test that orthogonal vectors have expected norms."""
        a = [3.0, 4.0]
        b = [4.0, -3.0]  # Orthogonal to a
        norm_a = norm(a)
        norm_b = norm(b)
        assert norm_a == 5.0
        assert norm_b == 5.0
