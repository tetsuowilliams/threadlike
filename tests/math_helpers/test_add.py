import pytest
import sys
import os

# Add the parent directory to the path to import from core_services
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core_services.math_helpers import add


class TestAdd:
    """Test cases for the vector addition function."""
    
    def test_add_positive_vectors(self):
        """Test addition of positive vectors."""
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        result = add(a, b)
        expected = [5.0, 7.0, 9.0]
        assert result == expected
    
    def test_add_mixed_signs(self):
        """Test addition of vectors with mixed positive and negative values."""
        a = [1.0, -2.0, 3.0]
        b = [-4.0, 5.0, -6.0]
        result = add(a, b)
        expected = [-3.0, 3.0, -3.0]
        assert result == expected
    
    def test_add_zero_vector(self):
        """Test addition with zero vector."""
        a = [1.0, 2.0, 3.0]
        b = [0.0, 0.0, 0.0]
        result = add(a, b)
        expected = [1.0, 2.0, 3.0]
        assert result == expected
    
    def test_add_negative_vectors(self):
        """Test addition of negative vectors."""
        a = [-1.0, -2.0, -3.0]
        b = [-4.0, -5.0, -6.0]
        result = add(a, b)
        expected = [-5.0, -7.0, -9.0]
        assert result == expected
    
    def test_add_commutative(self):
        """Test that addition is commutative (a + b = b + a)."""
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        result1 = add(a, b)
        result2 = add(b, a)
        assert result1 == result2
    
    def test_add_associative(self):
        """Test that addition is associative ((a + b) + c = a + (b + c))."""
        a = [1.0, 2.0]
        b = [3.0, 4.0]
        c = [5.0, 6.0]
        result1 = add(add(a, b), c)
        result2 = add(a, add(b, c))
        assert result1 == result2
    
    def test_add_single_element(self):
        """Test addition of single-element vectors."""
        a = [5.0]
        b = [3.0]
        result = add(a, b)
        expected = [8.0]
        assert result == expected
    
    def test_add_large_vectors(self):
        """Test addition of larger vectors."""
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [6.0, 7.0, 8.0, 9.0, 10.0]
        result = add(a, b)
        expected = [7.0, 9.0, 11.0, 13.0, 15.0]
        assert result == expected
    
    def test_add_float_precision(self):
        """Test addition with floating point precision."""
        a = [0.1, 0.2, 0.3]
        b = [0.4, 0.5, 0.6]
        result = add(a, b)
        expected = [0.5, 0.7, 0.9]
        assert all(abs(r - e) < 1e-10 for r, e in zip(result, expected))
    
    def test_add_empty_vectors(self):
        """Test addition of empty vectors."""
        a = []
        b = []
        result = add(a, b)
        expected = []
        assert result == expected
    
    def test_add_very_small_values(self):
        """Test addition with very small values."""
        a = [1e-10, 2e-10, 3e-10]
        b = [4e-10, 5e-10, 6e-10]
        result = add(a, b)
        expected = [5e-10, 7e-10, 9e-10]
        assert all(abs(r - e) < 1e-15 for r, e in zip(result, expected))
    
    def test_add_very_large_values(self):
        """Test addition with very large values."""
        a = [1e6, 2e6, 3e6]
        b = [4e6, 5e6, 6e6]
        result = add(a, b)
        expected = [5e6, 7e6, 9e6]
        assert all(abs(r - e) < 1e-6 for r, e in zip(result, expected))
    
    def test_add_identity_element(self):
        """Test addition with identity element (zero vector)."""
        a = [1.0, 2.0, 3.0]
        zero = [0.0, 0.0, 0.0]
        result = add(a, zero)
        assert result == a
    
    def test_add_does_not_modify_input(self):
        """Test that addition does not modify input vectors."""
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        a_original = a[:]
        b_original = b[:]
        add(a, b)
        assert a == a_original
        assert b == b_original
    
    def test_add_different_lengths_uses_shorter(self):
        """Test that addition uses the shorter vector length."""
        a = [1.0, 2.0]
        b = [1.0, 2.0, 3.0]
        result = add(a, b)
        # Should only use first 2 elements: [1+1, 2+2] = [2, 4]
        expected = [2.0, 4.0]
        assert result == expected
    
    def test_add_returns_new_list(self):
        """Test that addition returns a new list, not a reference to input."""
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        result = add(a, b)
        # Modify result and ensure inputs are unchanged
        result[0] = 999.0
        assert a[0] == 1.0
        assert b[0] == 4.0
