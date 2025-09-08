import pytest
import sys
import os

# Add the parent directory to the path to import from core_services
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core_services.math_helpers import scale


class TestScale:
    """Test cases for the vector scaling function."""
    
    def test_scale_positive_scalar(self):
        """Test scaling with positive scalar."""
        a = [1.0, 2.0, 3.0]
        s = 2.0
        result = scale(a, s)
        expected = [2.0, 4.0, 6.0]
        assert result == expected
    
    def test_scale_negative_scalar(self):
        """Test scaling with negative scalar."""
        a = [1.0, 2.0, 3.0]
        s = -2.0
        result = scale(a, s)
        expected = [-2.0, -4.0, -6.0]
        assert result == expected
    
    def test_scale_zero_scalar(self):
        """Test scaling with zero scalar."""
        a = [1.0, 2.0, 3.0]
        s = 0.0
        result = scale(a, s)
        expected = [0.0, 0.0, 0.0]
        assert result == expected
    
    def test_scale_identity_scalar(self):
        """Test scaling with identity scalar (1.0)."""
        a = [1.0, 2.0, 3.0]
        s = 1.0
        result = scale(a, s)
        expected = [1.0, 2.0, 3.0]
        assert result == expected
    
    def test_scale_fractional_scalar(self):
        """Test scaling with fractional scalar."""
        a = [2.0, 4.0, 6.0]
        s = 0.5
        result = scale(a, s)
        expected = [1.0, 2.0, 3.0]
        assert result == expected
    
    def test_scale_negative_vector(self):
        """Test scaling of negative vector."""
        a = [-1.0, -2.0, -3.0]
        s = 2.0
        result = scale(a, s)
        expected = [-2.0, -4.0, -6.0]
        assert result == expected
    
    def test_scale_mixed_signs(self):
        """Test scaling of vector with mixed signs."""
        a = [1.0, -2.0, 3.0]
        s = 3.0
        result = scale(a, s)
        expected = [3.0, -6.0, 9.0]
        assert result == expected
    
    def test_scale_single_element(self):
        """Test scaling of single-element vector."""
        a = [5.0]
        s = 2.0
        result = scale(a, s)
        expected = [10.0]
        assert result == expected
    
    def test_scale_large_vector(self):
        """Test scaling of larger vector."""
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        s = 1.5
        result = scale(a, s)
        expected = [1.5, 3.0, 4.5, 6.0, 7.5]
        assert result == expected
    
    def test_scale_float_precision(self):
        """Test scaling with floating point precision."""
        a = [0.1, 0.2, 0.3]
        s = 0.1
        result = scale(a, s)
        expected = [0.01, 0.02, 0.03]
        assert all(abs(r - e) < 1e-10 for r, e in zip(result, expected))
    
    def test_scale_empty_vector(self):
        """Test scaling of empty vector."""
        a = []
        s = 5.0
        result = scale(a, s)
        expected = []
        assert result == expected
    
    def test_scale_very_small_scalar(self):
        """Test scaling with very small scalar."""
        a = [1.0, 2.0, 3.0]
        s = 1e-10
        result = scale(a, s)
        expected = [1e-10, 2e-10, 3e-10]
        assert all(abs(r - e) < 1e-20 for r, e in zip(result, expected))
    
    def test_scale_very_large_scalar(self):
        """Test scaling with very large scalar."""
        a = [1.0, 2.0, 3.0]
        s = 1e6
        result = scale(a, s)
        expected = [1e6, 2e6, 3e6]
        assert all(abs(r - e) < 1e-6 for r, e in zip(result, expected))
    
    def test_scale_very_small_vector(self):
        """Test scaling of vector with very small values."""
        a = [1e-10, 2e-10, 3e-10]
        s = 2.0
        result = scale(a, s)
        expected = [2e-10, 4e-10, 6e-10]
        assert all(abs(r - e) < 1e-20 for r, e in zip(result, expected))
    
    def test_scale_very_large_vector(self):
        """Test scaling of vector with very large values."""
        a = [1e6, 2e6, 3e6]
        s = 0.5
        result = scale(a, s)
        expected = [5e5, 1e6, 1.5e6]
        assert all(abs(r - e) < 1e-6 for r, e in zip(result, expected))
    
    def test_scale_does_not_modify_input(self):
        """Test that scaling does not modify input vector."""
        a = [1.0, 2.0, 3.0]
        a_original = a[:]
        scale(a, 2.0)
        assert a == a_original
    
    def test_scale_returns_new_list(self):
        """Test that scaling returns a new list, not a reference to input."""
        a = [1.0, 2.0, 3.0]
        result = scale(a, 2.0)
        # Modify result and ensure input is unchanged
        result[0] = 999.0
        assert a[0] == 1.0
    
    def test_scale_distributive_property(self):
        """Test that scaling distributes over addition: s*(a+b) = s*a + s*b."""
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        s = 2.0
        
        # s*(a+b)
        from core_services.math_helpers import add
        result1 = scale(add(a, b), s)
        
        # s*a + s*b
        result2 = add(scale(a, s), scale(b, s))
        
        assert result1 == result2
    
    def test_scale_associative_property(self):
        """Test that scaling is associative: (s1*s2)*a = s1*(s2*a)."""
        a = [1.0, 2.0, 3.0]
        s1 = 2.0
        s2 = 3.0
        
        # (s1*s2)*a
        result1 = scale(a, s1 * s2)
        
        # s1*(s2*a)
        result2 = scale(scale(a, s2), s1)
        
        assert result1 == result2
