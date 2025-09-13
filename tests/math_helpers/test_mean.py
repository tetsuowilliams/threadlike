import pytest
import sys
import os

# Add the parent directory to the path to import from core_services
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core_services.math_helpers import mean


class TestMean:
    """Test cases for the mean function."""
    
    def test_mean_two_vectors(self):
        """Test mean of two vectors."""
        vs = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        result = mean(vs)
        expected = [2.5, 3.5, 4.5]
        assert result == expected
    
    def test_mean_three_vectors(self):
        """Test mean of three vectors."""
        vs = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        result = mean(vs)
        expected = [3.0, 4.0]  # (1+3+5)/3, (2+4+6)/3
        assert result == expected
    
    def test_mean_identical_vectors(self):
        """Test mean of identical vectors."""
        vs = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
        result = mean(vs)
        expected = [1.0, 2.0, 3.0]
        assert result == expected
    
    def test_mean_single_vector(self):
        """Test mean of single vector."""
        vs = [[1.0, 2.0, 3.0]]
        result = mean(vs)
        expected = [1.0, 2.0, 3.0]
        assert result == expected
    
    def test_mean_mixed_signs(self):
        """Test mean of vectors with mixed signs."""
        vs = [[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0], [7.0, -8.0, 9.0]]
        result = mean(vs)
        expected = [(1.0-4.0+7.0)/3, (-2.0+5.0-8.0)/3, (3.0-6.0+9.0)/3]
        expected = [4.0/3, -5.0/3, 6.0/3]
        assert all(abs(r - e) < 1e-10 for r, e in zip(result, expected))
    
    def test_mean_zero_vectors(self):
        """Test mean including zero vectors."""
        vs = [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]
        result = mean(vs)
        expected = [1.0, 2.0, 3.0]  # (0+1+2)/3, (0+2+4)/3, (0+3+6)/3
        assert result == expected
    
    def test_mean_negative_vectors(self):
        """Test mean of negative vectors."""
        vs = [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]]
        result = mean(vs)
        expected = [-2.5, -3.5, -4.5]
        assert result == expected
    
    def test_mean_large_vectors(self):
        """Test mean of larger vectors."""
        vs = [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0], [11.0, 12.0, 13.0, 14.0, 15.0]]
        result = mean(vs)
        expected = [6.0, 7.0, 8.0, 9.0, 10.0]  # (1+6+11)/3, (2+7+12)/3, etc.
        assert result == expected
    
    def test_mean_float_precision(self):
        """Test mean with floating point precision."""
        vs = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        result = mean(vs)
        expected = [0.3, 0.4]  # (0.1+0.3+0.5)/3, (0.2+0.4+0.6)/3
        assert all(abs(r - e) < 1e-10 for r, e in zip(result, expected))
    
    def test_mean_single_element_vectors(self):
        """Test mean of single-element vectors."""
        vs = [[1.0], [2.0], [3.0]]
        result = mean(vs)
        expected = [2.0]
        assert result == expected
    
    def test_mean_empty_list_raises_error(self):
        """Test that mean of empty list raises error."""
        vs = []
        with pytest.raises(IndexError):
            mean(vs)
    
    def test_mean_does_not_modify_input(self):
        """Test that mean does not modify input vectors."""
        vs = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        vs_original = [v[:] for v in vs]
        mean(vs)
        assert vs == vs_original
    
    def test_mean_returns_new_list(self):
        """Test that mean returns a new list, not a reference to input."""
        vs = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        result = mean(vs)
        # Modify result and ensure inputs are unchanged
        result[0] = 999.0
        assert vs[0][0] == 1.0
        assert vs[1][0] == 4.0
    
    def test_mean_commutative_property(self):
        """Test that mean is commutative (order doesn't matter)."""
        vs1 = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        vs2 = [[5.0, 6.0], [1.0, 2.0], [3.0, 4.0]]  # Same vectors, different order
        result1 = mean(vs1)
        result2 = mean(vs2)
        assert result1 == result2
    
    def test_mean_linearity_property(self):
        """Test that mean is linear: mean(a + b) = mean(a) + mean(b)."""
        vs_a = [[1.0, 2.0], [3.0, 4.0]]
        vs_b = [[5.0, 6.0], [7.0, 8.0]]
        
        # mean(a + b)
        from core_services.math_helpers import add
        vs_sum = [add(a, b) for a, b in zip(vs_a, vs_b)]
        result1 = mean(vs_sum)
        
        # mean(a) + mean(b)
        mean_a = mean(vs_a)
        mean_b = mean(vs_b)
        result2 = add(mean_a, mean_b)
        
        assert result1 == result2
    
    def test_mean_scaling_property(self):
        """Test that mean scales: mean(s * v) = s * mean(v)."""
        vs = [[1.0, 2.0], [3.0, 4.0]]
        s = 2.0
        
        # mean(s * v)
        from core_services.math_helpers import scale
        vs_scaled = [scale(v, s) for v in vs]
        result1 = mean(vs_scaled)
        
        # s * mean(v)
        mean_v = mean(vs)
        result2 = scale(mean_v, s)
        
        assert result1 == result2
