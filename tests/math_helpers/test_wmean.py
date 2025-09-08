import pytest
import sys
import os

# Add the parent directory to the path to import from core_services
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core_services.math_helpers import wmean


class TestWmean:
    """Test cases for the weighted mean function."""
    
    def test_wmean_equal_weights(self):
        """Test weighted mean with equal weights."""
        vecs = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        weights = [1.0, 1.0, 1.0]
        result_vec, result_weight = wmean(vecs, weights)
        expected_vec = [4.0, 5.0, 6.0]  # (1+4+7)/3, (2+5+8)/3, (3+6+9)/3
        expected_weight = 3.0
        assert result_vec == expected_vec
        assert result_weight == expected_weight
    
    def test_wmean_different_weights(self):
        """Test weighted mean with different weights."""
        vecs = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        weights = [2.0, 3.0, 1.0]
        result_vec, result_weight = wmean(vecs, weights)
        # Weighted mean: (1*2 + 3*3 + 5*1)/(2+3+1), (2*2 + 4*3 + 6*1)/(2+3+1)
        # = (2+9+5)/6, (4+12+6)/6 = 16/6, 22/6 = 8/3, 11/3
        expected_vec = [8.0/3.0, 11.0/3.0]
        expected_weight = 6.0
        assert result_vec == expected_vec
        assert result_weight == expected_weight
    
    def test_wmean_single_vector(self):
        """Test weighted mean with single vector."""
        vecs = [[1.0, 2.0, 3.0]]
        weights = [5.0]
        result_vec, result_weight = wmean(vecs, weights)
        expected_vec = [1.0, 2.0, 3.0]
        expected_weight = 5.0
        assert result_vec == expected_vec
        assert result_weight == expected_weight
    
    def test_wmean_zero_weights(self):
        """Test weighted mean with zero weights."""
        vecs = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        weights = [0.0, 0.0]
        result_vec, result_weight = wmean(vecs, weights)
        # Should handle division by zero gracefully
        expected_vec = [0.0, 0.0, 0.0]  # Due to max(W, 1e-12) in implementation
        expected_weight = 0.0
        assert result_vec == expected_vec
        assert result_weight == expected_weight
    
    def test_wmean_negative_weights(self):
        """Test weighted mean with negative weights."""
        vecs = [[1.0, 2.0], [3.0, 4.0]]
        weights = [2.0, -1.0]
        result_vec, result_weight = wmean(vecs, weights)
        # Weighted mean: (1*2 + 3*(-1))/(2+(-1)), (2*2 + 4*(-1))/(2+(-1))
        # = (2-3)/1, (4-4)/1 = -1/1, 0/1 = -1, 0
        expected_vec = [-1.0, 0.0]
        expected_weight = 1.0
        assert result_vec == expected_vec
        assert result_weight == expected_weight
    
    def test_wmean_mixed_signs_vectors(self):
        """Test weighted mean with mixed positive and negative vector values."""
        vecs = [[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]
        weights = [2.0, 3.0]
        result_vec, result_weight = wmean(vecs, weights)
        # Weighted mean: (1*2 + (-4)*3)/(2+3), (-2*2 + 5*3)/(2+3), (3*2 + (-6)*3)/(2+3)
        # = (2-12)/5, (-4+15)/5, (6-18)/5 = -10/5, 11/5, -12/5 = -2, 2.2, -2.4
        expected_vec = [-2.0, 11.0/5.0, -12.0/5.0]
        expected_weight = 5.0
        assert result_vec == expected_vec
        assert result_weight == expected_weight
    
    def test_wmean_large_vectors(self):
        """Test weighted mean with larger vectors."""
        vecs = [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0], [11.0, 12.0, 13.0, 14.0, 15.0]]
        weights = [1.0, 2.0, 3.0]
        result_vec, result_weight = wmean(vecs, weights)
        # Weighted mean: (1*1 + 6*2 + 11*3)/6, (2*1 + 7*2 + 12*3)/6, etc.
        # = (1+12+33)/6, (2+14+36)/6, (3+16+39)/6, (4+18+42)/6, (5+20+45)/6
        # = 46/6, 52/6, 58/6, 64/6, 70/6
        expected_vec = [46.0/6.0, 52.0/6.0, 58.0/6.0, 64.0/6.0, 70.0/6.0]
        expected_weight = 6.0
        assert result_vec == expected_vec
        assert result_weight == expected_weight
    
    def test_wmean_single_element_vectors(self):
        """Test weighted mean with single-element vectors."""
        vecs = [[1.0], [2.0], [3.0]]
        weights = [1.0, 2.0, 3.0]
        result_vec, result_weight = wmean(vecs, weights)
        # Weighted mean: (1*1 + 2*2 + 3*3)/(1+2+3) = (1+4+9)/6 = 14/6 = 7/3
        expected_vec = [7.0/3.0]
        expected_weight = 6.0
        assert result_vec == expected_vec
        assert result_weight == expected_weight
    
    def test_wmean_float_precision(self):
        """Test weighted mean with floating point precision."""
        vecs = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        weights = [0.1, 0.2, 0.3]
        result_vec, result_weight = wmean(vecs, weights)
        # Weighted mean: (0.1*0.1 + 0.3*0.2 + 0.5*0.3)/0.6, (0.2*0.1 + 0.4*0.2 + 0.6*0.3)/0.6
        # = (0.01+0.06+0.15)/0.6, (0.02+0.08+0.18)/0.6 = 0.22/0.6, 0.28/0.6
        expected_vec = [0.22/0.6, 0.28/0.6]
        expected_weight = 0.6
        assert all(abs(r - e) < 1e-10 for r, e in zip(result_vec, expected_vec))
        assert abs(result_weight - expected_weight) < 1e-10
    
    def test_wmean_very_small_weights(self):
        """Test weighted mean with very small weights."""
        vecs = [[1.0, 2.0], [3.0, 4.0]]
        weights = [1e-10, 2e-10]
        result_vec, result_weight = wmean(vecs, weights)
        # Weighted mean: (1*1e-10 + 3*2e-10)/(1e-10+2e-10), (2*1e-10 + 4*2e-10)/(1e-10+2e-10)
        # = (1e-10+6e-10)/3e-10, (2e-10+8e-10)/3e-10 = 7e-10/3e-10, 10e-10/3e-10 = 7/3, 10/3
        expected_vec = [7.0/3.0, 10.0/3.0]
        expected_weight = 3e-10
        assert all(abs(r - e) < 1e-10 for r, e in zip(result_vec, expected_vec))
        assert abs(result_weight - expected_weight) < 1e-10
    
    def test_wmean_very_large_weights(self):
        """Test weighted mean with very large weights."""
        vecs = [[1.0, 2.0], [3.0, 4.0]]
        weights = [1e6, 2e6]
        result_vec, result_weight = wmean(vecs, weights)
        # Weighted mean: (1*1e6 + 3*2e6)/(1e6+2e6), (2*1e6 + 4*2e6)/(1e6+2e6)
        # = (1e6+6e6)/3e6, (2e6+8e6)/3e6 = 7e6/3e6, 10e6/3e6 = 7/3, 10/3
        expected_vec = [7.0/3.0, 10.0/3.0]
        expected_weight = 3e6
        assert all(abs(r - e) < 1e-10 for r, e in zip(result_vec, expected_vec))
        assert abs(result_weight - expected_weight) < 1e-10
    
    def test_wmean_empty_vectors_raises_error(self):
        """Test that weighted mean with empty vectors raises error."""
        vecs = []
        weights = []
        with pytest.raises(IndexError):
            wmean(vecs, weights)
    
    def test_wmean_different_lengths_uses_shorter(self):
        """Test that weighted mean uses the shorter vector length."""
        vecs = [[1.0, 2.0], [3.0, 4.0, 5.0]]
        weights = [1.0, 1.0]
        result_vec, result_weight = wmean(vecs, weights)
        # Should only use first 2 elements: (1*1 + 3*1)/(1+1), (2*1 + 4*1)/(1+1) = 4/2, 6/2 = 2, 3
        expected_vec = [2.0, 3.0]
        expected_weight = 2.0
        assert result_vec == expected_vec
        assert result_weight == expected_weight
    
    def test_wmean_different_weights_length_uses_shorter(self):
        """Test that weighted mean uses the shorter length when weights and vectors have different lengths."""
        vecs = [[1.0, 2.0], [3.0, 4.0]]
        weights = [1.0, 2.0, 3.0]  # One extra weight
        result_vec, result_weight = wmean(vecs, weights)
        # Should only use first 2 weights: (1*1 + 3*2)/(1+2), (2*1 + 4*2)/(1+2) = 7/3, 10/3
        expected_vec = [7.0/3.0, 10.0/3.0]
        expected_weight = 3.0
        assert result_vec == expected_vec
        assert result_weight == expected_weight
    
    def test_wmean_does_not_modify_input(self):
        """Test that weighted mean does not modify input vectors."""
        vecs = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        weights = [1.0, 2.0]
        vecs_original = [v[:] for v in vecs]
        weights_original = weights[:]
        wmean(vecs, weights)
        assert vecs == vecs_original
        assert weights == weights_original
    
    def test_wmean_returns_new_vectors(self):
        """Test that weighted mean returns new vectors, not references to input."""
        vecs = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        weights = [1.0, 2.0]
        result_vec, result_weight = wmean(vecs, weights)
        # Modify result and ensure inputs are unchanged
        result_vec[0] = 999.0
        assert vecs[0][0] == 1.0
        assert vecs[1][0] == 4.0
    
    def test_wmean_commutative_property(self):
        """Test that weighted mean is commutative (order doesn't matter)."""
        vecs1 = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        weights1 = [1.0, 2.0, 3.0]
        result1_vec, result1_weight = wmean(vecs1, weights1)
        
        vecs2 = [[5.0, 6.0], [1.0, 2.0], [3.0, 4.0]]  # Same vectors, different order
        weights2 = [3.0, 1.0, 2.0]  # Same weights, different order
        result2_vec, result2_weight = wmean(vecs2, weights2)
        
        assert all(abs(r1 - r2) < 1e-10 for r1, r2 in zip(result1_vec, result2_vec))
        assert abs(result1_weight - result2_weight) < 1e-10
    
    def test_wmean_linearity_property(self):
        """Test that weighted mean is linear: wmean(a + b) = wmean(a) + wmean(b)."""
        vecs_a = [[1.0, 2.0], [3.0, 4.0]]
        weights_a = [1.0, 2.0]
        vecs_b = [[5.0, 6.0], [7.0, 8.0]]
        weights_b = [1.0, 2.0]
        
        # wmean(a + b)
        from core_services.math_helpers import add
        vecs_sum = [add(a, b) for a, b in zip(vecs_a, vecs_b)]
        weights_sum = [w1 + w2 for w1, w2 in zip(weights_a, weights_b)]
        result1_vec, result1_weight = wmean(vecs_sum, weights_sum)
        
        # wmean(a) + wmean(b)
        mean_a_vec, mean_a_weight = wmean(vecs_a, weights_a)
        mean_b_vec, mean_b_weight = wmean(vecs_b, weights_b)
        result2_vec = add(mean_a_vec, mean_b_vec)
        result2_weight = mean_a_weight + mean_b_weight
        
        assert all(abs(r1 - r2) < 1e-10 for r1, r2 in zip(result1_vec, result2_vec))
        assert abs(result1_weight - result2_weight) < 1e-10
    
    def test_wmean_scaling_property(self):
        """Test that weighted mean scales: wmean(s * v, w) = s * wmean(v, w)."""
        vecs = [[1.0, 2.0], [3.0, 4.0]]
        weights = [1.0, 2.0]
        s = 2.0
        
        # wmean(s * v, w)
        from core_services.math_helpers import scale
        vecs_scaled = [scale(v, s) for v in vecs]
        result1_vec, result1_weight = wmean(vecs_scaled, weights)
        
        # s * wmean(v, w)
        mean_vec, mean_weight = wmean(vecs, weights)
        result2_vec = scale(mean_vec, s)
        result2_weight = mean_weight
        
        assert all(abs(r1 - r2) < 1e-10 for r1, r2 in zip(result1_vec, result2_vec))
        assert abs(result1_weight - result2_weight) < 1e-10
