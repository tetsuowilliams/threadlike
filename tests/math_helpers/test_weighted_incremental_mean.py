import pytest
import sys
import os

# Add the parent directory to the path to import from core_services
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core_services.math_helpers import weighted_incremental_mean


class TestWeightedIncrementalMean:
    """Test cases for the weighted incremental mean function."""
    
    def test_first_element_none_previous(self):
        """Test first element when previous is None."""
        c_prev = None
        W_prev = 0.0
        e = [1.0, 2.0, 3.0]
        w = 2.0
        result_c, result_W = weighted_incremental_mean(c_prev, W_prev, e, w)
        expected_c = [1.0, 2.0, 3.0]
        expected_W = 2.0
        assert result_c == expected_c
        assert result_W == expected_W
    
    def test_first_element_zero_previous_weight(self):
        """Test first element when previous weight is zero."""
        c_prev = [0.0, 0.0, 0.0]
        W_prev = 0.0
        e = [1.0, 2.0, 3.0]
        w = 2.0
        result_c, result_W = weighted_incremental_mean(c_prev, W_prev, e, w)
        expected_c = [1.0, 2.0, 3.0]
        expected_W = 2.0
        assert result_c == expected_c
        assert result_W == expected_W
    
    def test_first_element_negative_previous_weight(self):
        """Test first element when previous weight is negative."""
        c_prev = [0.0, 0.0, 0.0]
        W_prev = -1.0
        e = [1.0, 2.0, 3.0]
        w = 2.0
        result_c, result_W = weighted_incremental_mean(c_prev, W_prev, e, w)
        expected_c = [1.0, 2.0, 3.0]
        expected_W = 2.0
        assert result_c == expected_c
        assert result_W == expected_W
    
    def test_second_element(self):
        """Test second element in sequence."""
        c_prev = [1.0, 2.0, 3.0]
        W_prev = 2.0
        e = [4.0, 5.0, 6.0]
        w = 3.0
        result_c, result_W = weighted_incremental_mean(c_prev, W_prev, e, w)
        # New weight: 2.0 + 3.0 = 5.0
        # New mean: (1*2 + 4*3)/5, (2*2 + 5*3)/5, (3*2 + 6*3)/5
        # = (2+12)/5, (4+15)/5, (6+18)/5 = 14/5, 19/5, 24/5
        expected_c = [14.0/5.0, 19.0/5.0, 24.0/5.0]
        expected_W = 5.0
        assert result_c == expected_c
        assert result_W == expected_W
    
    def test_third_element(self):
        """Test third element in sequence."""
        c_prev = [14.0/5.0, 19.0/5.0, 24.0/5.0]
        W_prev = 5.0
        e = [7.0, 8.0, 9.0]
        w = 1.0
        result_c, result_W = weighted_incremental_mean(c_prev, W_prev, e, w)
        # New weight: 5.0 + 1.0 = 6.0
        # New mean: (14/5*5 + 7*1)/6, (19/5*5 + 8*1)/6, (24/5*5 + 9*1)/6
        # = (14 + 7)/6, (19 + 8)/6, (24 + 9)/6 = 21/6, 27/6, 33/6
        expected_c = [21.0/6.0, 27.0/6.0, 33.0/6.0]
        expected_W = 6.0
        assert result_c == expected_c
        assert result_W == expected_W
    
    def test_equal_weights(self):
        """Test with equal weights."""
        c_prev = [1.0, 2.0, 3.0]
        W_prev = 1.0
        e = [4.0, 5.0, 6.0]
        w = 1.0
        result_c, result_W = weighted_incremental_mean(c_prev, W_prev, e, w)
        # New weight: 1.0 + 1.0 = 2.0
        # New mean: (1*1 + 4*1)/2, (2*1 + 5*1)/2, (3*1 + 6*1)/2
        # = 5/2, 7/2, 9/2
        expected_c = [2.5, 3.5, 4.5]
        expected_W = 2.0
        assert result_c == expected_c
        assert result_W == expected_W
    
    def test_different_weights(self):
        """Test with very different weights."""
        c_prev = [1.0, 2.0, 3.0]
        W_prev = 10.0
        e = [100.0, 200.0, 300.0]
        w = 1.0
        result_c, result_W = weighted_incremental_mean(c_prev, W_prev, e, w)
        # New weight: 10.0 + 1.0 = 11.0
        # New mean: (1*10 + 100*1)/11, (2*10 + 200*1)/11, (3*10 + 300*1)/11
        # = 110/11, 220/11, 330/11 = 10, 20, 30
        expected_c = [10.0, 20.0, 30.0]
        expected_W = 11.0
        assert result_c == expected_c
        assert result_W == expected_W
    
    def test_zero_weight(self):
        """Test with zero weight for new element."""
        c_prev = [1.0, 2.0, 3.0]
        W_prev = 2.0
        e = [4.0, 5.0, 6.0]
        w = 0.0
        result_c, result_W = weighted_incremental_mean(c_prev, W_prev, e, w)
        # New weight: 2.0 + 0.0 = 2.0
        # New mean: (1*2 + 4*0)/2, (2*2 + 5*0)/2, (3*2 + 6*0)/2
        # = 2/2, 4/2, 6/2 = 1, 2, 3 (unchanged)
        expected_c = [1.0, 2.0, 3.0]
        expected_W = 2.0
        assert result_c == expected_c
        assert result_W == expected_W
    
    def test_negative_weights(self):
        """Test with negative weights."""
        c_prev = [1.0, 2.0, 3.0]
        W_prev = 2.0
        e = [4.0, 5.0, 6.0]
        w = -1.0
        result_c, result_W = weighted_incremental_mean(c_prev, W_prev, e, w)
        # New weight: 2.0 + (-1.0) = 1.0
        # New mean: (1*2 + 4*(-1))/1, (2*2 + 5*(-1))/1, (3*2 + 6*(-1))/1
        # = (2-4)/1, (4-5)/1, (6-6)/1 = -2, -1, 0
        expected_c = [-2.0, -1.0, 0.0]
        expected_W = 1.0
        assert result_c == expected_c
        assert result_W == expected_W
    
    def test_single_element_vectors(self):
        """Test with single-element vectors."""
        c_prev = [5.0]
        W_prev = 2.0
        e = [10.0]
        w = 3.0
        result_c, result_W = weighted_incremental_mean(c_prev, W_prev, e, w)
        # New weight: 2.0 + 3.0 = 5.0
        # New mean: (5*2 + 10*3)/5 = (10 + 30)/5 = 40/5 = 8
        expected_c = [8.0]
        expected_W = 5.0
        assert result_c == expected_c
        assert result_W == expected_W
    
    def test_large_vectors(self):
        """Test with larger vectors."""
        c_prev = [1.0, 2.0, 3.0, 4.0, 5.0]
        W_prev = 1.0
        e = [6.0, 7.0, 8.0, 9.0, 10.0]
        w = 2.0
        result_c, result_W = weighted_incremental_mean(c_prev, W_prev, e, w)
        # New weight: 1.0 + 2.0 = 3.0
        # New mean: (1*1 + 6*2)/3, (2*1 + 7*2)/3, (3*1 + 8*2)/3, (4*1 + 9*2)/3, (5*1 + 10*2)/3
        # = 13/3, 16/3, 19/3, 22/3, 25/3
        expected_c = [13.0/3.0, 16.0/3.0, 19.0/3.0, 22.0/3.0, 25.0/3.0]
        expected_W = 3.0
        assert result_c == expected_c
        assert result_W == expected_W
    
    def test_float_precision(self):
        """Test with floating point precision."""
        c_prev = [0.1, 0.2, 0.3]
        W_prev = 0.5
        e = [0.4, 0.5, 0.6]
        w = 0.3
        result_c, result_W = weighted_incremental_mean(c_prev, W_prev, e, w)
        # New weight: 0.5 + 0.3 = 0.8
        # New mean: (0.1*0.5 + 0.4*0.3)/0.8, (0.2*0.5 + 0.5*0.3)/0.8, (0.3*0.5 + 0.6*0.3)/0.8
        # = (0.05 + 0.12)/0.8, (0.10 + 0.15)/0.8, (0.15 + 0.18)/0.8
        # = 0.17/0.8, 0.25/0.8, 0.33/0.8
        expected_c = [0.17/0.8, 0.25/0.8, 0.33/0.8]
        expected_W = 0.8
        assert all(abs(r - e) < 1e-10 for r, e in zip(result_c, expected_c))
        assert abs(result_W - expected_W) < 1e-10
    
    def test_does_not_modify_input(self):
        """Test that function does not modify input vectors."""
        c_prev = [1.0, 2.0, 3.0]
        e = [4.0, 5.0, 6.0]
        c_prev_original = c_prev[:]
        e_original = e[:]
        weighted_incremental_mean(c_prev, 2.0, e, 3.0)
        assert c_prev == c_prev_original
        assert e == e_original
    
    def test_returns_new_vectors(self):
        """Test that function returns new vectors, not references to input."""
        c_prev = [1.0, 2.0, 3.0]
        e = [4.0, 5.0, 6.0]
        result_c, result_W = weighted_incremental_mean(c_prev, 2.0, e, 3.0)
        # Modify result and ensure inputs are unchanged
        result_c[0] = 999.0
        assert c_prev[0] == 1.0
        assert e[0] == 4.0
    
    def test_consistency_with_batch_calculation(self):
        """Test that incremental calculation matches batch calculation."""
        # Simulate incremental updates
        vectors = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        weights = [2.0, 3.0, 1.0]
        
        # Incremental calculation
        c_prev = None
        W_prev = 0.0
        for v, w in zip(vectors, weights):
            c_prev, W_prev = weighted_incremental_mean(c_prev, W_prev, v, w)
        
        # Batch calculation (manual weighted mean)
        total_weight = sum(weights)
        batch_mean = [
            sum(v[i] * w for v, w in zip(vectors, weights)) / total_weight
            for i in range(len(vectors[0]))
        ]
        
        assert all(abs(inc - batch) < 1e-10 for inc, batch in zip(c_prev, batch_mean))
        assert abs(W_prev - total_weight) < 1e-10
