import pytest
import sys
import os

# Add the parent directory to the path to import from core_services
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core_services.math_helpers import ema_update_vec


class TestEmaUpdateVec:
    """Test cases for the EMA (Exponential Moving Average) update function."""
    
    def test_first_update_none_ema(self):
        """Test first update when EMA is None."""
        v_ema = None
        v_now = [1.0, 2.0, 3.0]
        beta = 0.5
        result = ema_update_vec(v_ema, v_now, beta)
        expected = [1.0, 2.0, 3.0]
        assert result == expected
    
    def test_second_update(self):
        """Test second update in sequence."""
        v_ema = [1.0, 2.0, 3.0]
        v_now = [4.0, 5.0, 6.0]
        beta = 0.5
        result = ema_update_vec(v_ema, v_now, beta)
        # EMA formula: (1-beta)*ema + beta*now
        # = (1-0.5)*[1,2,3] + 0.5*[4,5,6]
        # = 0.5*[1,2,3] + 0.5*[4,5,6]
        # = [0.5,1,1.5] + [2,2.5,3] = [2.5,3.5,4.5]
        expected = [2.5, 3.5, 4.5]
        assert result == expected
    
    def test_third_update(self):
        """Test third update in sequence."""
        v_ema = [2.5, 3.5, 4.5]
        v_now = [7.0, 8.0, 9.0]
        beta = 0.5
        result = ema_update_vec(v_ema, v_now, beta)
        # EMA formula: (1-0.5)*[2.5,3.5,4.5] + 0.5*[7,8,9]
        # = 0.5*[2.5,3.5,4.5] + 0.5*[7,8,9]
        # = [1.25,1.75,2.25] + [3.5,4,4.5] = [4.75,5.75,6.75]
        expected = [4.75, 5.75, 6.75]
        assert result == expected
    
    def test_beta_zero(self):
        """Test with beta = 0 (no update, keep EMA)."""
        v_ema = [1.0, 2.0, 3.0]
        v_now = [4.0, 5.0, 6.0]
        beta = 0.0
        result = ema_update_vec(v_ema, v_now, beta)
        # EMA formula: (1-0)*[1,2,3] + 0*[4,5,6] = [1,2,3]
        expected = [1.0, 2.0, 3.0]
        assert result == expected
    
    def test_beta_one(self):
        """Test with beta = 1 (complete update, use new value)."""
        v_ema = [1.0, 2.0, 3.0]
        v_now = [4.0, 5.0, 6.0]
        beta = 1.0
        result = ema_update_vec(v_ema, v_now, beta)
        # EMA formula: (1-1)*[1,2,3] + 1*[4,5,6] = [4,5,6]
        expected = [4.0, 5.0, 6.0]
        assert result == expected
    
    def test_beta_half(self):
        """Test with beta = 0.5 (equal weight to EMA and new value)."""
        v_ema = [1.0, 2.0, 3.0]
        v_now = [4.0, 5.0, 6.0]
        beta = 0.5
        result = ema_update_vec(v_ema, v_now, beta)
        # EMA formula: 0.5*[1,2,3] + 0.5*[4,5,6] = [2.5,3.5,4.5]
        expected = [2.5, 3.5, 4.5]
        assert result == expected
    
    def test_beta_quarter(self):
        """Test with beta = 0.25 (more weight to EMA)."""
        v_ema = [1.0, 2.0, 3.0]
        v_now = [4.0, 5.0, 6.0]
        beta = 0.25
        result = ema_update_vec(v_ema, v_now, beta)
        # EMA formula: 0.75*[1,2,3] + 0.25*[4,5,6]
        # = [0.75,1.5,2.25] + [1,1.25,1.5] = [1.75,2.75,3.75]
        expected = [1.75, 2.75, 3.75]
        assert result == expected
    
    def test_beta_three_quarters(self):
        """Test with beta = 0.75 (more weight to new value)."""
        v_ema = [1.0, 2.0, 3.0]
        v_now = [4.0, 5.0, 6.0]
        beta = 0.75
        result = ema_update_vec(v_ema, v_now, beta)
        # EMA formula: 0.25*[1,2,3] + 0.75*[4,5,6]
        # = [0.25,0.5,0.75] + [3,3.75,4.5] = [3.25,4.25,5.25]
        expected = [3.25, 4.25, 5.25]
        assert result == expected
    
    def test_negative_values(self):
        """Test with negative values."""
        v_ema = [-1.0, -2.0, -3.0]
        v_now = [-4.0, -5.0, -6.0]
        beta = 0.5
        result = ema_update_vec(v_ema, v_now, beta)
        # EMA formula: 0.5*[-1,-2,-3] + 0.5*[-4,-5,-6]
        # = [-0.5,-1,-1.5] + [-2,-2.5,-3] = [-2.5,-3.5,-4.5]
        expected = [-2.5, -3.5, -4.5]
        assert result == expected
    
    def test_mixed_signs(self):
        """Test with mixed positive and negative values."""
        v_ema = [1.0, -2.0, 3.0]
        v_now = [-4.0, 5.0, -6.0]
        beta = 0.5
        result = ema_update_vec(v_ema, v_now, beta)
        # EMA formula: 0.5*[1,-2,3] + 0.5*[-4,5,-6]
        # = [0.5,-1,1.5] + [-2,2.5,-3] = [-1.5,1.5,-1.5]
        expected = [-1.5, 1.5, -1.5]
        assert result == expected
    
    def test_single_element_vectors(self):
        """Test with single-element vectors."""
        v_ema = [5.0]
        v_now = [10.0]
        beta = 0.3
        result = ema_update_vec(v_ema, v_now, beta)
        # EMA formula: 0.7*5 + 0.3*10 = 3.5 + 3 = 6.5
        expected = [6.5]
        assert result == expected
    
    def test_large_vectors(self):
        """Test with larger vectors."""
        v_ema = [1.0, 2.0, 3.0, 4.0, 5.0]
        v_now = [6.0, 7.0, 8.0, 9.0, 10.0]
        beta = 0.2
        result = ema_update_vec(v_ema, v_now, beta)
        # EMA formula: 0.8*[1,2,3,4,5] + 0.2*[6,7,8,9,10]
        # = [0.8,1.6,2.4,3.2,4] + [1.2,1.4,1.6,1.8,2] = [2,3,4,5,6]
        expected = [2.0, 3.0, 4.0, 5.0, 6.0]
        assert result == expected
    
    def test_float_precision(self):
        """Test with floating point precision."""
        v_ema = [0.1, 0.2, 0.3]
        v_now = [0.4, 0.5, 0.6]
        beta = 0.1
        result = ema_update_vec(v_ema, v_now, beta)
        # EMA formula: 0.9*[0.1,0.2,0.3] + 0.1*[0.4,0.5,0.6]
        # = [0.09,0.18,0.27] + [0.04,0.05,0.06] = [0.13,0.23,0.33]
        expected = [0.13, 0.23, 0.33]
        assert all(abs(r - e) < 1e-10 for r, e in zip(result, expected))
    
    def test_very_small_beta(self):
        """Test with very small beta (almost no update)."""
        v_ema = [1.0, 2.0, 3.0]
        v_now = [100.0, 200.0, 300.0]
        beta = 1e-6
        result = ema_update_vec(v_ema, v_now, beta)
        # EMA formula: (1-1e-6)*[1,2,3] + 1e-6*[100,200,300]
        # ≈ [1,2,3] + [0.0001,0.0002,0.0003] ≈ [1,2,3]
        expected = [1.0, 2.0, 3.0]
        assert all(abs(r - e) < 1e-3 for r, e in zip(result, expected))
    
    def test_very_large_beta(self):
        """Test with beta very close to 1 (almost complete update)."""
        v_ema = [1.0, 2.0, 3.0]
        v_now = [100.0, 200.0, 300.0]
        beta = 1.0 - 1e-6
        result = ema_update_vec(v_ema, v_now, beta)
        # EMA formula: 1e-6*[1,2,3] + (1-1e-6)*[100,200,300]
        # ≈ [0.000001,0.000002,0.000003] + [100,200,300] ≈ [100,200,300]
        expected = [100.0, 200.0, 300.0]
        assert all(abs(r - e) < 1e-3 for r, e in zip(result, expected))
    
    def test_does_not_modify_input(self):
        """Test that function does not modify input vectors."""
        v_ema = [1.0, 2.0, 3.0]
        v_now = [4.0, 5.0, 6.0]
        v_ema_original = v_ema[:]
        v_now_original = v_now[:]
        ema_update_vec(v_ema, v_now, 0.5)
        assert v_ema == v_ema_original
        assert v_now == v_now_original
    
    def test_returns_new_vector(self):
        """Test that function returns a new vector, not a reference to input."""
        v_ema = [1.0, 2.0, 3.0]
        v_now = [4.0, 5.0, 6.0]
        result = ema_update_vec(v_ema, v_now, 0.5)
        # Modify result and ensure inputs are unchanged
        result[0] = 999.0
        assert v_ema[0] == 1.0
        assert v_now[0] == 4.0
    
    def test_ema_convergence_property(self):
        """Test that EMA converges to a constant value when input is constant."""
        v_ema = [1.0, 2.0, 3.0]
        constant_value = [5.0, 5.0, 5.0]
        beta = 0.1
        
        # Apply EMA update multiple times with constant value
        for _ in range(100):
            v_ema = ema_update_vec(v_ema, constant_value, beta)
        
        # Should converge close to the constant value
        assert all(abs(v - 5.0) < 0.01 for v in v_ema)
    
    def test_ema_smoothing_property(self):
        """Test that EMA smooths out rapid changes."""
        # Start with a value
        v_ema = [10.0, 10.0, 10.0]
        beta = 0.1
        
        # Apply alternating high and low values
        values = [[100.0, 100.0, 100.0], [0.0, 0.0, 0.0]] * 5
        
        for v_now in values:
            v_ema = ema_update_vec(v_ema, v_now, beta)
        
        # EMA should be somewhere between the extremes, not oscillating wildly
        assert all(0.0 < v < 100.0 for v in v_ema)
