import pytest
import sys
import os
import math

# Add the parent directory to the path to import from core_services
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core_services.math_helpers import cos


class TestCos:
    """Test cases for the cosine similarity function."""
    
    def test_cos_identical_vectors(self):
        """Test cosine similarity of identical vectors (should be 1)."""
        a = [1.0, 2.0, 3.0]
        b = [1.0, 2.0, 3.0]
        result = cos(a, b)
        assert abs(result - 1.0) < 1e-10
    
    def test_cos_orthogonal_vectors(self):
        """Test cosine similarity of orthogonal vectors (should be 0)."""
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        result = cos(a, b)
        assert abs(result - 0.0) < 1e-10
    
    def test_cos_opposite_vectors(self):
        """Test cosine similarity of opposite vectors (should be -1)."""
        a = [1.0, 2.0, 3.0]
        b = [-1.0, -2.0, -3.0]
        result = cos(a, b)
        assert abs(result - (-1.0)) < 1e-10
    
    def test_cos_45_degree_angle(self):
        """Test cosine similarity of vectors at 45-degree angle."""
        a = [1.0, 0.0]
        b = [1.0, 1.0]
        result = cos(a, b)
        expected = 1.0 / math.sqrt(2.0)  # cos(45°)
        assert abs(result - expected) < 1e-10
    
    def test_cos_60_degree_angle(self):
        """Test cosine similarity of vectors at 60-degree angle."""
        a = [1.0, 0.0]
        b = [0.5, math.sqrt(3)/2]
        result = cos(a, b)
        expected = 0.5  # cos(60°)
        assert abs(result - expected) < 1e-10
    
    def test_cos_30_degree_angle(self):
        """Test cosine similarity of vectors at 30-degree angle."""
        a = [1.0, 0.0]
        b = [math.sqrt(3)/2, 0.5]
        result = cos(a, b)
        expected = math.sqrt(3)/2  # cos(30°)
        assert abs(result - expected) < 1e-10
    
    def test_cos_scaled_vectors(self):
        """Test cosine similarity is invariant to scaling."""
        a = [1.0, 2.0, 3.0]
        b = [2.0, 4.0, 6.0]  # 2 * a
        result = cos(a, b)
        assert abs(result - 1.0) < 1e-10
    
    def test_cos_negative_scaled_vectors(self):
        """Test cosine similarity with negative scaling."""
        a = [1.0, 2.0, 3.0]
        b = [-2.0, -4.0, -6.0]  # -2 * a
        result = cos(a, b)
        assert abs(result - (-1.0)) < 1e-10
    
    def test_cos_mixed_signs(self):
        """Test cosine similarity with mixed positive and negative values."""
        a = [1.0, -2.0, 3.0]
        b = [4.0, 5.0, -6.0]
        result = cos(a, b)
        # Manual calculation: dot = 4 - 10 - 18 = -24
        # norm_a = sqrt(1 + 4 + 9) = sqrt(14)
        # norm_b = sqrt(16 + 25 + 36) = sqrt(77)
        # cos = -24 / (sqrt(14) * sqrt(77))
        expected = -24.0 / (math.sqrt(14.0) * math.sqrt(77.0))
        assert abs(result - expected) < 1e-10
    
    def test_cos_single_element(self):
        """Test cosine similarity of single-element vectors."""
        a = [5.0]
        b = [3.0]
        result = cos(a, b)
        assert abs(result - 1.0) < 1e-10  # Same direction
    
    def test_cos_large_vectors(self):
        """Test cosine similarity of larger vectors."""
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [2.0, 3.0, 4.0, 5.0, 6.0]
        result = cos(a, b)
        # Manual calculation
        dot_product = 2.0 + 6.0 + 12.0 + 20.0 + 30.0  # 70
        norm_a = math.sqrt(1.0 + 4.0 + 9.0 + 16.0 + 25.0)  # sqrt(55)
        norm_b = math.sqrt(4.0 + 9.0 + 16.0 + 25.0 + 36.0)  # sqrt(90)
        expected = dot_product / (norm_a * norm_b)
        assert abs(result - expected) < 1e-10
    
    def test_cos_float_precision(self):
        """Test cosine similarity with floating point precision."""
        a = [0.1, 0.2, 0.3]
        b = [0.4, 0.5, 0.6]
        result = cos(a, b)
        # Manual calculation
        dot_product = 0.04 + 0.10 + 0.18  # 0.32
        norm_a = math.sqrt(0.01 + 0.04 + 0.09)  # sqrt(0.14)
        norm_b = math.sqrt(0.16 + 0.25 + 0.36)  # sqrt(0.77)
        expected = dot_product / (norm_a * norm_b)
        assert abs(result - expected) < 1e-10
    
    def test_cos_zero_vectors(self):
        """Test cosine similarity with zero vectors."""
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        result = cos(a, b)
        # Should handle division by zero gracefully due to norm's minimum value
        assert abs(result) < 1e-10  # Should be close to 0
    
    def test_cos_empty_vectors(self):
        """Test cosine similarity of empty vectors."""
        a = []
        b = []
        result = cos(a, b)
        # Should handle empty vectors gracefully
        assert abs(result) < 1e-10  # Should be close to 0
    
    def test_cos_range_validation(self):
        """Test that cosine similarity is always in range [-1, 1]."""
        test_vectors = [
            ([1.0, 0.0], [0.0, 1.0]),
            ([1.0, 1.0], [1.0, -1.0]),
            ([1.0, 2.0], [3.0, 4.0]),
            ([0.0, 0.0], [1.0, 1.0]),
            ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ]
        
        for a, b in test_vectors:
            result = cos(a, b)
            assert -1.0 <= result <= 1.0, f"Cosine similarity {result} not in range [-1, 1] for vectors {a}, {b}"
