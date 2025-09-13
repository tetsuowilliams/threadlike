"""Pytest tests for HDBSCANClusterer."""

import pytest
import numpy as np
from unittest.mock import Mock
import uuid
from typing import List
from core_services.hdbscan_clusterer import HDBSCANClusterer
from models import Doc, ClusterSnapshot
from core_services.math_helpers import Vector


@pytest.fixture
def clusterer():
    """Create HDBSCANClusterer instance with default settings."""
    return HDBSCANClusterer(
        min_cluster_size=2,
        min_samples=1,
        cluster_selection_epsilon=0.0,
        n_jobs=1,
        min_mass=1.0,
        min_cohesion=0.3
    )


@pytest.fixture
def clusterer_strict():
    """Create HDBSCANClusterer with strict filtering."""
    return HDBSCANClusterer(
        min_cluster_size=5,
        min_samples=3,
        cluster_selection_epsilon=0.0,
        n_jobs=1,
        min_mass=10.0,
        min_cohesion=0.8
    )


@pytest.fixture
def centroid_long():
    """Create test centroid vector."""
    return [0.1, 0.2, 0.3, 0.4]


@pytest.fixture
def centroid_long_none():
    """Create None centroid for testing fallback behavior."""
    return None


def create_test_doc(doc_id: str, vec: List[float]) -> Doc:
    """Helper to create test documents."""
    return Doc(
        id=doc_id,
        ts=1234567890.0,
        url=f"https://example.com/{doc_id}",
        domain="example.com",
        title=f"Test Doc {doc_id}",
        text=f"Test content for {doc_id}",
        dtype="test",
        authority=0.8,
        vec=vec,
        hash=f"hash_{doc_id}",
        arm_id="test_arm"
    )


class TestHDBSCANClustererBasic:
    """Test basic clustering functionality."""
    
    def test_empty_docs_window(self, clusterer: HDBSCANClusterer, centroid_long: Vector):
        """Test clustering with empty docs window."""
        result = clusterer.cluster(centroid_long, [])
        assert result == []
    
    def test_single_doc(self, clusterer: HDBSCANClusterer, centroid_long: Vector):
        """Test clustering with single document."""
        doc = create_test_doc("doc1", [0.1, 0.2, 0.3, 0.4])
        result = clusterer.cluster(centroid_long, [doc])
        # Single doc should not form a cluster due to min_cluster_size=2
        assert result == []
    
    def test_two_docs(self, clusterer: HDBSCANClusterer, centroid_long: Vector):
        """Test clustering with two documents."""
        docs = [
            create_test_doc("doc1", [0.1, 0.2, 0.3, 0.4]),
            create_test_doc("doc2", [0.11, 0.21, 0.31, 0.41])
        ]
        result = clusterer.cluster(centroid_long, docs)
        # Two docs should form a cluster with min_cluster_size=2
        assert len(result) == 1
        cluster = result[0]
        assert cluster.size == 2
    
    def test_two_separate_clusters(self, clusterer: HDBSCANClusterer, centroid_long: Vector):
        """Test clustering with two separate clusters."""
        # Create two distinct groups of similar docs
        cluster1_docs = [
            create_test_doc("doc1", [0.1, 0.2, 0.3, 0.4]),
            create_test_doc("doc2", [0.11, 0.21, 0.31, 0.41]),
            create_test_doc("doc3", [0.12, 0.22, 0.32, 0.42])
        ]
        cluster2_docs = [
            create_test_doc("doc4", [0.9, 0.8, 0.7, 0.6]),
            create_test_doc("doc5", [0.91, 0.81, 0.71, 0.61]),
            create_test_doc("doc6", [0.92, 0.82, 0.72, 0.62])
        ]
        all_docs = cluster1_docs + cluster2_docs
        
        result = clusterer.cluster(centroid_long, all_docs)
        
        # Should form two clusters
        assert len(result) == 2
        assert all(cluster.cluster_id.startswith("h") for cluster in result)
        assert all(cluster.size == 3 for cluster in result)
        
        # Check that each cluster has the right docs
        doc_ids = [doc_id for cluster in result for doc_id in cluster.doc_ids]
        assert set(doc_ids) == {"doc1", "doc2", "doc3", "doc4", "doc5", "doc6"}


class TestHDBSCANClustererFiltering:
    """Test filtering based on mass and cohesion thresholds."""
    
    def test_min_cohesion_filtering(self, clusterer_strict: HDBSCANClusterer, centroid_long: Vector):
        """Test that clusters below min_cohesion are filtered out."""
        # Create docs that are too dissimilar to form cohesive clusters
        docs = [
            create_test_doc("doc1", [0.1, 0.2, 0.3, 0.4]),
            create_test_doc("doc2", [0.5, 0.6, 0.7, 0.8]),
            create_test_doc("doc3", [0.9, 0.1, 0.2, 0.3]),
            create_test_doc("doc4", [0.4, 0.5, 0.6, 0.7]),
            create_test_doc("doc5", [0.8, 0.9, 0.1, 0.2])
        ]
        result = clusterer_strict.cluster(centroid_long, docs)
        
        # With min_cohesion=0.8, these scattered docs should not form valid clusters
        assert result == []
    
    def test_noise_filtering(self, clusterer: HDBSCANClusterer, centroid_long: Vector):
        """Test that noise points (label -1) are filtered out."""
        # Create a clusterer with more lenient parameters for this test
        lenient_clusterer = HDBSCANClusterer(
            min_cluster_size=2,
            min_samples=1,
            cluster_selection_epsilon=0.0,
            n_jobs=1,
            min_mass=1.0,
            min_cohesion=0.1
        )
        side_length = 0.5 

        # Create a mix of very similar docs and one isolated noise point
        docs = [
            create_test_doc("doc1", [1.0, 0.0, 0.0, 0.0]),
            create_test_doc("doc2", [1.01, 0.01, 0.0, 0.0]),
            create_test_doc("noise1", [2.0, 2.0, 2.0, 2.0]),  # A distant noise point
        ]
        result = lenient_clusterer.cluster(centroid_long, docs)
        
        # Should only return clusters, not noise
        assert len(result) == 1
        cluster = result[0]
        assert "noise1" not in cluster.doc_ids
        assert len(cluster.doc_ids) == 2


class TestHDBSCANClustererNormalization:
    """Test L2 normalization and vector processing."""
    
    def test_l2_normalization(self, clusterer: HDBSCANClusterer, centroid_long: Vector):
        """Test that vectors are L2 normalized."""
        # Create docs with non-unit vectors
        docs = [
            create_test_doc("doc1", [2.0, 0.0, 0.0, 0.0]),  # Magnitude 2
            create_test_doc("doc2", [0.0, 3.0, 0.0, 0.0]),  # Magnitude 3
            create_test_doc("doc3", [0.0, 0.0, 4.0, 0.0]),  # Magnitude 4
        ]
        result = clusterer.cluster(centroid_long, docs)
        
        # Should still cluster despite different magnitudes
        assert len(result) == 1
        cluster = result[0]
        # Centroid should be normalized (unit vector)
        centroid_norm = np.linalg.norm(cluster.centroid_now)
        assert abs(centroid_norm - 1.0) < 1e-6
    
    def test_zero_vector_handling(self, clusterer: HDBSCANClusterer, centroid_long: Vector):
        """Test handling of zero vectors."""
        docs = [
            create_test_doc("doc1", [0.0, 0.0, 0.0, 0.0]),
            create_test_doc("doc2", [0.1, 0.2, 0.3, 0.4]),
            create_test_doc("doc3", [0.11, 0.21, 0.31, 0.41])
        ]
        result = clusterer.cluster(centroid_long, docs)
        
        # Should handle zero vector gracefully
        assert len(result) == 1
        cluster = result[0]
        assert cluster.size == 2


class TestHDBSCANClustererParentCentroid:
    """Test parent centroid handling."""
    
    def test_with_parent_centroid(self, clusterer: HDBSCANClusterer, centroid_long: Vector):
        """Test clustering with parent centroid."""
        docs = [
            create_test_doc("doc1", [0.1, 0.2, 0.3, 0.4]),
            create_test_doc("doc2", [0.11, 0.21, 0.31, 0.41]),
            create_test_doc("doc3", [0.12, 0.22, 0.32, 0.42])
        ]
        result = clusterer.cluster(centroid_long, docs)
        
        assert len(result) == 1
        cluster = result[0]
        # Separation should be calculated relative to parent centroid
        assert 0.0 <= cluster.separation_now <= 1.0
    
    def test_without_parent_centroid(self, clusterer: HDBSCANClusterer, centroid_long_none: Vector):
        """Test clustering without parent centroid (fallback to mean)."""
        docs = [
            create_test_doc("doc1", [0.1, 0.2, 0.3, 0.4]),
            create_test_doc("doc2", [0.11, 0.21, 0.31, 0.41]),
            create_test_doc("doc3", [0.12, 0.22, 0.32, 0.42])
        ]
        result = clusterer.cluster(centroid_long_none, docs)
        
        assert len(result) == 1
        cluster = result[0]
        # Should still calculate separation using mean of docs
        assert 0.0 <= cluster.separation_now <= 1.0


class TestHDBSCANClustererParameters:
    """Test parameter validation and effects."""
    
    def test_min_cluster_size_parameter(self, centroid_long: Vector):
        """Test that min_cluster_size parameter works correctly."""
        # Create clusterer with min_cluster_size=5
        clusterer = HDBSCANClusterer(min_cluster_size=5, min_samples=2)
        
        docs = [
            create_test_doc("doc1", [0.1, 0.2, 0.3, 0.4]),
            create_test_doc("doc2", [0.11, 0.21, 0.31, 0.41]),
            create_test_doc("doc3", [0.12, 0.22, 0.32, 0.42]),
            create_test_doc("doc4", [0.13, 0.23, 0.33, 0.43])
        ]
        result = clusterer.cluster(centroid_long, docs)
        
        # With min_cluster_size=5, 4 docs should not form a cluster
        assert result == []
    
    def test_min_samples_parameter(self, centroid_long: Vector):
        """Test that min_samples parameter affects clustering."""
        # Create clusterer with high min_samples
        clusterer = HDBSCANClusterer(min_cluster_size=3, min_samples=5)
        
        docs = [
            create_test_doc("doc1", [0.1, 0.2, 0.3, 0.4]),
            create_test_doc("doc2", [0.11, 0.21, 0.31, 0.41]),
            create_test_doc("doc3", [0.12, 0.22, 0.32, 0.42])
        ]
        result = clusterer.cluster(centroid_long, docs)
        
        # With min_samples=5, 3 docs should not form a cluster
        assert result == []
    
    def test_cluster_selection_epsilon_parameter(self, centroid_long: Vector):
        """Test that cluster_selection_epsilon parameter affects clustering."""
        # Create clusterer with high epsilon to merge close clusters
        clusterer = HDBSCANClusterer(
            min_cluster_size=2,
            min_mass=1,
            min_samples=2,
            cluster_selection_epsilon=0.5
        )
        
        # Create two close clusters
        cluster1_docs = [
            create_test_doc("doc1", [0.1, 0.2, 0.3, 0.4]),
            create_test_doc("doc2", [0.11, 0.21, 0.31, 0.41])
        ]
        cluster2_docs = [
            create_test_doc("doc3", [0.15, 0.25, 0.35, 0.45]),  # Close to cluster1
            create_test_doc("doc4", [0.16, 0.26, 0.36, 0.46])
        ]
        all_docs = cluster1_docs + cluster2_docs
        
        result = clusterer.cluster(centroid_long, all_docs)
        
        # With high epsilon, close clusters might be merged
        assert len(result) >= 1


class TestHDBSCANClustererEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_identical_docs(self, clusterer: HDBSCANClusterer, centroid_long: Vector):
        """Test clustering with identical documents."""
        docs = [
            create_test_doc("doc1", [0.1, 0.2, 0.3, 0.4]),
            create_test_doc("doc2", [0.1, 0.2, 0.3, 0.4]),  # Identical
            create_test_doc("doc3", [0.1, 0.2, 0.3, 0.4])   # Identical
        ]
        result = clusterer.cluster(centroid_long, docs)
        
        # Should still form a cluster
        assert len(result) == 1
        cluster = result[0]
        assert cluster.size == 3
        # Cohesion should be very high for identical docs
        assert cluster.cohesion_now > 0.9
    
    def test_very_dissimilar_docs(self, clusterer: HDBSCANClusterer, centroid_long: Vector):
        """Test clustering with very dissimilar documents."""
        docs = [
            create_test_doc("doc1", [1.0, 0.0, 0.0, 0.0]),
            create_test_doc("doc2", [0.0, 1.0, 0.0, 0.0]),
            create_test_doc("doc3", [0.0, 0.0, 1.0, 0.0])
        ]
        result = clusterer.cluster(centroid_long, docs)
        
        # May or may not form clusters depending on HDBSCAN's behavior
        # This tests that the method doesn't crash with orthogonal vectors
        assert isinstance(result, list)
        for cluster in result:
            assert isinstance(cluster, ClusterSnapshot)
    
    def test_single_dimension_vectors(self, clusterer: HDBSCANClusterer, centroid_long: Vector):
        """Test clustering with single dimension vectors."""
        # Create 1D centroid
        centroid_1d = [0.5]
        
        docs = [
            create_test_doc("doc1", [0.1]),
            create_test_doc("doc2", [0.11]),
            create_test_doc("doc3", [0.12])
        ]
        result = clusterer.cluster(centroid_1d, docs)
        
        # Should handle 1D vectors gracefully
        assert isinstance(result, list)
    
    def test_very_high_dimensional_vectors(self, clusterer: HDBSCANClusterer, centroid_long: Vector):
        """Test clustering with high dimensional vectors."""
        # Create high-dimensional vectors
        dim = 100
        docs = [
            create_test_doc("doc1", [0.1] * dim),
            create_test_doc("doc2", [0.11] * dim),
            create_test_doc("doc3", [0.12] * dim)
        ]
        
        # Create centroid with matching dimensionality
        centroid_hd = [0.2] * dim
        
        result = clusterer.cluster(centroid_hd, docs)
        
        # Should handle high dimensions gracefully
        assert isinstance(result, list)


class TestHDBSCANClustererHelperMethods:
    """Test internal helper methods."""
    
    def test_l2_normalize(self, clusterer: HDBSCANClusterer):
        """Test L2 normalization helper method."""
        # Test with non-unit vector
        V = np.array([[3.0, 4.0, 0.0]])
        normalized = clusterer._l2_normalize(V)
        
        # Should be unit vector
        norm = np.linalg.norm(normalized[0])
        assert abs(norm - 1.0) < 1e-6
        
        # Should preserve direction
        expected = np.array([3.0, 4.0, 0.0]) / 5.0
        assert np.allclose(normalized[0], expected)
