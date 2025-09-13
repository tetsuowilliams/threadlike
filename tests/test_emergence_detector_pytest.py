"""Pytest tests for EmergenceDetector."""

import pytest
from unittest.mock import Mock
import uuid
import time

from core_services.emergence_detector import EmergenceDetector
from models import Topic, ClusterSnapshot, ClusterState, Doc, TopicPolicy, NegativeRules
from protocols import EmergenceNamer


class MockNamer(EmergenceNamer):
    """Mock namer for testing."""
    
    def name_and_seeds(self, cluster_docs: list[Doc]) -> tuple[str, list[str]]:
        return "Test Topic", ["test", "seeds"]


@pytest.fixture
def policy():
    """Create test policy."""
    return TopicPolicy(
        m_min=6,
        tau_cohesion=0.55,
        tau_separation=0.70,
        persistence_min=2
    )


@pytest.fixture
def topic(policy):
    """Create test topic."""
    return Topic(
        id="parent-topic-id",
        name="Parent Topic",
        seeds=["parent", "seeds"],
        negative=NegativeRules(),
        policy=policy,
        centroid_long=[0.1, 0.2, 0.3]
    )


@pytest.fixture
def cluster_snapshot():
    """Create test cluster snapshot."""
    return ClusterSnapshot(
        cluster_id="cluster-1",
        centroid_now=[0.4, 0.5, 0.6],
        size=8,  # Above m_min
        cohesion_now=0.6,  # Above tau_cohesion
        separation_now=0.3,  # Below tau_separation (1 - 0.3 = 0.7 >= 0.70)
        doc_ids=["doc1", "doc2", "doc3"]
    )


@pytest.fixture
def cluster_state():
    """Create test cluster state."""
    return ClusterState(
        cluster_id="cluster-1",
        cohesion_ema=0.6,  # Above tau_cohesion
        separation_ema=0.3,  # Below tau_separation
        persistence=3  # Above persistence_min
    )


@pytest.fixture
def test_docs():
    """Create test documents."""
    return [
        Doc(
            id="doc1",
            ts=time.time(),
            url="http://test.com/doc1",
            domain="test.com",
            title="Test Doc 1",
            text="test content 1",
            dtype="blog",
            authority=0.8,
            vec=[0.1, 0.2, 0.3],
            hash="hash1",
            arm_id="arm1"
        ),
        Doc(
            id="doc2",
            ts=time.time(),
            url="http://test.com/doc2",
            domain="test.com",
            title="Test Doc 2",
            text="test content 2",
            dtype="blog",
            authority=0.9,
            vec=[0.2, 0.3, 0.4],
            hash="hash2",
            arm_id="arm2"
        )
    ]


@pytest.fixture
def detector():
    """Create EmergenceDetector instance."""
    return EmergenceDetector()


class TestEmergenceDetectorReady:
    """Test cases for EmergenceDetector.ready() method."""
    
    def test_ready_all_criteria_met(self, detector: EmergenceDetector, topic: Topic, cluster_snapshot: ClusterSnapshot, cluster_state: ClusterState):
        """Test ready() returns True when all criteria are met."""
        assert detector.ready(topic, cluster_snapshot, cluster_state) is True
    
    def test_ready_size_too_small(self, detector, topic, cluster_state):
        """Test ready() returns False when cluster size is too small."""
        small_snapshot = ClusterSnapshot(
            cluster_id="cluster-1",
            centroid_now=[0.4, 0.5, 0.6],
            size=3,  # Below m_min
            cohesion_now=0.6,
            separation_now=0.3,
            doc_ids=["doc1"]
        )
        assert detector.ready(topic, small_snapshot, cluster_state) is False
    
    def test_ready_cohesion_too_low(self, detector, topic, cluster_snapshot):
        """Test ready() returns False when cohesion is too low."""
        low_cohesion_state = ClusterState(
            cluster_id="cluster-1",
            cohesion_ema=0.4,  # Below tau_cohesion
            separation_ema=0.3,
            persistence=3
        )
        assert detector.ready(topic, cluster_snapshot, low_cohesion_state) is False
    
    def test_ready_separation_too_high(self, detector, topic, cluster_snapshot):
        """Test ready() returns False when separation is too high."""
        high_separation_state = ClusterState(
            cluster_id="cluster-1",
            cohesion_ema=0.6,
            separation_ema=0.2,  # Below tau_separation (1 - 0.2 = 0.8 > 0.70)
            persistence=3
        )
        assert detector.ready(topic, cluster_snapshot, high_separation_state) is False
    
    def test_ready_persistence_too_low(self, detector, topic, cluster_snapshot):
        """Test ready() returns False when persistence is too low."""
        low_persistence_state = ClusterState(
            cluster_id="cluster-1",
            cohesion_ema=0.6,
            separation_ema=0.3,
            persistence=1  # Below persistence_min
        )
        assert detector.ready(topic, cluster_snapshot, low_persistence_state) is False
    
    @pytest.mark.parametrize("cohesion_ema,expected", [
        (0.55, True),   # Exactly tau_cohesion
        (0.54, False),  # Just below tau_cohesion
        (0.56, True),   # Just above tau_cohesion
    ])
    def test_ready_cohesion_edge_cases(self, detector, topic, cluster_snapshot, cohesion_ema, expected):
        """Test ready() with cohesion at various thresholds."""
        state = ClusterState(
            cluster_id="cluster-1",
            cohesion_ema=cohesion_ema,
            separation_ema=0.3,
            persistence=3
        )
        assert detector.ready(topic, cluster_snapshot, state) is expected
    
    @pytest.mark.parametrize("separation_ema,expected", [
        (0.3, True),   # Exactly tau_separation (1 - 0.3 = 0.7 = 0.70)
        (0.29, False), # Just above tau_separation (1 - 0.29 = 0.71 > 0.70)
        (0.31, True),  # Just below tau_separation (1 - 0.31 = 0.69 < 0.70)
    ])
    def test_ready_separation_edge_cases(self, detector, topic, cluster_snapshot, separation_ema, expected):
        """Test ready() with separation at various thresholds."""
        state = ClusterState(
            cluster_id="cluster-1",
            cohesion_ema=0.6,
            separation_ema=separation_ema,
            persistence=3
        )
        assert detector.ready(topic, cluster_snapshot, state) is expected
    
    @pytest.mark.parametrize("persistence,expected", [
        (2, True),   # Exactly persistence_min
        (1, False),  # Just below persistence_min
        (3, True),   # Just above persistence_min
    ])
    def test_ready_persistence_edge_cases(self, detector, topic, cluster_snapshot, persistence, expected):
        """Test ready() with persistence at various thresholds."""
        state = ClusterState(
            cluster_id="cluster-1",
            cohesion_ema=0.6,
            separation_ema=0.3,
            persistence=persistence
        )
        assert detector.ready(topic, cluster_snapshot, state) is expected


class TestEmergenceDetectorPromote:
    """Test cases for EmergenceDetector.promote() method."""
    
    def test_promote_creates_correct_topic(self, detector: EmergenceDetector, topic: Topic, cluster_snapshot: ClusterSnapshot, cluster_state: ClusterState, test_docs: list[Doc]):
        """Test promote() creates a topic with correct properties."""
        namer = MockNamer()
        result = detector.promote(topic, cluster_snapshot, namer, test_docs)
        
        # Check basic properties
        assert isinstance(result, Topic)
        assert result.id != topic.id  # Different ID
        assert result.name == "Test Topic"
        assert result.seeds == ["test", "seeds"]
        assert result.emerged_from == topic.id
        
        # Check inherited properties
        assert result.negative == topic.negative
        assert result.policy == topic.policy
        
        # Check centroid and weight
        assert result.centroid_long == cluster_snapshot.centroid_now
        assert result.doc_count == cluster_snapshot.size
        assert result.centroid_short_ema is None
        
        # Check timestamp is recent
        assert result.last_updated_ts > time.time() - 1
    
    def test_promote_uses_namer(self, detector, topic, cluster_snapshot, test_docs):
        """Test promote() calls the namer with correct documents."""
        mock_namer = Mock(spec=EmergenceNamer)
        mock_namer.name_and_seeds.return_value = ("Mock Topic", ["mock", "seeds"])
        
        result = detector.promote(topic, cluster_snapshot, mock_namer, test_docs)
        
        # Verify namer was called with correct documents
        mock_namer.name_and_seeds.assert_called_once_with(test_docs)
        
        # Verify result uses namer output
        assert result.name == "Mock Topic"
        assert result.seeds == ["mock", "seeds"]
    
    def test_promote_centroid_copy(self, detector, topic, test_docs):
        """Test promote() creates a copy of the centroid."""
        original_centroid = [0.4, 0.5, 0.6]
        snapshot = ClusterSnapshot(
            cluster_id="cluster-1",
            centroid_now=original_centroid,
            size=8,
            cohesion_now=0.6,
            separation_now=0.3,
            doc_ids=["doc1"]
        )
        
        result = detector.promote(topic, snapshot, MockNamer(), test_docs)
        
        # Should be a copy, not the same object
        assert result.centroid_long == original_centroid
        assert result.centroid_long is not original_centroid
        
        # Modifying the original shouldn't affect the result
        original_centroid[0] = 999.0
        assert result.centroid_long[0] != 999.0
    
    def test_promote_empty_docs(self, detector, topic, cluster_snapshot):
        """Test promote() works with empty document list."""
        result = detector.promote(topic, cluster_snapshot, MockNamer(), [])
        
        # Should still create a valid topic
        assert isinstance(result, Topic)
        assert result.name == "Test Topic"
        assert result.seeds == ["test", "seeds"]


class TestEmergenceDetectorIntegration:
    """Integration tests for EmergenceDetector."""
    
    @pytest.mark.parametrize("size,cohesion,separation,persistence,expected", [
        (8, 0.6, 0.3, 3, True),   # All criteria met
        (3, 0.6, 0.3, 3, False),  # Size too small
        (8, 0.4, 0.3, 3, False),  # Cohesion too low
        (8, 0.6, 0.2, 3, False),  # Separation too high
        (8, 0.6, 0.3, 1, False),  # Persistence too low
    ])
    def test_ready_combinations(self, detector, topic, size, cohesion, separation, persistence, expected):
        """Test ready() with various combinations of parameters."""
        snapshot = ClusterSnapshot(
            cluster_id="cluster-1",
            centroid_now=[0.4, 0.5, 0.6],
            size=size,
            cohesion_now=cohesion,
            separation_now=separation,
            doc_ids=["doc1"] * size
        )
        state = ClusterState(
            cluster_id="cluster-1",
            cohesion_ema=cohesion,
            separation_ema=separation,
            persistence=persistence
        )
        assert detector.ready(topic, snapshot, state) is expected
