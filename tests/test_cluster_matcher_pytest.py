"""Pytest tests for ClusterMatcher."""

import pytest
from unittest.mock import Mock, MagicMock
import uuid
from typing import List
from core_services.cluster_matcher import ClusterMatcher
from models import Topic, ClusterSnapshot, ClusterState, TopicPolicy, NegativeRules
from protocols import Storage


@pytest.fixture
def matcher():
    """Create ClusterMatcher instance with default settings."""
    return ClusterMatcher(tau_match=0.90, max_age_ticks=6)


@pytest.fixture
def matcher_low_threshold():
    """Create ClusterMatcher with lower threshold for testing."""
    return ClusterMatcher(tau_match=0.50, max_age_ticks=6)


@pytest.fixture
def topic():
    """Create test topic."""
    return Topic(
        id="test-topic-id",
        name="Test Topic",
        seeds=["test", "seeds"],
        negative=NegativeRules(),
        policy=TopicPolicy(),
        centroid_long=[0.1, 0.2, 0.3]
    )


@pytest.fixture
def cluster_snapshot():
    """Create test cluster snapshot."""
    return ClusterSnapshot(
        cluster_id="snapshot-1",
        centroid_now=[0.43, 0.53, 0.63],
        size=5,
        cohesion_now=0.8,
        separation_now=0.3,
        doc_ids=["doc1", "doc2", "doc3"]
    )


@pytest.fixture
def mock_storage():
    """Create mock storage with cluster states."""
    storage = Mock(spec=Storage)
    storage.cluster_state = {}
    storage.save_cluster_state = Mock()
    storage.delete_cluster_state = Mock()
    return storage


class TestClusterMatcherMatchOrCreate:
    """Test cases for ClusterMatcher.match_or_create() method."""
    
    def test_match_existing_high_similarity(self, matcher: ClusterMatcher, topic: Topic, cluster_snapshot: ClusterSnapshot, mock_storage: Mock):
        """Test matching existing state when similarity is above threshold."""
        # Create existing state with high similarity
        existing_state = ClusterState(
            cluster_id="existing-1",
            centroid_ema=[0.41, 0.51, 0.61],  # Very similar to snapshot
            cohesion_ema=0.7,
            separation_ema=0.2,
            persistence=3
        )
        mock_storage.cluster_state = {("test-topic-id", "existing-1"): existing_state}
        
        result = matcher.match_or_create(mock_storage, topic, cluster_snapshot)
        
        assert result == existing_state
        assert result.cluster_id == "existing-1"
        mock_storage.save_cluster_state.assert_not_called()
    
    def test_no_match_low_similarity(self, matcher: ClusterMatcher, topic: Topic, cluster_snapshot: ClusterSnapshot, mock_storage: Mock):
        """Test creating new state when no existing state has high enough similarity."""
        # Create existing state with low similarity
        existing_state = ClusterState(
            cluster_id="existing-1",
            centroid_ema=[1.0, 0.0, 0.0],  # Very different from snapshot [0.4, 0.5, 0.6]
            cohesion_ema=0.7,
            separation_ema=0.2,
            persistence=3
        )
        mock_storage.cluster_state = {("test-topic-id", "existing-1"): existing_state}
        
        result = matcher.match_or_create(mock_storage, topic, cluster_snapshot)
        
        # Should create new state
        assert result != existing_state
        assert result.cluster_id.startswith("cand_")
        assert len(result.cluster_id) == 13  # "cand_" + 8 hex chars
        mock_storage.save_cluster_state.assert_called_once()
    
    def test_no_existing_states(self, matcher: ClusterMatcher, topic: Topic, cluster_snapshot: ClusterSnapshot, mock_storage: Mock):
        """Test creating new state when no existing states exist."""
        mock_storage.cluster_state = {}
        
        result = matcher.match_or_create(mock_storage, topic, cluster_snapshot)
        
        assert result.cluster_id.startswith("cand_")
        assert len(result.cluster_id) == 13
        mock_storage.save_cluster_state.assert_called_once()
    
    def test_existing_state_no_centroid_ema(self, matcher: ClusterMatcher, topic: Topic, cluster_snapshot: ClusterSnapshot, mock_storage: Mock):
        """Test that states without centroid_ema are ignored."""
        # Create state without centroid_ema
        existing_state = ClusterState(
            cluster_id="existing-1",
            centroid_ema=None,  # No centroid
            cohesion_ema=0.7,
            separation_ema=0.2,
            persistence=3
        )
        mock_storage.cluster_state = {("test-topic-id", "existing-1"): existing_state}
        
        result = matcher.match_or_create(mock_storage, topic, cluster_snapshot)
        
        # Should create new state since existing has no centroid_ema
        assert result != existing_state
        assert result.cluster_id.startswith("cand_")
        mock_storage.save_cluster_state.assert_called_once()
    
    def test_multiple_states_best_match(self, matcher: ClusterMatcher, topic: Topic, cluster_snapshot: ClusterSnapshot, mock_storage: Mock):
        """Test that the best matching state is selected from multiple options."""
        # Create multiple states with different similarities
        state1 = ClusterState(
            cluster_id="state-1",
            centroid_ema=[0.9, 0.8, 0.7],  # Low similarity
            cohesion_ema=0.7,
            separation_ema=0.2,
            persistence=3
        )
        state2 = ClusterState(
            cluster_id="state-2", 
            centroid_ema=[0.41, 0.51, 0.61],  # High similarity
            cohesion_ema=0.8,
            separation_ema=0.1,
            persistence=5
        )
        state3 = ClusterState(
            cluster_id="state-3",
            centroid_ema=[0.5, 0.6, 0.7],  # Medium similarity
            cohesion_ema=0.6,
            separation_ema=0.3,
            persistence=2
        )
        
        mock_storage.cluster_state = {
            ("test-topic-id", "state-1"): state1,
            ("test-topic-id", "state-2"): state2,
            ("test-topic-id", "state-3"): state3,
        }
        
        result = matcher.match_or_create(mock_storage, topic, cluster_snapshot)
        
        # Should match state2 (highest similarity)
        assert result == state2
        assert result.cluster_id == "state-2"
        mock_storage.save_cluster_state.assert_not_called()
    
    def test_exact_threshold_match(self, matcher: ClusterMatcher, topic: Topic, cluster_snapshot: ClusterSnapshot, mock_storage: Mock):
        """Test behavior when similarity is exactly at threshold."""
        # Create state with similarity exactly at threshold
        existing_state = ClusterState(
            cluster_id="existing-1",
            centroid_ema=[0.4, 0.5, 0.6],  # Exact match (cosine = 1.0)
            cohesion_ema=0.7,
            separation_ema=0.2,
            persistence=3
        )
        mock_storage.cluster_state = {("test-topic-id", "existing-1"): existing_state}
        
        result = matcher.match_or_create(mock_storage, topic, cluster_snapshot)
        
        # Should match since similarity (1.0) >= threshold (0.90)
        assert result == existing_state
        mock_storage.save_cluster_state.assert_not_called()
    
    def test_just_below_threshold(self, matcher: ClusterMatcher, topic: Topic, cluster_snapshot: ClusterSnapshot, mock_storage: Mock):
        """Test behavior when similarity is just below threshold."""
        # Create state with similarity just below threshold
        existing_state = ClusterState(
            cluster_id="existing-1",
            centroid_ema=[0.0, 1.0, 0.0],  # Similar but not quite enough (cos ~0.57)
            cohesion_ema=0.7,
            separation_ema=0.2,
            persistence=3
        )
        mock_storage.cluster_state = {("test-topic-id", "existing-1"): existing_state}
        
        result = matcher.match_or_create(mock_storage, topic, cluster_snapshot)
        
        # Should create new state since similarity < threshold
        assert result != existing_state
        assert result.cluster_id.startswith("cand_")
        mock_storage.save_cluster_state.assert_called_once()
    
    def test_lower_threshold_matches(self, matcher_low_threshold: ClusterMatcher, topic: Topic, cluster_snapshot: ClusterSnapshot, mock_storage: Mock):
        """Test that lower threshold allows more matches."""
        # Create state with medium similarity
        existing_state = ClusterState(
            cluster_id="existing-1",
            centroid_ema=[0.35, 0.45, 0.55],  # Medium similarity
            cohesion_ema=0.7,
            separation_ema=0.2,
            persistence=3
        )
        mock_storage.cluster_state = {("test-topic-id", "existing-1"): existing_state}
        
        result = matcher_low_threshold.match_or_create(mock_storage, topic, cluster_snapshot)
        
        # Should match with lower threshold (0.50)
        assert result == existing_state
        mock_storage.save_cluster_state.assert_not_called()
    
    def test_new_state_properties(self, matcher: ClusterMatcher, topic: Topic, cluster_snapshot: ClusterSnapshot, mock_storage: Mock):
        """Test that new state has correct properties."""
        mock_storage.cluster_state = {}
        
        result = matcher.match_or_create(mock_storage, topic, cluster_snapshot)
        
        # Check new state properties
        assert isinstance(result, ClusterState)
        assert result.cluster_id.startswith("cand_")
        assert result.centroid_ema is None
        assert result.cohesion_ema == 0.0
        assert result.separation_ema == 0.0
        assert result.persistence == 0
        
        # Verify storage was called with correct parameters
        mock_storage.save_cluster_state.assert_called_once_with(topic.id, result)
    
    def test_different_topic_ids_isolated(self, matcher: ClusterMatcher, topic: Topic, cluster_snapshot: ClusterSnapshot, mock_storage: Mock):
        """Test that states from different topics are not considered."""
        # Create states for different topic
        other_topic_state = ClusterState(
            cluster_id="other-topic-state",
            centroid_ema=[0.41, 0.51, 0.61],  # High similarity
            cohesion_ema=0.7,
            separation_ema=0.2,
            persistence=3
        )
        mock_storage.cluster_state = {("other-topic-id", "other-topic-state"): other_topic_state}
        
        result = matcher.match_or_create(mock_storage, topic, cluster_snapshot)
        
        # Should create new state since other topic's states are ignored
        assert result != other_topic_state
        assert result.cluster_id.startswith("cand_")
        mock_storage.save_cluster_state.assert_called_once()
    
    @pytest.mark.parametrize("centroid_ema,expected_match", [
        ([0.4, 0.5, 0.6], True),      # Exact match (cos = 1.0)
        ([0.0, 1.0, 0.0], False),     # Below threshold (cos ~0.57)
        ([1.0, 0.0, 0.0], False),     # Very different (cos ~0.46)
        ([0.41, 0.51, 0.61], True),   # Very close (cos ~0.999)
    ])
    def test_similarity_thresholds(self, matcher: ClusterMatcher, topic: Topic, cluster_snapshot: ClusterSnapshot, mock_storage: Mock, centroid_ema: List[float], expected_match: bool):
        """Test various similarity thresholds."""
        existing_state = ClusterState(
            cluster_id="existing-1",
            centroid_ema=centroid_ema,
            cohesion_ema=0.7,
            separation_ema=0.2,
            persistence=3
        )
        mock_storage.cluster_state = {("test-topic-id", "existing-1"): existing_state}
        
        result = matcher.match_or_create(mock_storage, topic, cluster_snapshot)
        
        if expected_match:
            assert result == existing_state
            mock_storage.save_cluster_state.assert_not_called()
        else:
            assert result != existing_state
            assert result.cluster_id.startswith("cand_")
            mock_storage.save_cluster_state.assert_called_once()


class TestClusterMatcherListStates:
    """Test cases for ClusterMatcher.list_states() method."""
    
    def test_list_states_filters_by_topic_id(self, matcher: ClusterMatcher, mock_storage: Mock):
        """Test that list_states only returns states for the specified topic."""
        # Create states for different topics
        state1 = ClusterState(cluster_id="state-1")
        state2 = ClusterState(cluster_id="state-2")
        state3 = ClusterState(cluster_id="state-3")
        
        mock_storage.cluster_state = {
            ("topic-1", "state-1"): state1,
            ("topic-1", "state-2"): state2,
            ("topic-2", "state-3"): state3,
        }
        
        result = matcher.list_states(mock_storage, "topic-1")
        
        assert len(result) == 2
        assert state1 in result
        assert state2 in result
        assert state3 not in result
    
    def test_list_states_empty(self, matcher, mock_storage):
        """Test list_states when no states exist."""
        mock_storage.cluster_state = {}
        
        result = matcher.list_states(mock_storage, "topic-1")
        
        assert result == []


class TestClusterMatcherExpireStale:
    """Test cases for ClusterMatcher.expire_stale() method."""
    
    def test_expire_stale_removes_old_states(self, matcher: ClusterMatcher, mock_storage: Mock):
        """Test that stale states are removed."""
        import time
        
        now = time.time()
        state1 = ClusterState(cluster_id="state-1", last_seen_ts=now - 100)  # 100 seconds ago
        state2 = ClusterState(cluster_id="state-2", last_seen_ts=now - 30)   # 30 seconds ago
        state3 = ClusterState(cluster_id="state-3", last_seen_ts=now - 200)  # 200 seconds ago
        
        mock_storage.cluster_state = {
            ("topic-1", "state-1"): state1,
            ("topic-1", "state-2"): state2,
            ("topic-1", "state-3"): state3,
        }
        
        # Test with max_age_days=0.001 (about 86 seconds) - should remove state1 and state3
        matcher.expire_stale(mock_storage, "topic-1", max_age_days=0.001)
        
        # Should delete state-1 and state-3, keep state-2
        mock_storage.delete_cluster_state.assert_any_call("topic-1", "state-1")
        mock_storage.delete_cluster_state.assert_any_call("topic-1", "state-3")
        assert mock_storage.delete_cluster_state.call_count == 2
    
    def test_expire_stale_keeps_recent_states(self, matcher: ClusterMatcher, mock_storage: Mock):
        """Test that recent states are not removed."""
        import time
        
        now = time.time()
        state1 = ClusterState(cluster_id="state-1", last_seen_ts=now - 30)   # 30 seconds ago
        state2 = ClusterState(cluster_id="state-2", last_seen_ts=now - 10)   # 10 seconds ago
        
        mock_storage.cluster_state = {
            ("topic-1", "state-1"): state1,
            ("topic-1", "state-2"): state2,
        }
        
        # Test with max_age_days=1 (86400 seconds) - should keep both states
        matcher.expire_stale(mock_storage, "topic-1", max_age_days=1)
        
        # Should not delete any states
        mock_storage.delete_cluster_state.assert_not_called()
    
    def test_expire_stale_different_topic(self, matcher: ClusterMatcher, mock_storage: Mock):
        """Test that only states for specified topic are considered for expiry."""
        import time
        
        now = time.time()
        state1 = ClusterState(cluster_id="state-1", last_seen_ts=now - 100)  # 100 seconds ago
        state2 = ClusterState(cluster_id="state-2", last_seen_ts=now - 100)  # 100 seconds ago
        
        mock_storage.cluster_state = {
            ("topic-1", "state-1"): state1,
            ("topic-2", "state-2"): state2,
        }
        
        # Test with max_age_days=0.001 (about 86 seconds) - should only remove state-1
        matcher.expire_stale(mock_storage, "topic-1", max_age_days=0.001)
        
        # Should only delete state-1 (from topic-1)
        mock_storage.delete_cluster_state.assert_called_once_with("topic-1", "state-1")
