"""Tests for session state management."""

import pytest
import tempfile
from pathlib import Path

from linalg_tutor.core.progress.session_state import SessionState, LearningPath


@pytest.fixture
def temp_session():
    """Create a session with temporary data directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        session = SessionState(data_dir=Path(tmpdir))
        yield session


def test_session_state_initialization(temp_session):
    """Test session state initializes correctly."""
    assert temp_session.path is not None
    assert temp_session.path.current_topic is None
    assert temp_session.path.current_chapter == 1
    assert len(temp_session.path.completed_topics) == 0
    assert len(temp_session.path.topics_in_progress) == 0


def test_session_state_chapters():
    """Test chapter list is correct."""
    assert len(SessionState.CHAPTERS) == 10
    assert SessionState.CHAPTERS[0]["name"] == "Vectors"
    assert SessionState.CHAPTERS[0]["topic"] == "vectors"
    assert SessionState.CHAPTERS[9]["name"] == "Applications"


def test_get_current_chapter(temp_session):
    """Test getting current chapter."""
    current = temp_session.get_current_chapter()
    assert current is not None
    assert current["id"] == 1
    assert current["name"] == "Vectors"


def test_get_next_chapter(temp_session):
    """Test getting next chapter."""
    # Start at chapter 1
    next_ch = temp_session.get_next_chapter()
    assert next_ch["id"] == 2
    assert next_ch["name"] == "Matrices"

    # Move to chapter 5
    temp_session.path.current_chapter = 5
    next_ch = temp_session.get_next_chapter()
    assert next_ch["id"] == 6
    assert next_ch["name"] == "Determinants"

    # At last chapter
    temp_session.path.current_chapter = 10
    next_ch = temp_session.get_next_chapter()
    assert next_ch is None  # No more chapters


def test_update_activity(temp_session):
    """Test updating activity."""
    temp_session.update_activity("vectors", exercises_completed=5, time_spent=120.5)

    assert temp_session.path.current_topic == "vectors"
    assert temp_session.path.total_exercises_completed == 5
    assert temp_session.path.total_time_spent == 120.5
    assert "vectors" in temp_session.path.topics_in_progress
    assert temp_session.path.last_activity is not None


def test_mark_topic_complete(temp_session):
    """Test marking topic as complete."""
    # Add to in-progress first
    temp_session.update_activity("vectors", 10, 200)
    assert "vectors" in temp_session.path.topics_in_progress

    # Mark complete
    temp_session.mark_topic_complete("vectors")

    assert "vectors" in temp_session.path.completed_topics
    assert "vectors" not in temp_session.path.topics_in_progress
    # Should advance to next chapter
    assert temp_session.path.current_chapter == 2
    assert temp_session.path.current_topic == "matrices"


def test_get_recommended_chapter(temp_session):
    """Test getting recommended chapter."""
    # No progress - should recommend first
    recommended = temp_session.get_recommended_chapter()
    assert recommended["id"] == 1
    assert recommended["name"] == "Vectors"

    # Start vectors
    temp_session.update_activity("vectors", 5, 100)
    recommended = temp_session.get_recommended_chapter()
    assert recommended["name"] == "Vectors"  # Continue current

    # Complete vectors
    temp_session.mark_topic_complete("vectors")
    recommended = temp_session.get_recommended_chapter()
    assert recommended["id"] == 2
    assert recommended["name"] == "Matrices"


def test_get_progress_summary(temp_session):
    """Test progress summary."""
    summary = temp_session.get_progress_summary()

    assert summary["total_chapters"] == 10
    assert summary["completed_chapters"] == 0
    assert summary["in_progress_chapters"] == 0
    assert summary["progress_percent"] == 0
    assert summary["total_exercises"] == 0
    assert summary["total_time"] == 0.0

    # Add some progress
    temp_session.update_activity("vectors", 15, 300.5)
    temp_session.mark_topic_complete("vectors")
    temp_session.update_activity("matrices", 10, 200)

    summary = temp_session.get_progress_summary()
    assert summary["completed_chapters"] == 1
    assert summary["in_progress_chapters"] == 1
    assert summary["progress_percent"] == 10.0  # 1/10 * 100
    assert summary["total_exercises"] == 25
    assert summary["total_time"] == 500.5


def test_session_state_persistence(temp_session):
    """Test session state saves and loads correctly."""
    # Update state
    temp_session.update_activity("vectors", 20, 400)
    temp_session.mark_topic_complete("vectors")

    # Create new session with same data dir
    new_session = SessionState(data_dir=temp_session.data_dir)

    # Check state persisted
    assert new_session.path.current_topic == "matrices"
    assert new_session.path.current_chapter == 2
    assert "vectors" in new_session.path.completed_topics
    assert new_session.path.total_exercises_completed == 20
    assert new_session.path.total_time_spent == 400.0


def test_reset(temp_session):
    """Test resetting progress."""
    # Add progress
    temp_session.update_activity("vectors", 50, 1000)
    temp_session.mark_topic_complete("vectors")

    # Reset
    temp_session.reset()

    # Check everything cleared
    assert temp_session.path.current_topic is None
    assert temp_session.path.current_chapter == 1
    assert len(temp_session.path.completed_topics) == 0
    assert len(temp_session.path.topics_in_progress) == 0
    assert temp_session.path.total_exercises_completed == 0
    assert temp_session.path.total_time_spent == 0.0


def test_multiple_topics_in_progress(temp_session):
    """Test handling multiple topics in progress."""
    temp_session.update_activity("vectors", 5, 100)
    temp_session.update_activity("matrices", 3, 60)
    temp_session.update_activity("linear_systems", 2, 40)

    assert len(temp_session.path.topics_in_progress) == 3
    assert "vectors" in temp_session.path.topics_in_progress
    assert "matrices" in temp_session.path.topics_in_progress
    assert "linear_systems" in temp_session.path.topics_in_progress


def test_chapter_completion_advancement(temp_session):
    """Test that completing current chapter advances to next."""
    # Start at chapter 1
    temp_session.path.current_chapter = 1
    temp_session.path.current_topic = "vectors"

    # Complete it
    temp_session.mark_topic_complete("vectors")

    # Should advance
    assert temp_session.path.current_chapter == 2
    assert temp_session.path.current_topic == "matrices"

    # Complete multiple in a row
    temp_session.mark_topic_complete("matrices")
    assert temp_session.path.current_chapter == 3

    temp_session.mark_topic_complete("linear_systems")
    assert temp_session.path.current_chapter == 4
