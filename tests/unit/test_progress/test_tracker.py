"""Tests for progress tracking."""

import pytest
from linalg_tutor.core.exercises import ExerciseResult
from linalg_tutor.core.progress import ProgressTracker


def test_progress_tracker_record_attempt(progress_tracker, sample_vector_add_exercise):
    """Test recording an exercise attempt."""
    result = ExerciseResult(
        correct=True,
        user_answer=[4, 6],
        correct_answer=[4, 6],
        feedback="Correct!",
        hints_used=0,
        time_spent=10.5,
    )

    progress_tracker.record_attempt(result, sample_vector_add_exercise, time_spent=10.5)

    # Check that progress was recorded
    topic_progress = progress_tracker.get_topic_progress("vectors")
    assert topic_progress is not None
    assert topic_progress.exercises_attempted == 1
    assert topic_progress.exercises_correct == 1


def test_progress_tracker_multiple_attempts(progress_tracker, sample_vector_add_exercise):
    """Test recording multiple attempts."""
    # First attempt - correct
    result1 = ExerciseResult(
        correct=True,
        user_answer=[4, 6],
        correct_answer=[4, 6],
        feedback="Correct!",
        hints_used=0,
    )
    progress_tracker.record_attempt(result1, sample_vector_add_exercise)

    # Second attempt - incorrect
    result2 = ExerciseResult(
        correct=False,
        user_answer=[5, 6],
        correct_answer=[4, 6],
        feedback="Incorrect",
        hints_used=1,
    )
    progress_tracker.record_attempt(result2, sample_vector_add_exercise)

    # Check progress
    topic_progress = progress_tracker.get_topic_progress("vectors")
    assert topic_progress.exercises_attempted == 2
    assert topic_progress.exercises_correct == 1


def test_progress_tracker_mastery_calculation(progress_tracker, sample_vector_add_exercise):
    """Test mastery level calculation."""
    # Record some attempts
    for i in range(10):
        result = ExerciseResult(
            correct=(i % 2 == 0),  # 50% accuracy
            user_answer=[4, 6],
            correct_answer=[4, 6],
            feedback="Test",
            hints_used=0,
        )
        progress_tracker.record_attempt(result, sample_vector_add_exercise)

    mastery = progress_tracker.get_mastery_level("vectors")
    assert 0.0 <= mastery <= 1.0
    # With 50% accuracy, mastery should be moderate
    assert 0.3 <= mastery <= 0.7


def test_progress_tracker_statistics(progress_tracker, sample_vector_add_exercise):
    """Test overall statistics."""
    # Record some attempts
    for i in range(5):
        result = ExerciseResult(
            correct=True,
            user_answer=[4, 6],
            correct_answer=[4, 6],
            feedback="Correct!",
            hints_used=0,
            time_spent=5.0,
        )
        progress_tracker.record_attempt(result, sample_vector_add_exercise, time_spent=5.0)

    stats = progress_tracker.get_statistics()

    assert stats["total_attempts"] == 5
    assert stats["total_correct"] == 5
    assert stats["accuracy"] == 1.0
    assert stats["topics_started"] == 1


def test_progress_tracker_recommend_next_topic(progress_tracker):
    """Test topic recommendation."""
    next_topic = progress_tracker.recommend_next_topic()

    # Should recommend "vectors" for a new user
    assert next_topic == "vectors"


def test_progress_tracker_context_manager(sample_vector_add_exercise):
    """Test using tracker as context manager."""
    with ProgressTracker(db_path=":memory:") as tracker:
        result = ExerciseResult(
            correct=True,
            user_answer=[4, 6],
            correct_answer=[4, 6],
            feedback="Correct!",
            hints_used=0,
        )
        tracker.record_attempt(result, sample_vector_add_exercise)

    # Tracker should be closed after context
    # Accessing the session after close should raise an error or be safe
    assert tracker.session is not None
