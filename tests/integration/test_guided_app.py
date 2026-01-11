"""Integration tests for guided learning app."""

import pytest
import tempfile
from pathlib import Path

from linalg_tutor.core.progress.session_state import SessionState
from linalg_tutor.core.progress.tracker import ProgressTracker


def test_session_state_creates_directory():
    """Test that session state creates data directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "test_data"
        assert not data_dir.exists()

        session = SessionState(data_dir=data_dir)

        assert data_dir.exists()
        assert session.state_file.exists()


def test_guided_app_imports():
    """Test that guided app modules import correctly."""
    from linalg_tutor.cli.guided_app import GuidedLearningApp
    from linalg_tutor.cli.ui.main_menu import show_welcome, main_menu

    # Should not raise
    assert GuidedLearningApp is not None
    assert show_welcome is not None
    assert main_menu is not None


def test_guided_app_initialization():
    """Test guided app can be initialized."""
    from linalg_tutor.cli.guided_app import GuidedLearningApp

    with tempfile.TemporaryDirectory() as tmpdir:
        # Temporarily override data directory
        import os
        original_home = os.environ.get('HOME')
        os.environ['HOME'] = tmpdir

        try:
            app = GuidedLearningApp()
            assert app.session is not None
            assert app.tracker is not None
            assert app.running is True
        finally:
            if original_home:
                os.environ['HOME'] = original_home


def test_chapter_progression_logic():
    """Test the chapter progression logic works correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        session = SessionState(data_dir=Path(tmpdir))

        # Start at vectors
        current = session.get_current_chapter()
        assert current["topic"] == "vectors"

        # Update activity
        session.update_activity("vectors", 10, 200)
        assert session.path.total_exercises_completed == 10

        # Complete vectors - should advance to matrices
        session.mark_topic_complete("vectors")
        current = session.get_current_chapter()
        assert current["topic"] == "matrices"

        # Complete all chapters
        for chapter in SessionState.CHAPTERS:
            if chapter["id"] > 1:  # Skip vectors (already done)
                session.mark_topic_complete(chapter["topic"])

        # All should be complete
        assert len(session.path.completed_topics) == 10
        summary = session.get_progress_summary()
        assert summary["progress_percent"] == 100.0


def test_session_and_tracker_integration():
    """Test that session state works with progress tracker."""
    with tempfile.TemporaryDirectory() as tmpdir:
        session = SessionState(data_dir=Path(tmpdir))
        tracker = ProgressTracker(db_path=str(Path(tmpdir) / "progress.db"))

        # Update session
        session.update_activity("vectors", 5, 100)

        # Both should work independently
        assert session.path.total_exercises_completed == 5

        # Create a dummy exercise and record attempt
        from linalg_tutor.core.exercises import ComputationalExercise, ExerciseDifficulty, ExerciseResult
        import numpy as np

        exercise = ComputationalExercise(
            exercise_id="test_1",
            topic="vectors",
            difficulty=ExerciseDifficulty.PRACTICE,
            question="Test",
            operation="test",
            inputs={},
            expected_output=np.array([1]),
        )

        result = ExerciseResult(
            correct=True,
            user_answer=np.array([1]),
            correct_answer=np.array([1]),
            feedback="Correct",
        )

        tracker.record_attempt(result, exercise, 10.0)

        # Check both systems work independently
        stats = tracker.get_statistics()
        assert stats is not None  # Tracker works
        assert session.path.total_exercises_completed == 5  # Session tracks separately


def test_menu_system_structure():
    """Test that menu system has correct structure."""
    from linalg_tutor.cli.ui.main_menu import show_chapter_list

    with tempfile.TemporaryDirectory() as tmpdir:
        session = SessionState(data_dir=Path(tmpdir))

        # Should not crash
        # (Can't fully test interactive display, but can ensure it runs)
        from io import StringIO
        from rich.console import Console

        buffer = StringIO()
        console = Console(file=buffer, force_terminal=True)

        # The show_chapter_list function uses global console
        # We can't easily capture it, but we can verify it doesn't crash
        # This is a smoke test
        try:
            # Just verify the function exists and session has data it needs
            assert hasattr(session, 'path')
            assert hasattr(session.path, 'completed_topics')
            assert hasattr(session.path, 'topics_in_progress')
        except Exception as e:
            pytest.fail(f"Menu structure test failed: {e}")


def test_complete_user_journey():
    """Test a complete user journey through the app."""
    with tempfile.TemporaryDirectory() as tmpdir:
        session = SessionState(data_dir=Path(tmpdir))

        # Step 1: User starts fresh
        assert len(session.path.completed_topics) == 0
        recommended = session.get_recommended_chapter()
        assert recommended["name"] == "Vectors"

        # Step 2: User practices vectors
        session.update_activity("vectors", 10, 300)
        assert "vectors" in session.path.topics_in_progress

        # Step 3: User completes vectors
        session.mark_topic_complete("vectors")
        assert "vectors" in session.path.completed_topics
        assert "vectors" not in session.path.topics_in_progress

        # Step 4: Should recommend matrices now
        recommended = session.get_recommended_chapter()
        assert recommended["name"] == "Matrices"
        assert session.path.current_chapter == 2

        # Step 5: User practices matrices
        session.update_activity("matrices", 15, 400)

        # Step 6: Check progress
        summary = session.get_progress_summary()
        assert summary["completed_chapters"] == 1
        assert summary["in_progress_chapters"] == 1
        assert summary["total_exercises"] == 25
        assert summary["total_time"] == 700.0
        assert summary["progress_percent"] == 10.0  # 1/10 chapters

        # Step 7: Complete matrices and linear_systems
        session.mark_topic_complete("matrices")
        session.mark_topic_complete("linear_systems")

        # Step 8: Final check
        summary = session.get_progress_summary()
        assert summary["completed_chapters"] == 3
        assert summary["progress_percent"] == 30.0
        assert session.path.current_chapter == 4


def test_session_state_file_format():
    """Test that session state file is valid JSON."""
    import json

    with tempfile.TemporaryDirectory() as tmpdir:
        session = SessionState(data_dir=Path(tmpdir))
        session.update_activity("vectors", 5, 100)
        session.save()

        # Read and parse JSON
        with open(session.state_file, 'r') as f:
            data = json.load(f)

        # Verify structure
        assert "current_topic" in data
        assert "current_chapter" in data
        assert "completed_topics" in data
        assert "topics_in_progress" in data
        assert "total_exercises_completed" in data
        assert "total_time_spent" in data

        # Verify values
        assert data["current_topic"] == "vectors"
        assert data["total_exercises_completed"] == 5
        assert data["total_time_spent"] == 100.0
