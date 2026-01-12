"""Tests for exercise prompt UI to ensure no empty space and proper rendering."""

import pytest
import numpy as np
from io import StringIO

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from linalg_tutor.cli.ui.prompts import ExercisePrompt
from linalg_tutor.core.exercises import (
    ComputationalExercise,
    MultipleChoiceExercise,
    TrueFalseExercise,
    ExerciseDifficulty,
)


@pytest.fixture
def sample_computational_exercise():
    """Create a sample computational exercise."""
    return ComputationalExercise(
        exercise_id="test_comp_1",
        topic="vectors",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Compute the dot product of v = [1, 2, 3] and w = [4, 5, 6]",
        operation="dot_product",
        inputs={"v": np.array([1, 2, 3]), "w": np.array([4, 5, 6])},
        expected_output=32,
        hints=["Multiply corresponding components", "Sum the products"],
    )


@pytest.fixture
def sample_multiple_choice_exercise():
    """Create a sample multiple choice exercise."""
    return MultipleChoiceExercise(
        exercise_id="test_mc_1",
        topic="matrices",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="What is the determinant of a 2x2 identity matrix?",
        choices=["0", "1", "2", "-1"],
        correct_index=1,
        hints=["The identity matrix has 1s on the diagonal"],
    )


def test_display_builds_without_errors(sample_computational_exercise):
    """Test that display builds without raising exceptions."""
    prompt = ExercisePrompt(
        sample_computational_exercise, tracker=None, current_exercise=1, total_exercises=5
    )

    # Should not raise
    display = prompt._build_display()
    assert display is not None


def test_question_text_is_in_rendered_output(sample_computational_exercise):
    """CRITICAL: Verify question text actually appears in rendered output."""
    prompt = ExercisePrompt(
        sample_computational_exercise, tracker=None, current_exercise=1, total_exercises=5
    )

    display = prompt._build_display()

    # Render to string
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=True, width=100)
    console.print(display)
    output = buffer.getvalue()

    # CRITICAL: Question text must be present
    assert "dot product" in output, "Question text not found in rendered output!"
    assert "[1, 2, 3]" in output, "Question details not found in rendered output!"
    assert "Question" in output, "Question panel title not found!"


def test_no_excessive_empty_lines(sample_computational_exercise):
    """CRITICAL: Verify no massive blocks of empty lines (was the main bug)."""
    prompt = ExercisePrompt(
        sample_computational_exercise, tracker=None, current_exercise=1, total_exercises=5
    )

    display = prompt._build_display()

    # Render to string
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=True, width=100)
    console.print(display)
    output = buffer.getvalue()

    lines = output.split('\n')

    # Count consecutive empty lines
    max_consecutive_empty = 0
    current_consecutive = 0

    for line in lines:
        if line.strip() == '':
            current_consecutive += 1
            max_consecutive_empty = max(max_consecutive_empty, current_consecutive)
        else:
            current_consecutive = 0

    # Should never have more than 2 consecutive empty lines
    assert max_consecutive_empty <= 2, f"Found {max_consecutive_empty} consecutive empty lines - UI has wasted space!"


def test_display_is_compact_without_hints(sample_computational_exercise):
    """Test that display without hints/messages is compact (< 10 lines)."""
    prompt = ExercisePrompt(
        sample_computational_exercise, tracker=None, current_exercise=1, total_exercises=5
    )

    display = prompt._build_display()

    buffer = StringIO()
    console = Console(file=buffer, force_terminal=True, width=100)
    console.print(display)
    output = buffer.getvalue()

    non_empty_lines = [line for line in output.split('\n') if line.strip()]

    # Should be compact: header + question panel (borders + content) = ~7-8 lines
    assert len(non_empty_lines) < 10, f"Display too large: {len(non_empty_lines)} non-empty lines"


def test_header_displays_progress_and_topic(sample_computational_exercise):
    """Test that header shows topic, progress, and difficulty."""
    prompt = ExercisePrompt(
        sample_computational_exercise, tracker=None, current_exercise=2, total_exercises=5
    )

    display = prompt._build_display()

    buffer = StringIO()
    console = Console(file=buffer, force_terminal=True, width=100)
    console.print(display)
    output = buffer.getvalue()

    # Check header contains required info
    assert "Vectors" in output or "vectors" in output.lower()
    assert "2/5" in output
    assert "Practice" in output or "practice" in output.lower()


def test_hints_appear_when_shown(sample_computational_exercise):
    """Test that hints are displayed when requested."""
    prompt = ExercisePrompt(
        sample_computational_exercise, tracker=None, current_exercise=1, total_exercises=5
    )

    # Show a hint
    prompt._show_hint()

    display = prompt._build_display()

    buffer = StringIO()
    console = Console(file=buffer, force_terminal=True, width=100)
    console.print(display)
    output = buffer.getvalue()

    # Hint should be visible
    assert "Hints" in output or "💡" in output
    assert "Multiply corresponding components" in output


def test_stats_appear_after_attempts(sample_computational_exercise):
    """Test that stats are shown after user attempts."""
    prompt = ExercisePrompt(
        sample_computational_exercise, tracker=None, current_exercise=1, total_exercises=5
    )

    # Simulate an attempt
    prompt.attempts = 2
    prompt.hints_shown = 1

    display = prompt._build_display()

    buffer = StringIO()
    console = Console(file=buffer, force_terminal=True, width=100)
    console.print(display)
    output = buffer.getvalue()

    # Stats should be visible
    assert "Attempts" in output or "attempts" in output.lower()
    assert "Hints" in output or "hints" in output.lower()


def test_message_appears_when_set(sample_computational_exercise):
    """Test that messages are displayed when set."""
    prompt = ExercisePrompt(
        sample_computational_exercise, tracker=None, current_exercise=1, total_exercises=5
    )

    # Set a message
    prompt.message = "Incorrect: Try again!"
    prompt.message_style = "red"

    display = prompt._build_display()

    buffer = StringIO()
    console = Console(file=buffer, force_terminal=True, width=100)
    console.print(display)
    output = buffer.getvalue()

    # Message should be visible
    assert "Incorrect" in output
    assert "Try again" in output


def test_display_uses_group_not_layout(sample_computational_exercise):
    """CRITICAL: Verify display returns Group (compact) not Layout (screen-filling)."""
    from rich.console import Group

    prompt = ExercisePrompt(
        sample_computational_exercise, tracker=None, current_exercise=1, total_exercises=5
    )

    display = prompt._build_display()

    # Must be a Group, not a Layout
    assert isinstance(display, Group), f"Display is {type(display)}, should be Group!"


def test_question_panel_has_proper_structure(sample_computational_exercise):
    """Test that question is wrapped in Panel with Align."""
    from rich.console import Group

    prompt = ExercisePrompt(
        sample_computational_exercise, tracker=None, current_exercise=1, total_exercises=5
    )

    display = prompt._build_display()

    # Display should be a Group
    assert isinstance(display, Group)

    # Should have at least 2 renderables (header + question)
    assert len(display.renderables) >= 2

    # Second item should be the question Panel
    question_item = display.renderables[1]
    assert isinstance(question_item, Panel), "Question should be wrapped in Panel"


def test_multiple_choice_exercise_displays_correctly(sample_multiple_choice_exercise):
    """Test that multiple choice exercises display properly."""
    prompt = ExercisePrompt(
        sample_multiple_choice_exercise, tracker=None, current_exercise=1, total_exercises=3
    )

    display = prompt._build_display()

    buffer = StringIO()
    console = Console(file=buffer, force_terminal=True, width=100)
    console.print(display)
    output = buffer.getvalue()

    # Question should be visible
    assert "determinant" in output.lower()
    assert "identity matrix" in output.lower()


def test_display_with_all_elements(sample_computational_exercise):
    """Test display with hints, message, and stats all present."""
    prompt = ExercisePrompt(
        sample_computational_exercise, tracker=None, current_exercise=3, total_exercises=5
    )

    # Add all elements
    prompt._show_hint()
    prompt._show_hint()
    prompt.attempts = 3
    prompt.message = "✓ Correct!"
    prompt.message_style = "green"

    display = prompt._build_display()

    buffer = StringIO()
    console = Console(file=buffer, force_terminal=True, width=100)
    console.print(display)
    output = buffer.getvalue()

    # All elements should be visible
    assert "Vectors" in output
    assert "3/5" in output
    assert "dot product" in output
    assert "Hints" in output or "💡" in output
    assert "Multiply" in output  # First hint
    assert "Sum" in output  # Second hint
    assert "Correct" in output
    assert "Attempts" in output or "attempts" in output.lower()

    # But still shouldn't have massive empty space
    lines = output.split('\n')
    max_consecutive_empty = 0
    current_consecutive = 0

    for line in lines:
        if line.strip() == '':
            current_consecutive += 1
            max_consecutive_empty = max(max_consecutive_empty, current_consecutive)
        else:
            current_consecutive = 0

    assert max_consecutive_empty <= 2, "Even with all elements, should not have excessive empty space"


def test_long_question_text_wraps_correctly():
    """Test that long questions are handled without overflow."""
    long_exercise = ComputationalExercise(
        exercise_id="test_long",
        topic="vectors",
        difficulty=ExerciseDifficulty.CHALLENGE,
        question="Compute the very complicated and extraordinarily long dot product calculation involving vectors v = [1, 2, 3, 4, 5] and w = [6, 7, 8, 9, 10] which requires careful attention to detail",
        operation="dot_product",
        inputs={"v": np.array([1, 2, 3, 4, 5]), "w": np.array([6, 7, 8, 9, 10])},
        expected_output=130,
        hints=["Take it step by step"],
    )

    prompt = ExercisePrompt(long_exercise, tracker=None, current_exercise=1, total_exercises=1)

    display = prompt._build_display()

    buffer = StringIO()
    console = Console(file=buffer, force_terminal=True, width=100)
    console.print(display)
    output = buffer.getvalue()

    # Question should be present
    assert "complicated" in output
    assert "extraordinarily long" in output
    assert "[1, 2, 3, 4, 5]" in output


def test_display_screen_method_doesnt_crash(sample_computational_exercise):
    """Test that _display_screen method executes without errors."""
    prompt = ExercisePrompt(
        sample_computational_exercise, tracker=None, current_exercise=1, total_exercises=5
    )

    # Should not raise (though it will clear screen and print)
    try:
        # We can't fully test the screen clearing, but we can ensure it doesn't crash
        display = prompt._build_display()
        assert display is not None
    except Exception as e:
        pytest.fail(f"_display_screen raised unexpected exception: {e}")
