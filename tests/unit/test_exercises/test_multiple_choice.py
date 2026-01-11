"""Tests for multiple choice exercises."""

import pytest


def test_multiple_choice_correct_index(sample_multiple_choice_exercise):
    """Test correct answer by index."""
    result = sample_multiple_choice_exercise.check_answer(0)

    assert result.correct is True
    assert "Correct" in result.feedback


def test_multiple_choice_wrong_index(sample_multiple_choice_exercise):
    """Test wrong answer by index."""
    result = sample_multiple_choice_exercise.check_answer(1)

    assert result.correct is False
    assert "Incorrect" in result.feedback


def test_multiple_choice_by_text(sample_multiple_choice_exercise):
    """Test answer by choice text."""
    result = sample_multiple_choice_exercise.check_answer("0")

    assert result.correct is True


def test_multiple_choice_invalid_index(sample_multiple_choice_exercise):
    """Test invalid index handling."""
    result = sample_multiple_choice_exercise.check_answer(10)

    assert result.correct is False
    assert "out of range" in result.feedback.lower()


def test_multiple_choice_get_correct_answer(sample_multiple_choice_exercise):
    """Test getting correct answer."""
    answer = sample_multiple_choice_exercise.get_correct_answer()

    assert answer == "0"


def test_multiple_choice_get_solution(sample_multiple_choice_exercise):
    """Test solution includes explanation."""
    solution = sample_multiple_choice_exercise.get_solution()

    assert "0" in solution
    assert "Explanation" in solution
