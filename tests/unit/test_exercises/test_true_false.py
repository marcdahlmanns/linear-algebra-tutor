"""Tests for true/false exercises."""

import pytest


def test_true_false_correct_boolean(sample_true_false_exercise):
    """Test correct answer with boolean."""
    result = sample_true_false_exercise.check_answer(False)

    assert result.correct is True
    assert "Correct" in result.feedback


def test_true_false_wrong_boolean(sample_true_false_exercise):
    """Test wrong answer with boolean."""
    result = sample_true_false_exercise.check_answer(True)

    assert result.correct is False
    assert "Incorrect" in result.feedback


def test_true_false_string_input(sample_true_false_exercise):
    """Test string input variants."""
    # Test various string representations
    result1 = sample_true_false_exercise.check_answer("false")
    assert result1.correct is True

    result2 = sample_true_false_exercise.check_answer("False")
    assert result2.correct is True

    result3 = sample_true_false_exercise.check_answer("f")
    assert result3.correct is True

    result4 = sample_true_false_exercise.check_answer("no")
    assert result4.correct is True


def test_true_false_integer_input(sample_true_false_exercise):
    """Test integer input (0 = False, 1 = True)."""
    result = sample_true_false_exercise.check_answer(0)

    assert result.correct is True


def test_true_false_invalid_string(sample_true_false_exercise):
    """Test invalid string input."""
    result = sample_true_false_exercise.check_answer("maybe")

    assert result.correct is False
    assert "Invalid" in result.feedback


def test_true_false_get_correct_answer(sample_true_false_exercise):
    """Test getting correct answer."""
    answer = sample_true_false_exercise.get_correct_answer()

    assert answer is False


def test_true_false_get_solution(sample_true_false_exercise):
    """Test solution includes explanation."""
    solution = sample_true_false_exercise.get_solution()

    assert "False" in solution
    assert "Explanation" in solution
