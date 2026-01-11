"""Tests for computational exercises."""

import pytest
import numpy as np
from linalg_tutor.core.exercises import ComputationalExercise, ExerciseDifficulty


def test_computational_exercise_correct_answer(sample_vector_add_exercise):
    """Test correct answer validation."""
    result = sample_vector_add_exercise.check_answer([4, 6])

    assert result.correct is True
    assert result.feedback == "Correct!"


def test_computational_exercise_wrong_answer(sample_vector_add_exercise):
    """Test wrong answer detection."""
    result = sample_vector_add_exercise.check_answer([5, 6])

    assert result.correct is False
    assert "Incorrect" in result.feedback


def test_computational_exercise_wrong_shape(sample_vector_add_exercise):
    """Test wrong shape detection."""
    result = sample_vector_add_exercise.check_answer([4, 6, 8])

    assert result.correct is False
    assert "Shape mismatch" in result.feedback


def test_computational_exercise_numpy_array_input(sample_vector_add_exercise):
    """Test numpy array input."""
    result = sample_vector_add_exercise.check_answer(np.array([4, 6]))

    assert result.correct is True


def test_computational_exercise_numerical_tolerance():
    """Test numerical tolerance in comparisons."""
    exercise = ComputationalExercise(
        exercise_id="test_tol",
        topic="test",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Test",
        operation="test",
        inputs={},
        expected_output=np.array([1.0, 2.0]),
        tolerance=1e-5,
    )

    # Slightly different values within tolerance
    result = exercise.check_answer([1.0000001, 2.0000001])
    assert result.correct is True

    # Values outside tolerance
    result = exercise.check_answer([1.1, 2.0])
    assert result.correct is False


def test_computational_exercise_get_correct_answer(sample_vector_add_exercise):
    """Test getting correct answer."""
    answer = sample_vector_add_exercise.get_correct_answer()

    assert np.array_equal(answer, np.array([4, 6]))


def test_computational_exercise_matrix_multiplication(sample_matrix_multiply_exercise):
    """Test matrix multiplication exercise."""
    # Correct answer
    correct_answer = [[19, 22], [43, 50]]
    result = sample_matrix_multiply_exercise.check_answer(correct_answer)

    assert result.correct is True


def test_computational_exercise_get_solution(sample_vector_add_exercise):
    """Test solution generation."""
    solution = sample_vector_add_exercise.get_solution(show_steps=True)

    assert "Solution" in solution or "Step" in solution
    assert len(solution) > 0
