"""Pytest configuration and fixtures."""

import pytest
import numpy as np
from linalg_tutor.core.exercises import (
    ComputationalExercise,
    MultipleChoiceExercise,
    TrueFalseExercise,
    FillInExercise,
    ExerciseDifficulty,
)
from linalg_tutor.core.progress import ProgressTracker


@pytest.fixture
def sample_vector_add_exercise():
    """Create a sample vector addition exercise."""
    return ComputationalExercise(
        exercise_id="test_vec_add",
        topic="vectors",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Add v = [1, 2] and w = [3, 4]",
        operation="vector_add",
        inputs={"v": np.array([1, 2]), "w": np.array([3, 4])},
        expected_output=np.array([4, 6]),
        hints=["Add component-wise"],
    )


@pytest.fixture
def sample_matrix_multiply_exercise():
    """Create a sample matrix multiplication exercise."""
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    C = A @ B

    return ComputationalExercise(
        exercise_id="test_mat_mul",
        topic="matrices",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Multiply A and B",
        operation="matrix_multiply",
        inputs={"A": A, "B": B},
        expected_output=C,
    )


@pytest.fixture
def sample_multiple_choice_exercise():
    """Create a sample multiple choice exercise."""
    return MultipleChoiceExercise(
        exercise_id="test_mc",
        topic="vectors",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="What is the dot product of orthogonal vectors?",
        choices=["0", "1", "-1", "undefined"],
        correct_index=0,
        explanation="Orthogonal vectors have a dot product of zero.",
    )


@pytest.fixture
def sample_true_false_exercise():
    """Create a sample true/false exercise."""
    return TrueFalseExercise(
        exercise_id="test_tf",
        topic="matrices",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Matrix multiplication is commutative (AB = BA for all matrices A and B)",
        correct_answer=False,
        explanation="Matrix multiplication is NOT commutative in general.",
    )


@pytest.fixture
def sample_fill_in_exercise():
    """Create a sample fill-in exercise."""
    return FillInExercise(
        exercise_id="test_fill",
        topic="vectors",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="The magnitude of vector [3, 4] is ___",
        answer_type="numerical",
        correct_answer=5.0,
    )


@pytest.fixture
def progress_tracker():
    """Create a progress tracker with in-memory database."""
    tracker = ProgressTracker(db_path=":memory:")
    yield tracker
    tracker.close()
