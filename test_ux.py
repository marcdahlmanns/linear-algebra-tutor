#!/usr/bin/env python3
"""Quick test of the improved UX."""

from linalg_tutor.core.exercises import ComputationalExercise, ExerciseDifficulty
from linalg_tutor.cli.ui.prompts import ExercisePrompt
import numpy as np

# Create a test exercise
exercise = ComputationalExercise(
    exercise_id="test_vec_add",
    topic="vectors",
    difficulty=ExerciseDifficulty.PRACTICE,
    question="Add the vectors v = [3, 4] and w = [1, 2]",
    operation="vector_add",
    inputs={"v": np.array([3, 4]), "w": np.array([1, 2])},
    expected_output=np.array([4, 6]),
    hints=[
        "Add component-wise: v + w = [v₁+w₁, v₂+w₂]",
        "First component: 3 + 1 = 4",
        "Second component: 4 + 2 = 6",
    ],
    tags=["vector_addition", "basic"],
)

# Run the improved prompt
prompt = ExercisePrompt(exercise, current_exercise=1, total_exercises=3)
result = prompt.run()

print(f"\nResult: {result.correct}")
