"""Linear Algebra Tutor - Interactive CLI teaching tool."""

__version__ = "0.1.0"
__author__ = "Linear Algebra Tutor Team"

from .core.exercises import (
    Exercise,
    ExerciseResult,
    ExerciseType,
    ExerciseDifficulty,
    ComputationalExercise,
    MultipleChoiceExercise,
    TrueFalseExercise,
    FillInExercise,
)

from .core.solver import (
    Solver,
    Solution,
    SolutionStep,
    get_solver,
)

__all__ = [
    "__version__",
    "__author__",
    "Exercise",
    "ExerciseResult",
    "ExerciseType",
    "ExerciseDifficulty",
    "ComputationalExercise",
    "MultipleChoiceExercise",
    "TrueFalseExercise",
    "FillInExercise",
    "Solver",
    "Solution",
    "SolutionStep",
    "get_solver",
]
