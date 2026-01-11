"""Base classes for the exercise system."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class ExerciseType(str, Enum):
    """Types of exercises available."""

    COMPUTATIONAL = "computational"
    FILL_IN = "fill_in"
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"


class ExerciseDifficulty(str, Enum):
    """Difficulty levels for exercises."""

    PRACTICE = "practice"  # Direct application of learned concept
    APPLICATION = "application"  # Requires combining concepts
    CHALLENGE = "challenge"  # Multi-step problems, edge cases


class ExerciseResult(BaseModel):
    """Result of an exercise attempt."""

    correct: bool
    user_answer: Any
    correct_answer: Any
    feedback: str
    hints_used: int = 0
    time_spent: Optional[float] = None  # seconds

    class Config:
        arbitrary_types_allowed = True


class Exercise(BaseModel, ABC):
    """Abstract base class for all exercises.

    All exercise types inherit from this class and implement:
    - check_answer(): Validate user's answer
    - get_solution(): Get solution with optional step-by-step explanation
    - generate_similar(): Generate a similar exercise with different values
    """

    exercise_id: str = Field(..., description="Unique identifier for this exercise")
    topic: str = Field(..., description="Topic this exercise belongs to")
    difficulty: ExerciseDifficulty = Field(..., description="Difficulty level")
    question: str = Field(..., description="The question text")
    hints: List[str] = Field(default_factory=list, description="List of hints")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")

    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def check_answer(self, user_answer: Any) -> ExerciseResult:
        """Validate user's answer and return result.

        Args:
            user_answer: The answer provided by the user

        Returns:
            ExerciseResult containing validation results and feedback
        """
        pass

    @abstractmethod
    def get_solution(self, show_steps: bool = True) -> str:
        """Get the solution to this exercise.

        Args:
            show_steps: If True, include step-by-step explanation

        Returns:
            Formatted solution string
        """
        pass

    @abstractmethod
    def get_correct_answer(self) -> Any:
        """Get the correct answer without explanation.

        Returns:
            The correct answer in its raw form
        """
        pass

    def generate_similar(self) -> "Exercise":
        """Generate a similar exercise with different values.

        This is optional - default implementation raises NotImplementedError.
        Individual exercise types can override this if they support generation.

        Returns:
            A new exercise similar to this one

        Raises:
            NotImplementedError: If generation is not supported for this exercise
        """
        raise NotImplementedError(
            f"Exercise type {self.__class__.__name__} does not support generation"
        )
