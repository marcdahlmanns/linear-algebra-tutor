"""Fill-in-the-blank exercises."""

from typing import Any, List, Union

import numpy as np
from pydantic import Field

from .base import Exercise, ExerciseResult, ExerciseDifficulty


class FillInExercise(Exercise):
    """Fill-in-the-blank exercise.

    Used for completing partial solutions, filling in missing matrix entries,
    or completing definitions/theorems.

    Supports both:
    - Text answers (exact string matching)
    - Numerical answers (with tolerance)
    """

    answer_type: str = Field(
        default="text", description="Type of answer: 'text' or 'numerical'"
    )
    correct_answer: Union[str, float, List[float]] = Field(
        ..., description="The correct answer(s) to fill in"
    )
    case_sensitive: bool = Field(
        default=False, description="For text answers, whether case matters"
    )
    tolerance: float = Field(
        default=1e-5, description="For numerical answers, relative tolerance"
    )
    atol: float = Field(
        default=1e-8, description="For numerical answers, absolute tolerance"
    )
    accept_equivalent: bool = Field(
        default=True,
        description="For text answers, accept mathematically equivalent forms (e.g., whitespace differences)",
    )

    def check_answer(self, user_answer: Any) -> ExerciseResult:
        """Check if user's fill-in answer is correct.

        Args:
            user_answer: User's answer (string or number)

        Returns:
            ExerciseResult with validation results
        """
        try:
            if self.answer_type == "numerical":
                return self._check_numerical_answer(user_answer)
            else:  # text
                return self._check_text_answer(user_answer)

        except Exception as e:
            return ExerciseResult(
                correct=False,
                user_answer=user_answer,
                correct_answer=self.correct_answer,
                feedback=f"Error processing answer: {str(e)}",
                hints_used=0,
            )

    def _check_numerical_answer(self, user_answer: Any) -> ExerciseResult:
        """Check numerical fill-in answer."""
        try:
            # Handle list of numbers
            if isinstance(self.correct_answer, list):
                if not isinstance(user_answer, (list, tuple)):
                    return ExerciseResult(
                        correct=False,
                        user_answer=user_answer,
                        correct_answer=self.correct_answer,
                        feedback=f"Expected a list of {len(self.correct_answer)} numbers.",
                        hints_used=0,
                    )

                if len(user_answer) != len(self.correct_answer):
                    return ExerciseResult(
                        correct=False,
                        user_answer=user_answer,
                        correct_answer=self.correct_answer,
                        feedback=f"Expected {len(self.correct_answer)} values, got {len(user_answer)}.",
                        hints_used=0,
                    )

                user_array = np.array(user_answer, dtype=float)
                correct_array = np.array(self.correct_answer, dtype=float)
            else:
                # Single number
                user_array = np.array(float(user_answer))
                correct_array = np.array(float(self.correct_answer))

            is_close = np.allclose(
                user_array, correct_array, rtol=self.tolerance, atol=self.atol
            )

            feedback = "Correct!" if is_close else "Incorrect. Check your calculation."

            return ExerciseResult(
                correct=is_close,
                user_answer=user_answer,
                correct_answer=self.correct_answer,
                feedback=feedback,
                hints_used=0,
            )

        except (ValueError, TypeError) as e:
            return ExerciseResult(
                correct=False,
                user_answer=user_answer,
                correct_answer=self.correct_answer,
                feedback=f"Invalid numerical input: {str(e)}",
                hints_used=0,
            )

    def _check_text_answer(self, user_answer: Any) -> ExerciseResult:
        """Check text fill-in answer."""
        if not isinstance(user_answer, str):
            return ExerciseResult(
                correct=False,
                user_answer=user_answer,
                correct_answer=self.correct_answer,
                feedback="Expected a text answer.",
                hints_used=0,
            )

        user_text = user_answer
        correct_text = str(self.correct_answer)

        # Normalize whitespace if accept_equivalent is True
        if self.accept_equivalent:
            user_text = ' '.join(user_text.split())
            correct_text = ' '.join(correct_text.split())

        # Case sensitivity
        if not self.case_sensitive:
            user_text = user_text.lower()
            correct_text = correct_text.lower()

        is_correct = user_text == correct_text

        feedback = (
            "Correct!"
            if is_correct
            else f"Incorrect. The correct answer is: {self.correct_answer}"
        )

        return ExerciseResult(
            correct=is_correct,
            user_answer=user_answer,
            correct_answer=self.correct_answer,
            feedback=feedback,
            hints_used=0,
        )

    def get_solution(self, show_steps: bool = True) -> str:
        """Get the solution.

        Args:
            show_steps: Ignored for fill-in (no steps to show)

        Returns:
            Formatted solution string
        """
        return f"Answer: {self.correct_answer}\n"

    def get_correct_answer(self) -> Any:
        """Get the correct answer.

        Returns:
            The correct answer
        """
        return self.correct_answer
