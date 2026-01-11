"""True/False exercises."""

from typing import Any

from pydantic import Field

from .base import Exercise, ExerciseResult, ExerciseDifficulty


class TrueFalseExercise(Exercise):
    """True or False question for quick conceptual checks.

    Used for testing understanding of properties, definitions, and theorems.
    """

    correct_answer: bool = Field(..., description="The correct answer (True or False)")
    explanation: str = Field(
        default="", description="Explanation of why the statement is true or false"
    )

    def check_answer(self, user_answer: Any) -> ExerciseResult:
        """Check if user's answer matches the correct boolean value.

        Args:
            user_answer: User's answer (bool, or string like "true"/"false")

        Returns:
            ExerciseResult with validation results
        """
        try:
            # Handle various input formats
            if isinstance(user_answer, bool):
                user_bool = user_answer
            elif isinstance(user_answer, str):
                user_lower = user_answer.lower().strip()
                if user_lower in ('true', 't', 'yes', 'y', '1'):
                    user_bool = True
                elif user_lower in ('false', 'f', 'no', 'n', '0'):
                    user_bool = False
                else:
                    return ExerciseResult(
                        correct=False,
                        user_answer=user_answer,
                        correct_answer=self.correct_answer,
                        feedback=f"Invalid answer '{user_answer}'. Please answer True or False.",
                        hints_used=0,
                    )
            elif isinstance(user_answer, int):
                user_bool = bool(user_answer)
            else:
                return ExerciseResult(
                    correct=False,
                    user_answer=user_answer,
                    correct_answer=self.correct_answer,
                    feedback=f"Invalid answer type. Expected True or False.",
                    hints_used=0,
                )

            is_correct = user_bool == self.correct_answer

            if is_correct:
                feedback = "Correct!"
                if self.explanation:
                    feedback += f" {self.explanation}"
            else:
                feedback = f"Incorrect. The statement is {self.correct_answer}."
                if self.explanation:
                    feedback += f"\n\nExplanation: {self.explanation}"

            return ExerciseResult(
                correct=is_correct,
                user_answer=user_bool,
                correct_answer=self.correct_answer,
                feedback=feedback,
                hints_used=0,
            )

        except Exception as e:
            return ExerciseResult(
                correct=False,
                user_answer=user_answer,
                correct_answer=self.correct_answer,
                feedback=f"Error processing answer: {str(e)}",
                hints_used=0,
            )

    def get_solution(self, show_steps: bool = True) -> str:
        """Get the solution.

        Args:
            show_steps: Ignored for true/false (no steps to show)

        Returns:
            Formatted solution string
        """
        solution = f"Answer: {self.correct_answer}\n"
        if self.explanation:
            solution += f"\nExplanation:\n{self.explanation}\n"
        return solution

    def get_correct_answer(self) -> bool:
        """Get the correct answer.

        Returns:
            The correct boolean value
        """
        return self.correct_answer
