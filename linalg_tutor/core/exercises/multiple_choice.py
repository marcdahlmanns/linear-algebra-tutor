"""Multiple choice exercises."""

from typing import Any, List

from pydantic import Field, field_validator

from .base import Exercise, ExerciseResult, ExerciseDifficulty


class MultipleChoiceExercise(Exercise):
    """Multiple choice question with predefined options.

    Used for conceptual questions, definitions, and property verification.
    """

    choices: List[str] = Field(..., description="List of answer choices", min_length=2)
    correct_index: int = Field(..., description="Index of the correct answer (0-based)")
    explanation: str = Field(
        default="", description="Explanation of why the answer is correct"
    )

    @field_validator('correct_index')
    @classmethod
    def validate_correct_index(cls, v: int, info) -> int:
        """Ensure correct_index is valid."""
        # Note: We can't access choices here in Pydantic v2, will validate in model_validator
        if v < 0:
            raise ValueError("correct_index must be non-negative")
        return v

    def check_answer(self, user_answer: Any) -> ExerciseResult:
        """Check if user selected the correct choice.

        Args:
            user_answer: Index of user's selection (0-based) or the choice text

        Returns:
            ExerciseResult with validation results
        """
        try:
            # Handle both index and text answers
            if isinstance(user_answer, int):
                user_index = user_answer
            elif isinstance(user_answer, str):
                # Try to find the choice text in our list
                try:
                    user_index = self.choices.index(user_answer)
                except ValueError:
                    return ExerciseResult(
                        correct=False,
                        user_answer=user_answer,
                        correct_answer=self.choices[self.correct_index],
                        feedback=f"Invalid choice. Please select from the given options.",
                        hints_used=0,
                    )
            else:
                return ExerciseResult(
                    correct=False,
                    user_answer=user_answer,
                    correct_answer=self.choices[self.correct_index],
                    feedback=f"Invalid answer type. Expected an index (0-{len(self.choices)-1}) or choice text.",
                    hints_used=0,
                )

            # Validate index is in range
            if user_index < 0 or user_index >= len(self.choices):
                return ExerciseResult(
                    correct=False,
                    user_answer=user_answer,
                    correct_answer=self.choices[self.correct_index],
                    feedback=f"Index out of range. Must be between 0 and {len(self.choices)-1}.",
                    hints_used=0,
                )

            is_correct = user_index == self.correct_index

            if is_correct:
                feedback = "Correct!"
                if self.explanation:
                    feedback += f" {self.explanation}"
            else:
                feedback = f"Incorrect. The correct answer is: {self.choices[self.correct_index]}"
                if self.explanation:
                    feedback += f"\n\nExplanation: {self.explanation}"

            return ExerciseResult(
                correct=is_correct,
                user_answer=user_answer,
                correct_answer=self.choices[self.correct_index],
                feedback=feedback,
                hints_used=0,
            )

        except Exception as e:
            return ExerciseResult(
                correct=False,
                user_answer=user_answer,
                correct_answer=self.choices[self.correct_index],
                feedback=f"Error processing answer: {str(e)}",
                hints_used=0,
            )

    def get_solution(self, show_steps: bool = True) -> str:
        """Get the solution.

        Args:
            show_steps: Ignored for multiple choice (no steps to show)

        Returns:
            Formatted solution string
        """
        solution = f"Correct Answer: {self.choices[self.correct_index]}\n"
        if self.explanation:
            solution += f"\nExplanation:\n{self.explanation}\n"
        return solution

    def get_correct_answer(self) -> Any:
        """Get the correct answer.

        Returns:
            The correct choice text
        """
        return self.choices[self.correct_index]
