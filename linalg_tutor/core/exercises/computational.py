"""Computational exercises requiring numerical answers."""

from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import Field

from .base import Exercise, ExerciseResult, ExerciseDifficulty


class ComputationalExercise(Exercise):
    """Exercise requiring numerical computation.

    This exercise type is used for problems like:
    - Matrix multiplication
    - Vector dot products
    - Solving linear systems
    - Computing eigenvalues
    - etc.

    The answer is validated using numerical tolerance to handle floating-point errors.
    """

    operation: str = Field(..., description="Type of operation (e.g., 'matrix_multiply')")
    inputs: Dict[str, Any] = Field(
        ..., description="Input data for the problem (matrices, vectors, scalars, etc.)"
    )
    expected_output: Union[np.ndarray, float, int] = Field(
        ..., description="Expected answer"
    )
    tolerance: float = Field(
        default=1e-5, description="Relative tolerance for numerical comparison"
    )
    atol: float = Field(
        default=1e-8, description="Absolute tolerance for numerical comparison"
    )
    solution_steps: Optional[List[Dict[str, str]]] = Field(
        default=None, description="Pre-computed solution steps"
    )

    class Config:
        arbitrary_types_allowed = True

    def check_answer(self, user_answer: Any) -> ExerciseResult:
        """Check numerical answer with tolerance.

        Args:
            user_answer: User's answer (can be list, numpy array, or scalar)

        Returns:
            ExerciseResult with validation results
        """
        try:
            # Convert user answer to numpy array
            if isinstance(user_answer, (list, tuple)):
                user_array = np.array(user_answer, dtype=float)
            elif isinstance(user_answer, np.ndarray):
                user_array = user_answer
            elif isinstance(user_answer, (int, float)):
                user_array = np.array(user_answer, dtype=float)
            else:
                return ExerciseResult(
                    correct=False,
                    user_answer=user_answer,
                    correct_answer=self._format_answer(self.expected_output),
                    feedback=f"Invalid answer format. Expected a number or list, got {type(user_answer).__name__}",
                    hints_used=0,
                )

            expected_array = np.atleast_1d(self.expected_output)
            user_array = np.atleast_1d(user_array)

            # Check shape
            if user_array.shape != expected_array.shape:
                return ExerciseResult(
                    correct=False,
                    user_answer=user_answer,
                    correct_answer=self._format_answer(self.expected_output),
                    feedback=f"Shape mismatch: expected {expected_array.shape}, got {user_array.shape}",
                    hints_used=0,
                )

            # Check values with tolerance
            is_close = np.allclose(
                user_array, expected_array, rtol=self.tolerance, atol=self.atol
            )

            if is_close:
                feedback = "Correct!"
            else:
                # Provide more specific feedback
                max_diff = np.max(np.abs(user_array - expected_array))
                feedback = f"Incorrect. Maximum difference from expected: {max_diff:.6f}"

            return ExerciseResult(
                correct=is_close,
                user_answer=user_answer,
                correct_answer=self._format_answer(self.expected_output),
                feedback=feedback,
                hints_used=0,
            )

        except (ValueError, TypeError) as e:
            return ExerciseResult(
                correct=False,
                user_answer=user_answer,
                correct_answer=self._format_answer(self.expected_output),
                feedback=f"Error processing answer: {str(e)}",
                hints_used=0,
            )

    def get_solution(self, show_steps: bool = True) -> str:
        """Get solution with optional step-by-step explanation.

        Args:
            show_steps: If True, show step-by-step solution

        Returns:
            Formatted solution string
        """
        if show_steps and self.solution_steps:
            # Use pre-computed steps
            solution = "Solution:\n\n"
            for i, step in enumerate(self.solution_steps, 1):
                solution += f"Step {i}: {step.get('description', '')}\n"
                if 'expression' in step:
                    solution += f"  {step['expression']}\n"
                if 'explanation' in step:
                    solution += f"  {step['explanation']}\n"
                solution += "\n"
            solution += f"Final Answer: {self._format_answer(self.expected_output)}\n"
            return solution
        elif show_steps:
            # Try to get solver to generate steps
            from ..solver import get_solver

            try:
                solver = get_solver(self.operation)
                if solver:
                    solution_obj = solver.solve_with_steps(self.inputs, self.expected_output)
                    return self._format_solution(solution_obj)
            except (ImportError, KeyError, NotImplementedError):
                pass

        # Fallback: just show the answer
        return f"Answer: {self._format_answer(self.expected_output)}"

    def get_correct_answer(self) -> Any:
        """Get the correct answer.

        Returns:
            The expected output
        """
        return self.expected_output

    def _format_answer(self, answer: Union[np.ndarray, float, int]) -> Any:
        """Format answer for display.

        Args:
            answer: The answer to format

        Returns:
            Formatted answer (list for arrays, number for scalars)
        """
        if isinstance(answer, np.ndarray):
            if answer.ndim == 0:
                return float(answer)
            return answer.tolist()
        return answer

    def _format_solution(self, solution_obj: Any) -> str:
        """Format a Solution object into a string.

        Args:
            solution_obj: Solution object from solver

        Returns:
            Formatted solution string
        """
        if hasattr(solution_obj, 'problem_statement'):
            output = f"{solution_obj.problem_statement}\n\n"
        else:
            output = "Solution:\n\n"

        if hasattr(solution_obj, 'steps'):
            for step in solution_obj.steps:
                output += f"Step {step.step_number}: {step.description}\n"
                output += f"  {step.mathematical_expression}\n"
                if step.explanation:
                    output += f"  {step.explanation}\n"
                output += "\n"

        if hasattr(solution_obj, 'final_answer'):
            output += f"Final Answer: {self._format_answer(solution_obj.final_answer)}\n"

        return output
