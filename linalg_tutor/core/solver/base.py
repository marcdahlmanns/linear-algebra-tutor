"""Base classes for the step-by-step solver system."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SolutionStep(BaseModel):
    """A single step in a solution."""

    step_number: int = Field(..., description="Step number in sequence")
    description: str = Field(..., description="Brief description of this step")
    mathematical_expression: str = Field(
        ..., description="Mathematical expression or computation for this step"
    )
    explanation: str = Field(
        default="", description="Detailed explanation of why we do this step"
    )
    intermediate_result: Optional[Any] = Field(
        default=None, description="Intermediate result after this step"
    )

    class Config:
        arbitrary_types_allowed = True


class Solution(BaseModel):
    """Complete solution with step-by-step explanation."""

    problem_statement: str = Field(..., description="Statement of the problem")
    steps: List[SolutionStep] = Field(
        default_factory=list, description="List of solution steps"
    )
    final_answer: Any = Field(..., description="The final answer")
    verification: Optional[str] = Field(
        default=None, description="Optional verification or check of the answer"
    )

    class Config:
        arbitrary_types_allowed = True


class Solver(ABC):
    """Abstract base class for solvers.

    Each solver knows how to solve a specific type of problem
    (e.g., matrix multiplication, eigenvalues, Gaussian elimination)
    and generate step-by-step solutions.
    """

    @abstractmethod
    def solve_with_steps(self, inputs: Dict[str, Any], expected: Any) -> Solution:
        """Generate step-by-step solution for a problem.

        Args:
            inputs: Dictionary of input data (matrices, vectors, etc.)
            expected: The expected answer (used for verification)

        Returns:
            Solution object with steps and final answer
        """
        pass

    def solve(self, inputs: Dict[str, Any]) -> Any:
        """Solve the problem and return just the answer.

        This is a convenience method that just returns the final answer
        without generating the full step-by-step solution.

        Args:
            inputs: Dictionary of input data

        Returns:
            The final answer
        """
        solution = self.solve_with_steps(inputs, None)
        return solution.final_answer
