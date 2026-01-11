"""Exercise generators for linear systems."""

import numpy as np
from typing import Optional

from .base import ExerciseGenerator, GeneratorConfig
from ..exercises import ComputationalExercise, ExerciseDifficulty


class LinearSystemGenerator(ExerciseGenerator):
    """Generator for linear systems Ax = b."""

    def __init__(self, config: Optional[GeneratorConfig] = None):
        """Initialize with non-singular matrix configuration."""
        if config is None:
            config = GeneratorConfig(avoid_singular=True)
        else:
            config.avoid_singular = True
        super().__init__(config)

    def generate(self) -> ComputationalExercise:
        """Generate a solvable linear system with known solution.

        Strategy: Generate solution x first, then compute b = Ax
        """
        n = self.random_dimension()

        # Generate random solution vector
        x_solution = self.random_vector(n)

        # Generate non-singular coefficient matrix
        A = self.random_matrix(n, n)

        # Ensure A is invertible
        attempts = 0
        while abs(np.linalg.det(A)) < 0.1 and attempts < 20:
            A = self.random_matrix(n, n)
            attempts += 1

        # Compute b from known solution
        b = A @ x_solution

        question = (
            f"Solve the linear system Ax = b where:\n"
            f"A = {self.format_matrix(A)}\n"
            f"b = {self.format_vector(b)}"
        )

        hints = [
            "Use row reduction (RREF) on the augmented matrix [A|b]",
            f"The system has {n} equations and {n} unknowns",
            "Apply Gaussian elimination to solve",
        ]

        return ComputationalExercise(
            exercise_id=self.generate_exercise_id("lin_sys"),
            topic="linear_systems",
            difficulty=self.config.difficulty,
            question=question,
            operation="linear_system",
            inputs={"A": A, "b": b},
            expected_output=x_solution,
            hints=hints,
            tags=["linear_system", "generated"],
        )


class TriangularSystemGenerator(ExerciseGenerator):
    """Generator for triangular linear systems (easier to solve by hand)."""

    def generate(self) -> ComputationalExercise:
        """Generate an upper triangular system."""
        n = self.random_dimension()

        # Generate random solution
        x_solution = self.random_vector(n)

        # Create upper triangular matrix
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    # Diagonal: ensure non-zero
                    A[i, j] = self.random_scalar(nonzero=True)
                else:
                    A[i, j] = self.random_value()

        # Compute b
        b = A @ x_solution

        question = (
            f"Solve the triangular system Ax = b where:\n"
            f"A = {self.format_matrix(A)}\n"
            f"b = {self.format_vector(b)}"
        )

        hints = [
            "This is an upper triangular system - use back-substitution",
            f"Start from the last equation: x_{n} = ...",
            "Then substitute back to find previous variables",
        ]

        return ComputationalExercise(
            exercise_id=self.generate_exercise_id("tri_sys"),
            topic="linear_systems",
            difficulty=ExerciseDifficulty.PRACTICE,
            question=question,
            operation="linear_system",
            inputs={"A": A, "b": b},
            expected_output=x_solution,
            hints=hints,
            tags=["triangular_system", "generated"],
        )


class Simple2x2SystemGenerator(ExerciseGenerator):
    """Generator for simple 2×2 linear systems."""

    def __init__(self, config: Optional[GeneratorConfig] = None):
        """Initialize with 2×2 configuration."""
        if config is None:
            config = GeneratorConfig(min_dimension=2, max_dimension=2, avoid_singular=True)
        else:
            config.min_dimension = 2
            config.max_dimension = 2
            config.avoid_singular = True
        super().__init__(config)

    def generate(self) -> ComputationalExercise:
        """Generate a simple 2×2 linear system."""
        # Generate solution with small integers
        x_solution = np.array([self.random_scalar(nonzero=False) for _ in range(2)])

        # Generate coefficient matrix
        A = self.random_matrix(2, 2)

        # Ensure good determinant
        attempts = 0
        while abs(np.linalg.det(A)) < 0.5 and attempts < 20:
            A = self.random_matrix(2, 2)
            attempts += 1

        b = A @ x_solution

        # Format as equations
        a11, a12 = A[0, 0], A[0, 1]
        a21, a22 = A[1, 0], A[1, 1]
        b1, b2 = b[0], b[1]

        eq1 = f"{a11:.4g}x + {a12:.4g}y = {b1:.4g}"
        eq2 = f"{a21:.4g}x + {a22:.4g}y = {b2:.4g}"

        question = f"Solve the system of equations:\n{eq1}\n{eq2}"

        hints = [
            "Use elimination or substitution method",
            f"Multiply first equation by {a21/a11:.4g} to eliminate x",
            "Or use matrix methods: x = A⁻¹b",
        ]

        return ComputationalExercise(
            exercise_id=self.generate_exercise_id("sys_2x2"),
            topic="linear_systems",
            difficulty=ExerciseDifficulty.PRACTICE,
            question=question,
            operation="linear_system",
            inputs={"A": A, "b": b},
            expected_output=x_solution,
            hints=hints,
            tags=["linear_system", "2x2", "generated"],
        )
