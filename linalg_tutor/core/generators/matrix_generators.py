"""Exercise generators for matrix operations."""

import numpy as np
from typing import Optional

from .base import ExerciseGenerator, GeneratorConfig
from ..exercises import ComputationalExercise, ExerciseDifficulty


class MatrixAdditionGenerator(ExerciseGenerator):
    """Generator for matrix addition exercises."""

    def generate(self) -> ComputationalExercise:
        """Generate a random matrix addition exercise."""
        rows = self.random_dimension()
        cols = self.random_dimension()

        A = self.random_matrix(rows, cols)
        B = self.random_matrix(rows, cols)

        result = A + B

        question = f"Add the matrices A = {self.format_matrix(A)} and B = {self.format_matrix(B)}"

        hints = [
            f"Matrix addition is element-wise: (A + B)[i,j] = A[i,j] + B[i,j]",
            f"First element: A[0,0] + B[0,0] = {A[0,0]:.4g} + {B[0,0]:.4g} = {result[0,0]:.4g}",
        ]

        if cols >= 2:
            hints.append(f"Element (0,1): {A[0,1]:.4g} + {B[0,1]:.4g} = {result[0,1]:.4g}")

        return ComputationalExercise(
            exercise_id=self.generate_exercise_id("mat_add"),
            topic="matrices",
            difficulty=self.config.difficulty,
            question=question,
            operation="matrix_add",
            inputs={"A": A, "B": B},
            expected_output=result,
            hints=hints,
            tags=["matrix_addition", "generated"],
        )


class MatrixScalarMultGenerator(ExerciseGenerator):
    """Generator for matrix scalar multiplication exercises."""

    def generate(self) -> ComputationalExercise:
        """Generate a random matrix scalar multiplication exercise."""
        rows = self.random_dimension()
        cols = self.random_dimension()

        scalar = self.random_scalar(nonzero=True)
        matrix = self.random_matrix(rows, cols)

        result = scalar * matrix

        question = f"Multiply the matrix A = {self.format_matrix(matrix)} by the scalar {scalar:.4g}"

        hints = [
            f"Scalar multiplication: multiply each element by {scalar:.4g}",
            f"Element (0,0): {scalar:.4g} × {matrix[0,0]:.4g} = {result[0,0]:.4g}",
        ]

        if cols >= 2:
            hints.append(f"Element (0,1): {scalar:.4g} × {matrix[0,1]:.4g} = {result[0,1]:.4g}")

        return ComputationalExercise(
            exercise_id=self.generate_exercise_id("mat_scalar"),
            topic="matrices",
            difficulty=self.config.difficulty,
            question=question,
            operation="matrix_scalar_mult",
            inputs={"scalar": scalar, "A": matrix},
            expected_output=result,
            hints=hints,
            tags=["matrix_scalar_multiplication", "generated"],
        )


class MatrixMultiplicationGenerator(ExerciseGenerator):
    """Generator for matrix multiplication exercises."""

    def generate(self) -> ComputationalExercise:
        """Generate a random matrix multiplication exercise."""
        # A is m×n, B is n×p
        m = self.random_dimension()
        n = self.random_dimension()
        p = self.random_dimension()

        A = self.random_matrix(m, n)
        B = self.random_matrix(n, p)

        result = A @ B

        question = f"Multiply the matrices A = {self.format_matrix(A)} and B = {self.format_matrix(B)}"

        # Generate hints for first element
        row_0_A = A[0, :]
        col_0_B = B[:, 0]

        products = [f"({row_0_A[i]:.4g})({col_0_B[i]:.4g})" for i in range(n)]

        hints = [
            f"Matrix multiplication: (AB)[i,j] = sum of (row i of A) × (column j of B)",
            f"For element (0,0): row 0 of A · column 0 of B",
            f"= {' + '.join(products)} = {result[0,0]:.4g}",
        ]

        return ComputationalExercise(
            exercise_id=self.generate_exercise_id("mat_mult"),
            topic="matrices",
            difficulty=self.config.difficulty,
            question=question,
            operation="matrix_multiply",
            inputs={"A": A, "B": B},
            expected_output=result,
            hints=hints,
            tags=["matrix_multiplication", "generated"],
        )


class MatrixTransposeGenerator(ExerciseGenerator):
    """Generator for matrix transpose exercises."""

    def generate(self) -> ComputationalExercise:
        """Generate a random matrix transpose exercise."""
        rows = self.random_dimension()
        cols = self.random_dimension()

        matrix = self.random_matrix(rows, cols)
        result = matrix.T

        question = f"Find the transpose of A = {self.format_matrix(matrix)}"

        hints = [
            "Transpose: flip rows and columns (Aᵀ[i,j] = A[j,i])",
            f"Original A[0,0] = {matrix[0,0]:.4g} → Aᵀ[0,0] = {result[0,0]:.4g}",
        ]

        if rows >= 2 and cols >= 2:
            hints.append(f"Original A[0,1] = {matrix[0,1]:.4g} → Aᵀ[1,0] = {result[1,0]:.4g}")
            hints.append(f"Original A[1,0] = {matrix[1,0]:.4g} → Aᵀ[0,1] = {result[0,1]:.4g}")

        return ComputationalExercise(
            exercise_id=self.generate_exercise_id("mat_transpose"),
            topic="matrices",
            difficulty=self.config.difficulty,
            question=question,
            operation="matrix_transpose",
            inputs={"A": matrix},
            expected_output=result,
            hints=hints,
            tags=["matrix_transpose", "generated"],
        )


class DeterminantGenerator(ExerciseGenerator):
    """Generator for determinant exercises (2×2 only for now)."""

    def __init__(self, config: Optional[GeneratorConfig] = None):
        """Initialize with 2×2 only configuration."""
        if config is None:
            config = GeneratorConfig(min_dimension=2, max_dimension=2)
        else:
            config.min_dimension = 2
            config.max_dimension = 2
        super().__init__(config)

    def generate(self) -> ComputationalExercise:
        """Generate a random 2×2 determinant exercise."""
        matrix = self.random_matrix(2, 2)

        a, b = matrix[0, 0], matrix[0, 1]
        c, d = matrix[1, 0], matrix[1, 1]

        result = np.linalg.det(matrix)

        question = f"Calculate the determinant of A = {self.format_matrix(matrix)}"

        hints = [
            "For 2×2 matrix: det(A) = ad - bc",
            f"det(A) = ({a:.4g})({d:.4g}) - ({b:.4g})({c:.4g})",
            f"= {a*d:.4g} - {b*c:.4g} = {result:.4g}",
        ]

        return ComputationalExercise(
            exercise_id=self.generate_exercise_id("det_2x2"),
            topic="matrices",
            difficulty=self.config.difficulty,
            question=question,
            operation="determinant",
            inputs={"A": matrix},
            expected_output=result,
            hints=hints,
            tags=["determinant", "2x2", "generated"],
        )


class MatrixInverseGenerator(ExerciseGenerator):
    """Generator for matrix inverse exercises (2×2 only)."""

    def __init__(self, config: Optional[GeneratorConfig] = None):
        """Initialize with 2×2 and non-singular configuration."""
        if config is None:
            config = GeneratorConfig(min_dimension=2, max_dimension=2, avoid_singular=True)
        else:
            config.min_dimension = 2
            config.max_dimension = 2
            config.avoid_singular = True
        super().__init__(config)

    def generate(self) -> ComputationalExercise:
        """Generate a random 2×2 matrix inverse exercise."""
        matrix = self.random_matrix(2, 2)

        # Ensure invertible
        attempts = 0
        while abs(np.linalg.det(matrix)) < 0.1 and attempts < 20:
            matrix = self.random_matrix(2, 2)
            attempts += 1

        result = np.linalg.inv(matrix)

        a, b = matrix[0, 0], matrix[0, 1]
        c, d = matrix[1, 0], matrix[1, 1]
        det = a * d - b * c

        question = f"Find the inverse of A = {self.format_matrix(matrix)}"

        hints = [
            f"For 2×2 matrix: A⁻¹ = (1/det(A)) × [[d, -b], [-c, a]]",
            f"First calculate det(A) = ad - bc = {det:.4g}",
            f"Then A⁻¹ = (1/{det:.4g}) × [[{d:.4g}, {-b:.4g}], [{-c:.4g}, {a:.4g}]]",
        ]

        return ComputationalExercise(
            exercise_id=self.generate_exercise_id("mat_inv"),
            topic="matrices",
            difficulty=ExerciseDifficulty.APPLICATION,  # Inverse is harder
            question=question,
            operation="matrix_inverse",
            inputs={"A": matrix},
            expected_output=result,
            hints=hints,
            tags=["matrix_inverse", "2x2", "generated"],
        )
