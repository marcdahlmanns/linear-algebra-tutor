"""Exercise generators for vector operations."""

import numpy as np
from typing import Optional

from .base import ExerciseGenerator, GeneratorConfig
from ..exercises import ComputationalExercise, ExerciseDifficulty


class VectorAdditionGenerator(ExerciseGenerator):
    """Generator for vector addition exercises."""

    def generate(self) -> ComputationalExercise:
        """Generate a random vector addition exercise."""
        # Generate two vectors of same dimension
        dimension = self.random_dimension()
        v = self.random_vector(dimension)
        w = self.random_vector(dimension)

        # Calculate answer
        result = v + w

        # Generate question
        question = f"Add the vectors v = {self.format_vector(v)} and w = {self.format_vector(w)}"

        # Generate hints based on dimension
        hints = [
            f"Add component-wise: v + w = [v₁+w₁, v₂+w₂, ...]",
            f"First component: {v[0]:.4g} + {w[0]:.4g} = {result[0]:.4g}",
        ]

        if dimension >= 2:
            hints.append(f"Second component: {v[1]:.4g} + {w[1]:.4g} = {result[1]:.4g}")
        if dimension == 3:
            hints.append(f"Third component: {v[2]:.4g} + {w[2]:.4g} = {result[2]:.4g}")

        return ComputationalExercise(
            exercise_id=self.generate_exercise_id("vec_add"),
            topic="vectors",
            difficulty=self.config.difficulty,
            question=question,
            operation="vector_add",
            inputs={"v": v, "w": w},
            expected_output=result,
            hints=hints,
            tags=["vector_addition", "generated"],
        )


class VectorScalarMultGenerator(ExerciseGenerator):
    """Generator for scalar multiplication exercises."""

    def generate(self) -> ComputationalExercise:
        """Generate a random scalar multiplication exercise."""
        dimension = self.random_dimension()
        scalar = self.random_scalar(nonzero=True)
        vector = self.random_vector(dimension)

        result = scalar * vector

        question = f"Multiply the vector v = {self.format_vector(vector)} by the scalar {scalar:.4g}"

        hints = [
            f"Scalar multiplication: {scalar:.4g}v = [{scalar:.4g}v₁, {scalar:.4g}v₂, ...]",
            f"Multiply each component by {scalar:.4g}",
        ]

        for i in range(min(2, dimension)):
            hints.append(f"Component {i}: {scalar:.4g} × {vector[i]:.4g} = {result[i]:.4g}")

        return ComputationalExercise(
            exercise_id=self.generate_exercise_id("vec_scalar"),
            topic="vectors",
            difficulty=self.config.difficulty,
            question=question,
            operation="scalar_mult",
            inputs={"scalar": scalar, "v": vector},
            expected_output=result,
            hints=hints,
            tags=["scalar_multiplication", "generated"],
        )


class DotProductGenerator(ExerciseGenerator):
    """Generator for dot product exercises."""

    def generate(self) -> ComputationalExercise:
        """Generate a random dot product exercise."""
        dimension = self.random_dimension()
        v = self.random_vector(dimension)
        w = self.random_vector(dimension)

        result = np.dot(v, w)

        question = f"Compute the dot product of v = {self.format_vector(v)} and w = {self.format_vector(w)}"

        # Generate step-by-step hints
        hints = [
            f"Dot product: v · w = v₁w₁ + v₂w₂ + ...",
        ]

        # Show first few products
        products = []
        for i in range(dimension):
            products.append(f"({v[i]:.4g})({w[i]:.4g})")

        hints.append(f"Calculate: {' + '.join(products)}")

        # Show intermediate calculation
        partial_sums = []
        for i in range(dimension):
            partial_sums.append(f"{v[i] * w[i]:.4g}")

        hints.append(f"= {' + '.join(partial_sums)} = {result:.4g}")

        return ComputationalExercise(
            exercise_id=self.generate_exercise_id("dot_prod"),
            topic="vectors",
            difficulty=self.config.difficulty,
            question=question,
            operation="dot_product",
            inputs={"v": v, "w": w},
            expected_output=result,
            hints=hints,
            tags=["dot_product", "generated"],
        )


class VectorNormGenerator(ExerciseGenerator):
    """Generator for vector norm (magnitude) exercises."""

    def generate(self) -> ComputationalExercise:
        """Generate a random vector norm exercise."""
        dimension = self.random_dimension()
        vector = self.random_vector(dimension)

        result = np.linalg.norm(vector)

        question = f"Find the magnitude (norm) of v = {self.format_vector(vector)}"

        hints = [
            f"Magnitude formula: |v| = √(v₁² + v₂² + ...)",
        ]

        # Show squares
        squares = [f"{v:.4g}²" for v in vector]
        sum_squares = sum(v**2 for v in vector)

        hints.append(f"Square each component: {' + '.join(squares)}")
        hints.append(f"= √({sum_squares:.4g}) = {result:.4g}")

        return ComputationalExercise(
            exercise_id=self.generate_exercise_id("vec_norm"),
            topic="vectors",
            difficulty=self.config.difficulty,
            question=question,
            operation="vector_norm",
            inputs={"v": vector},
            expected_output=result,
            hints=hints,
            tags=["vector_norm", "magnitude", "generated"],
        )


class CrossProductGenerator(ExerciseGenerator):
    """Generator for cross product exercises (3D only)."""

    def __init__(self, config: Optional[GeneratorConfig] = None):
        """Initialize with 3D-only configuration."""
        if config is None:
            config = GeneratorConfig(min_dimension=3, max_dimension=3)
        else:
            config.min_dimension = 3
            config.max_dimension = 3
        super().__init__(config)

    def generate(self) -> ComputationalExercise:
        """Generate a random cross product exercise."""
        v = self.random_vector(3)
        w = self.random_vector(3)

        result = np.cross(v, w)

        question = f"Compute the cross product v × w for v = {self.format_vector(v)} and w = {self.format_vector(w)}"

        hints = [
            "Cross product formula: v × w = [v₂w₃ - v₃w₂, v₃w₁ - v₁w₃, v₁w₂ - v₂w₁]",
            f"First component: ({v[1]:.4g})({w[2]:.4g}) - ({v[2]:.4g})({w[1]:.4g}) = {result[0]:.4g}",
            f"Second component: ({v[2]:.4g})({w[0]:.4g}) - ({v[0]:.4g})({w[2]:.4g}) = {result[1]:.4g}",
            f"Third component: ({v[0]:.4g})({w[1]:.4g}) - ({v[1]:.4g})({w[0]:.4g}) = {result[2]:.4g}",
        ]

        return ComputationalExercise(
            exercise_id=self.generate_exercise_id("cross_prod"),
            topic="vectors",
            difficulty=self.config.difficulty,
            question=question,
            operation="cross_product",
            inputs={"v": v, "w": w},
            expected_output=result,
            hints=hints,
            tags=["cross_product", "generated"],
        )
