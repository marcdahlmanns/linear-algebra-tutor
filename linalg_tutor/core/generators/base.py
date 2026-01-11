"""Base classes for exercise generators."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import random
import numpy as np

from ..exercises import Exercise, ExerciseDifficulty


@dataclass
class GeneratorConfig:
    """Configuration for exercise generation."""

    # Difficulty settings
    difficulty: ExerciseDifficulty = ExerciseDifficulty.PRACTICE

    # Dimension settings
    min_dimension: int = 2
    max_dimension: int = 3

    # Value ranges
    min_value: int = -10
    max_value: int = 10

    # Integer constraints
    integers_only: bool = True

    # Special constraints
    avoid_zero: bool = True
    avoid_singular: bool = True  # For matrices

    # Seed for reproducibility
    seed: Optional[int] = None

    def __post_init__(self):
        """Initialize random seed if provided."""
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)


class ExerciseGenerator(ABC):
    """Abstract base class for exercise generators.

    Generators create randomized exercises with controlled parameters
    for infinite practice.
    """

    def __init__(self, config: Optional[GeneratorConfig] = None):
        """Initialize generator with configuration.

        Args:
            config: Generator configuration (uses defaults if None)
        """
        self.config = config or GeneratorConfig()

    @abstractmethod
    def generate(self) -> Exercise:
        """Generate a new random exercise.

        Returns:
            A randomly generated exercise instance
        """
        pass

    def generate_batch(self, count: int) -> List[Exercise]:
        """Generate multiple exercises.

        Args:
            count: Number of exercises to generate

        Returns:
            List of generated exercises
        """
        return [self.generate() for _ in range(count)]

    # Helper methods for subclasses

    def random_dimension(self) -> int:
        """Generate a random dimension within configured range."""
        return random.randint(self.config.min_dimension, self.config.max_dimension)

    def random_value(self) -> float:
        """Generate a random value within configured range."""
        if self.config.integers_only:
            value = random.randint(self.config.min_value, self.config.max_value)
            if self.config.avoid_zero and value == 0:
                value = random.choice([self.config.min_value, self.config.max_value])
            return float(value)
        else:
            return random.uniform(self.config.min_value, self.config.max_value)

    def random_vector(self, dimension: Optional[int] = None) -> np.ndarray:
        """Generate a random vector.

        Args:
            dimension: Vector dimension (random if None)

        Returns:
            Random numpy array
        """
        if dimension is None:
            dimension = self.random_dimension()

        vector = np.array([self.random_value() for _ in range(dimension)])

        # Ensure at least one non-zero if avoiding all zeros
        if self.config.avoid_zero and np.allclose(vector, 0):
            vector[0] = self.random_value()

        return vector

    def random_matrix(self, rows: Optional[int] = None, cols: Optional[int] = None) -> np.ndarray:
        """Generate a random matrix.

        Args:
            rows: Number of rows (random if None)
            cols: Number of columns (random if None)

        Returns:
            Random numpy array
        """
        if rows is None:
            rows = self.random_dimension()
        if cols is None:
            cols = self.random_dimension()

        matrix = np.array([[self.random_value() for _ in range(cols)] for _ in range(rows)])

        # Check for singular matrix if square and avoiding singular
        if self.config.avoid_singular and rows == cols:
            attempts = 0
            while abs(np.linalg.det(matrix)) < 1e-6 and attempts < 10:
                matrix = np.array([[self.random_value() for _ in range(cols)] for _ in range(rows)])
                attempts += 1

        return matrix

    def random_scalar(self, nonzero: bool = False) -> float:
        """Generate a random scalar.

        Args:
            nonzero: If True, ensures scalar is not zero

        Returns:
            Random scalar value
        """
        value = self.random_value()
        if nonzero and abs(value) < 1e-6:
            value = random.choice([self.config.min_value, self.config.max_value])
        return value

    def format_vector(self, vec: np.ndarray) -> str:
        """Format vector for display in question."""
        if self.config.integers_only:
            return "[" + ", ".join(str(int(v)) for v in vec) + "]"
        return "[" + ", ".join(f"{v:.2g}" for v in vec) + "]"

    def format_matrix(self, mat: np.ndarray) -> str:
        """Format matrix for display in question."""
        rows = []
        for row in mat:
            if self.config.integers_only:
                row_str = "[" + ", ".join(str(int(v)) for v in row) + "]"
            else:
                row_str = "[" + ", ".join(f"{v:.2g}" for v in row) + "]"
            rows.append(row_str)
        return "[" + ", ".join(rows) + "]"

    def generate_exercise_id(self, prefix: str) -> str:
        """Generate a unique exercise ID.

        Args:
            prefix: Prefix for the ID (e.g., 'vec_add', 'mat_mult')

        Returns:
            Unique exercise ID with random suffix
        """
        random_suffix = random.randint(10000, 99999)
        return f"{prefix}_gen_{random_suffix}"
