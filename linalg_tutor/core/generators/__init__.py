"""Exercise generator system for infinite practice."""

from .base import ExerciseGenerator, GeneratorConfig
from .vector_generators import (
    VectorAdditionGenerator,
    VectorScalarMultGenerator,
    DotProductGenerator,
    VectorNormGenerator,
    CrossProductGenerator,
)
from .matrix_generators import (
    MatrixAdditionGenerator,
    MatrixScalarMultGenerator,
    MatrixMultiplicationGenerator,
    MatrixTransposeGenerator,
    DeterminantGenerator,
    MatrixInverseGenerator,
)
from .linear_systems import (
    LinearSystemGenerator,
    TriangularSystemGenerator,
    Simple2x2SystemGenerator,
)

# Generator registry for easy access
GENERATOR_REGISTRY = {
    "vector_add": VectorAdditionGenerator,
    "vector_scalar": VectorScalarMultGenerator,
    "dot_product": DotProductGenerator,
    "vector_norm": VectorNormGenerator,
    "cross_product": CrossProductGenerator,
    "matrix_add": MatrixAdditionGenerator,
    "matrix_scalar": MatrixScalarMultGenerator,
    "matrix_multiply": MatrixMultiplicationGenerator,
    "matrix_transpose": MatrixTransposeGenerator,
    "determinant": DeterminantGenerator,
    "matrix_inverse": MatrixInverseGenerator,
    "linear_system": LinearSystemGenerator,
    "triangular_system": TriangularSystemGenerator,
    "simple_2x2_system": Simple2x2SystemGenerator,
}


def get_generator(name: str, config: GeneratorConfig = None):
    """Get a generator instance by name.

    Args:
        name: Generator name (e.g., 'vector_add', 'matrix_multiply')
        config: Optional generator configuration

    Returns:
        Generator instance
    """
    generator_class = GENERATOR_REGISTRY.get(name)
    if generator_class is None:
        raise ValueError(f"Unknown generator: {name}. Available: {list(GENERATOR_REGISTRY.keys())}")
    return generator_class(config)


__all__ = [
    "ExerciseGenerator",
    "GeneratorConfig",
    "VectorAdditionGenerator",
    "VectorScalarMultGenerator",
    "DotProductGenerator",
    "VectorNormGenerator",
    "CrossProductGenerator",
    "MatrixAdditionGenerator",
    "MatrixScalarMultGenerator",
    "MatrixMultiplicationGenerator",
    "MatrixTransposeGenerator",
    "DeterminantGenerator",
    "MatrixInverseGenerator",
    "LinearSystemGenerator",
    "TriangularSystemGenerator",
    "Simple2x2SystemGenerator",
    "GENERATOR_REGISTRY",
    "get_generator",
]
