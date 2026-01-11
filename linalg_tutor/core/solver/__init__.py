"""Step-by-step solver system."""

from typing import Optional

from .base import Solver, Solution, SolutionStep
from .vector_ops import VectorAdditionSolver, VectorDotProductSolver, VectorScalarMultiplySolver
from .matrix_ops import MatrixMultiplySolver, MatrixAdditionSolver, MatrixTransposeSolver
from .gaussian_elimination import GaussianEliminationSolver, RREFSolver, LinearSystemSolver
from .eigenvalue import EigenvalueSolver, CharacteristicPolynomialSolver
from .decomposition import LUDecompositionSolver, QRDecompositionSolver, SVDSolver

# Registry of available solvers
_SOLVER_REGISTRY = {
    "vector_add": VectorAdditionSolver,
    "vector_dot": VectorDotProductSolver,
    "vector_scalar_multiply": VectorScalarMultiplySolver,
    "matrix_multiply": MatrixMultiplySolver,
    "matrix_add": MatrixAdditionSolver,
    "matrix_transpose": MatrixTransposeSolver,
    "gaussian_elimination": GaussianEliminationSolver,
    "rref": RREFSolver,
    "linear_system": LinearSystemSolver,
    "eigenvalue": EigenvalueSolver,
    "characteristic_polynomial": CharacteristicPolynomialSolver,
    "lu_decomposition": LUDecompositionSolver,
    "qr_decomposition": QRDecompositionSolver,
    "svd": SVDSolver,
}


def get_solver(operation: str) -> Optional[Solver]:
    """Get a solver instance for the given operation.

    Args:
        operation: Name of the operation (e.g., 'matrix_multiply')

    Returns:
        Solver instance, or None if operation not found
    """
    solver_class = _SOLVER_REGISTRY.get(operation)
    if solver_class:
        return solver_class()
    return None


def register_solver(operation: str, solver_class: type):
    """Register a new solver for an operation.

    Args:
        operation: Name of the operation
        solver_class: Solver class to register
    """
    _SOLVER_REGISTRY[operation] = solver_class


__all__ = [
    "Solver",
    "Solution",
    "SolutionStep",
    "VectorAdditionSolver",
    "VectorDotProductSolver",
    "VectorScalarMultiplySolver",
    "MatrixMultiplySolver",
    "MatrixAdditionSolver",
    "MatrixTransposeSolver",
    "GaussianEliminationSolver",
    "RREFSolver",
    "LinearSystemSolver",
    "EigenvalueSolver",
    "CharacteristicPolynomialSolver",
    "LUDecompositionSolver",
    "QRDecompositionSolver",
    "SVDSolver",
    "get_solver",
    "register_solver",
]
