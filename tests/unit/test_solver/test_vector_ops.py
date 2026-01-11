"""Tests for vector operation solvers."""

import pytest
import numpy as np
from linalg_tutor.core.solver import VectorAdditionSolver, VectorDotProductSolver


def test_vector_addition_solver():
    """Test vector addition solver."""
    solver = VectorAdditionSolver()

    inputs = {"v": np.array([1, 2]), "w": np.array([3, 4])}
    solution = solver.solve_with_steps(inputs, expected=None)

    assert solution.problem_statement is not None
    assert len(solution.steps) > 0
    assert np.array_equal(solution.final_answer, np.array([4, 6]))


def test_vector_dot_product_solver():
    """Test dot product solver."""
    solver = VectorDotProductSolver()

    inputs = {"v": np.array([1, 2, 3]), "w": np.array([4, 5, 6])}
    solution = solver.solve_with_steps(inputs, expected=None)

    assert solution.problem_statement is not None
    assert len(solution.steps) > 0
    # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert solution.final_answer == 32


def test_vector_addition_solution_has_steps():
    """Test that solution contains detailed steps."""
    solver = VectorAdditionSolver()

    inputs = {"v": np.array([2, 3]), "w": np.array([1, -1])}
    solution = solver.solve_with_steps(inputs, expected=None)

    # Check that we have multiple steps
    assert len(solution.steps) >= 2

    # Check that steps have descriptions
    for step in solution.steps:
        assert step.description is not None
        assert len(step.description) > 0
        assert step.mathematical_expression is not None


def test_solver_solve_method():
    """Test the solve() convenience method."""
    solver = VectorAdditionSolver()

    inputs = {"v": np.array([1, 2]), "w": np.array([3, 4])}
    result = solver.solve(inputs)

    assert np.array_equal(result, np.array([4, 6]))
