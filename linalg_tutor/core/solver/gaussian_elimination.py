"""Gaussian elimination and row reduction solvers."""

import numpy as np
from typing import List, Tuple, Optional
from .simple_solution import SimpleSolution, SimpleSolutionStep


class GaussianEliminationSolver:
    """Solver for Gaussian elimination to row echelon form."""

    def solve_with_steps(self, A: np.ndarray, b: Optional[np.ndarray] = None) -> SimpleSolution:
        """Perform Gaussian elimination with detailed steps.

        Args:
            A: Coefficient matrix
            b: Optional right-hand side vector (for solving linear systems)

        Returns:
            SimpleSolution with step-by-step row operations
        """
        # Create augmented matrix if b is provided
        if b is not None:
            if len(b.shape) == 1:
                b = b.reshape(-1, 1)
            matrix = np.hstack([A.astype(float), b.astype(float)])
            description = "Gaussian Elimination (Augmented Matrix)"
        else:
            matrix = A.astype(float).copy()
            description = "Gaussian Elimination"

        steps = []
        m, n = matrix.shape

        # Initial state
        steps.append(SimpleSolutionStep(
                description="Starting matrix",
                mathematical_expression=self._format_matrix(matrix),
                explanation="We'll perform row operations to get row echelon form (REF)",
            )
        )

        current_row = 0
        for col in range(n - (1 if b is not None else 0)):  # Don't pivot on augmented column
            if current_row >= m:
                break

            # Find pivot
            pivot_row = self._find_pivot(matrix, current_row, col)

            if pivot_row is None:
                steps.append(
                    SimpleSolutionStep(
                        description=f"Column {col}: No pivot available",
                        mathematical_expression=f"All entries in column {col} from row {current_row} onward are zero",
                        explanation="Skip to next column",
                    )
                )
                continue

            # Swap rows if needed
            if pivot_row != current_row:
                matrix[[current_row, pivot_row]] = matrix[[pivot_row, current_row]]
                steps.append(
                    SimpleSolutionStep(
                        description=f"Swap rows {current_row} ↔ {pivot_row}",
                        mathematical_expression=f"R{current_row} ↔ R{pivot_row}",
                        explanation=f"Move non-zero pivot element to row {current_row}",
                        intermediate_result=self._format_matrix(matrix),
                    )
                )

            pivot_value = matrix[current_row, col]

            # Eliminate below pivot
            for row in range(current_row + 1, m):
                if abs(matrix[row, col]) < 1e-10:
                    continue

                factor = matrix[row, col] / pivot_value
                matrix[row] = matrix[row] - factor * matrix[current_row]

                steps.append(
                    SimpleSolutionStep(
                        description=f"Eliminate entry at row {row}, column {col}",
                        mathematical_expression=f"R{row} → R{row} - ({factor:.4g})R{current_row}",
                        explanation=f"Make matrix[{row},{col}] = 0 using row {current_row}",
                        intermediate_result=self._format_matrix(matrix),
                    )
                )

            current_row += 1

        # Clean up near-zero entries
        matrix[np.abs(matrix) < 1e-10] = 0

        # Final result
        steps.append(
            SimpleSolutionStep(
                description="Row Echelon Form (REF) achieved",
                mathematical_expression=self._format_matrix(matrix),
                explanation="Matrix is now in row echelon form with zeros below pivots",
            )
        )

        return SimpleSolution(
            operation=description,
            final_answer=matrix,
            steps=steps,
        )

    def _find_pivot(self, matrix: np.ndarray, start_row: int, col: int) -> Optional[int]:
        """Find the row with the largest absolute value in column (partial pivoting)."""
        m = matrix.shape[0]
        max_val = 0
        max_row = None

        for row in range(start_row, m):
            if abs(matrix[row, col]) > max_val:
                max_val = abs(matrix[row, col])
                max_row = row

        if max_val < 1e-10:
            return None
        return max_row

    def _format_matrix(self, matrix: np.ndarray) -> str:
        """Format matrix for display."""
        rows = []
        for row in matrix:
            formatted_row = "  ".join(f"{val:8.4g}" if abs(val) > 1e-10 else "       0" for val in row)
            rows.append(f"[ {formatted_row} ]")
        return "\n".join(rows)


class RREFSolver:
    """Solver for reduced row echelon form (RREF)."""

    def solve_with_steps(self, A: np.ndarray, b: Optional[np.ndarray] = None) -> SimpleSolution:
        """Perform row reduction to RREF with detailed steps.

        Args:
            A: Coefficient matrix
            b: Optional right-hand side vector

        Returns:
            Solution with step-by-step row operations
        """
        # First get to REF using Gaussian elimination
        ge_solver = GaussianEliminationSolver()
        ref_solution = ge_solver.solve_with_steps(A, b)

        matrix = ref_solution.final_answer.copy()
        steps = ref_solution.steps.copy()

        m, n = matrix.shape

        steps.append(
            SimpleSolutionStep(
                description="Phase 2: Back-substitution to RREF",
                mathematical_expression="",
                explanation="Now we'll make leading entries = 1 and eliminate above pivots",
            )
        )

        # Identify pivot columns
        pivots = []
        for row in range(m):
            for col in range(n):
                if abs(matrix[row, col]) > 1e-10:
                    pivots.append((row, col))
                    break

        # Scale pivot rows to make leading entries = 1
        for row, col in pivots:
            pivot_value = matrix[row, col]
            if abs(pivot_value - 1.0) > 1e-10:
                matrix[row] = matrix[row] / pivot_value
                steps.append(
                    SimpleSolutionStep(
                        description=f"Scale row {row} to make pivot = 1",
                        mathematical_expression=f"R{row} → R{row} / {pivot_value:.4g}",
                        explanation=f"Make leading entry in row {row} equal to 1",
                        intermediate_result=self._format_matrix(matrix),
                    )
                )

        # Eliminate above pivots (back-substitution)
        for row, col in reversed(pivots):
            for above_row in range(row):
                if abs(matrix[above_row, col]) < 1e-10:
                    continue

                factor = matrix[above_row, col]
                matrix[above_row] = matrix[above_row] - factor * matrix[row]

                steps.append(
                    SimpleSolutionStep(
                        description=f"Eliminate entry at row {above_row}, column {col}",
                        mathematical_expression=f"R{above_row} → R{above_row} - ({factor:.4g})R{row}",
                        explanation=f"Make matrix[{above_row},{col}] = 0",
                        intermediate_result=self._format_matrix(matrix),
                    )
                )

        # Clean up near-zero entries
        matrix[np.abs(matrix) < 1e-10] = 0

        steps.append(
            SimpleSolutionStep(
                description="Reduced Row Echelon Form (RREF) achieved",
                mathematical_expression=self._format_matrix(matrix),
                explanation="Matrix is in RREF: leading 1s, zeros above and below pivots",
            )
        )

        return SimpleSolution(
            operation="Row Reduction to RREF",
            final_answer=matrix,
            steps=steps,
        )

    def _format_matrix(self, matrix: np.ndarray) -> str:
        """Format matrix for display."""
        rows = []
        for row in matrix:
            formatted_row = "  ".join(f"{val:8.4g}" if abs(val) > 1e-10 else "       0" for val in row)
            rows.append(f"[ {formatted_row} ]")
        return "\n".join(rows)


class LinearSystemSolver:
    """Solver for systems of linear equations Ax = b."""

    def solve_with_steps(self, A: np.ndarray, b: np.ndarray) -> SimpleSolution:
        """Solve linear system with detailed steps.

        Args:
            A: Coefficient matrix (m × n)
            b: Right-hand side vector (m × 1)

        Returns:
            Solution with step-by-step solution process
        """
        steps = []

        steps.append(
            SimpleSolutionStep(
                description="Linear System",
                mathematical_expression=f"Ax = b\n\nA =\n{self._format_matrix(A)}\n\nb = {b}",
                explanation=f"Solve for x in a {A.shape[0]}×{A.shape[1]} system",
            )
        )

        # Use RREF solver
        rref_solver = RREFSolver()
        rref_solution = rref_solver.solve_with_steps(A, b)

        # Combine steps
        steps.extend(rref_solution.steps)

        augmented = rref_solution.final_answer
        m, n = A.shape

        # Analyze the solution
        # Check for inconsistency (row like [0 0 0 | b] where b ≠ 0)
        for row in range(m):
            left_part = augmented[row, :n]
            right_part = augmented[row, n:]

            if np.allclose(left_part, 0) and not np.allclose(right_part, 0):
                steps.append(
                    SimpleSolutionStep(
                        description="System is INCONSISTENT",
                        mathematical_expression=f"Row {row}: 0 = {right_part[0]:.4g}",
                        explanation="No solution exists (contradiction found)",
                    )
                )
                return SimpleSolution(
                    operation="Solve Linear System",
                    final_answer=None,
                    steps=steps,
                )

        # Extract solution
        solution_vector = np.zeros(n)
        free_variables = []
        pivot_cols = []

        for row in range(min(m, n)):
            # Find leading 1
            leading_col = None
            for col in range(n):
                if abs(augmented[row, col]) > 1e-10:
                    if abs(augmented[row, col] - 1.0) < 1e-10:
                        leading_col = col
                    break

            if leading_col is not None:
                pivot_cols.append(leading_col)
                solution_vector[leading_col] = augmented[row, n] if n < augmented.shape[1] else 0

        # Identify free variables
        for col in range(n):
            if col not in pivot_cols:
                free_variables.append(col)

        if free_variables:
            steps.append(
                SimpleSolutionStep(
                    description="Infinitely many solutions",
                    mathematical_expression=f"Free variables: x_{free_variables}",
                    explanation=f"System has {len(free_variables)} free variable(s)",
                )
            )
        else:
            steps.append(
                SimpleSolutionStep(
                    description="Unique solution found",
                    mathematical_expression=f"x = {solution_vector}",
                    explanation="System has exactly one solution",
                )
            )

        return SimpleSolution(
            operation="Solve Linear System",
            final_answer=solution_vector,
            steps=steps,
        )

    def _format_matrix(self, matrix: np.ndarray) -> str:
        """Format matrix for display."""
        rows = []
        for row in matrix:
            formatted_row = "  ".join(f"{val:8.4g}" if abs(val) > 1e-10 else "       0" for val in row)
            rows.append(f"[ {formatted_row} ]")
        return "\n".join(rows)
