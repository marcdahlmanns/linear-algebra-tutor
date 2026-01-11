"""Solvers for matrix operations."""

from typing import Any, Dict

import numpy as np

from .base import Solver, Solution, SolutionStep


class MatrixMultiplySolver(Solver):
    """Step-by-step solver for matrix multiplication."""

    def solve_with_steps(self, inputs: Dict[str, Any], expected: Any) -> Solution:
        """Generate step-by-step solution for matrix multiplication.

        Args:
            inputs: Dict with keys 'A' and 'B' (matrices to multiply)
            expected: Expected result (optional)

        Returns:
            Solution with steps
        """
        A = np.array(inputs['A'])
        B = np.array(inputs['B'])

        m, n = A.shape
        n2, p = B.shape

        steps = []

        # Step 1: Check dimensions
        steps.append(
            SolutionStep(
                step_number=1,
                description="Check dimension compatibility",
                mathematical_expression=f"A is {m}×{n}, B is {n2}×{p}",
                explanation=f"Matrices are compatible because inner dimensions match ({n} = {n2}). Result will be {m}×{p}.",
            )
        )

        # Step 2: Explain computation method
        steps.append(
            SolutionStep(
                step_number=2,
                description="Matrix multiplication formula",
                mathematical_expression="C[i,j] = Σ(A[i,k] × B[k,j]) for k=0 to n-1",
                explanation="Each element C[i,j] is the dot product of row i of A with column j of B",
            )
        )

        # Step 3: Compute result
        C = A @ B

        # For small matrices, show element-by-element computation
        if m * p <= 6:  # Show details for small matrices only
            for i in range(m):
                for j in range(p):
                    # Compute dot product step
                    dot_terms = [f"({A[i, k]})({B[k, j]})" for k in range(n)]
                    dot_expr = " + ".join(dot_terms)
                    dot_values = [A[i, k] * B[k, j] for k in range(n)]
                    dot_sum = sum(dot_values)

                    steps.append(
                        SolutionStep(
                            step_number=len(steps) + 1,
                            description=f"Compute C[{i},{j}]",
                            mathematical_expression=f"C[{i},{j}] = {dot_expr} = {' + '.join(map(str, dot_values))} = {dot_sum}",
                            explanation=f"Dot product of row {i} of A with column {j} of B",
                            intermediate_result=dot_sum,
                        )
                    )
        else:
            # For larger matrices, just show the formula was applied
            steps.append(
                SolutionStep(
                    step_number=3,
                    description="Apply formula to all elements",
                    mathematical_expression=f"Compute all {m}×{p} = {m*p} elements using the formula",
                    explanation="Each element is computed as a dot product",
                )
            )

        # Final step: show result
        steps.append(
            SolutionStep(
                step_number=len(steps) + 1,
                description="Final result",
                mathematical_expression=f"C = {C.tolist()}",
                explanation=f"This is the {m}×{p} matrix result",
            )
        )

        return Solution(
            problem_statement=f"Multiply matrices A ({m}×{n}) and B ({n2}×{p})",
            steps=steps,
            final_answer=C,
            verification=f"Verify: Result is {m}×{p} as expected",
        )


class MatrixAdditionSolver(Solver):
    """Step-by-step solver for matrix addition."""

    def solve_with_steps(self, inputs: Dict[str, Any], expected: Any) -> Solution:
        """Generate step-by-step solution for matrix addition.

        Args:
            inputs: Dict with keys 'A' and 'B' (matrices to add)
            expected: Expected result (optional)

        Returns:
            Solution with steps
        """
        A = np.array(inputs['A'])
        B = np.array(inputs['B'])

        m, n = A.shape

        steps = []

        # Step 1: Check dimensions
        steps.append(
            SolutionStep(
                step_number=1,
                description="Check dimensions",
                mathematical_expression=f"A is {m}×{n}, B is {B.shape[0]}×{B.shape[1]}",
                explanation="Matrices must have the same dimensions for addition",
            )
        )

        # Step 2: Add element-wise
        C = A + B

        steps.append(
            SolutionStep(
                step_number=2,
                description="Add corresponding elements",
                mathematical_expression="C[i,j] = A[i,j] + B[i,j] for all i,j",
                explanation="Matrix addition is performed element-wise",
                intermediate_result=C.tolist(),
            )
        )

        # Step 3: Show result
        steps.append(
            SolutionStep(
                step_number=3,
                description="Final result",
                mathematical_expression=f"C = {C.tolist()}",
                explanation=f"This is the {m}×{n} matrix result",
            )
        )

        return Solution(
            problem_statement=f"Add matrices A and B (both {m}×{n})",
            steps=steps,
            final_answer=C,
            verification=f"Result has same dimensions as inputs: {m}×{n}",
        )


class MatrixTransposeSolver(Solver):
    """Step-by-step solver for matrix transpose."""

    def solve_with_steps(self, inputs: Dict[str, Any], expected: Any) -> Solution:
        """Generate step-by-step solution for matrix transpose.

        Args:
            inputs: Dict with key 'A' (matrix to transpose)
            expected: Expected result (optional)

        Returns:
            Solution with steps
        """
        A = np.array(inputs['A'])
        m, n = A.shape

        steps = []

        # Step 1: Explain transpose
        steps.append(
            SolutionStep(
                step_number=1,
                description="Transpose definition",
                mathematical_expression=f"A is {m}×{n}, A^T will be {n}×{m}",
                explanation="Transpose swaps rows and columns: (A^T)[i,j] = A[j,i]",
            )
        )

        # Step 2: Compute transpose
        A_T = A.T

        steps.append(
            SolutionStep(
                step_number=2,
                description="Swap rows and columns",
                mathematical_expression="Rows become columns, columns become rows",
                explanation=f"Row i of A becomes column i of A^T",
                intermediate_result=A_T.tolist(),
            )
        )

        # Step 3: Show result
        steps.append(
            SolutionStep(
                step_number=3,
                description="Final result",
                mathematical_expression=f"A^T = {A_T.tolist()}",
                explanation=f"This is the {n}×{m} transposed matrix",
            )
        )

        return Solution(
            problem_statement=f"Transpose matrix A ({m}×{n})",
            steps=steps,
            final_answer=A_T,
            verification=f"Verify: A^T is {n}×{m}",
        )
