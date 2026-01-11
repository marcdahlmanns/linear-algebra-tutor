"""Eigenvalue and eigenvector solvers."""

import numpy as np
from typing import List, Tuple
from .simple_solution import SimpleSolution, SimpleSolutionStep


class EigenvalueSolver:
    """Solver for eigenvalues and eigenvectors (numerical approach)."""

    def solve_with_steps(self, A: np.ndarray) -> SimpleSolution:
        """Find eigenvalues and eigenvectors with explanation.

        Args:
            A: Square matrix

        Returns:
            Solution with eigenvalue computation steps
        """
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"Matrix must be square, got {A.shape}")

        n = A.shape[0]
        steps = []

        steps.append(
            SimpleSolutionStep(
                description="Finding Eigenvalues and Eigenvectors",
                mathematical_expression=f"Matrix A ({n}×{n}):\n{self._format_matrix(A)}",
                explanation="Solve Av = λv for eigenvalues λ and eigenvectors v",
            )
        )

        # For 2x2 matrices, show characteristic polynomial explicitly
        if n == 2:
            steps.extend(self._solve_2x2_characteristic(A))
        else:
            steps.append(
                SimpleSolutionStep(
                    description="Characteristic Equation",
                    mathematical_expression="det(A - λI) = 0",
                    explanation=f"For {n}×{n} matrix, we use numerical methods",
                )
            )

        # Compute eigenvalues and eigenvectors using NumPy
        eigenvalues, eigenvectors = np.linalg.eig(A)

        # Sort by magnitude
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        steps.append(
            SimpleSolutionStep(
                description="Eigenvalues computed",
                mathematical_expression=self._format_eigenvalues(eigenvalues),
                explanation=f"Found {len(eigenvalues)} eigenvalue(s)",
            )
        )

        # Show eigenvectors
        for i, (λ, v) in enumerate(zip(eigenvalues, eigenvectors.T)):
            # Normalize to make first non-zero entry positive
            first_nonzero = np.argmax(np.abs(v))
            if v[first_nonzero] < 0:
                v = -v

            steps.append(
                SimpleSolutionStep(
                    description=f"Eigenvector for λ_{i+1} = {λ:.4g}",
                    mathematical_expression=f"v_{i+1} = {self._format_vector(v)}",
                    explanation=f"Verify: Av_{i+1} = {λ:.4g}v_{i+1}",
                )
            )

            # Verification
            Av = A @ v
            λv = λ * v
            error = np.linalg.norm(Av - λv)

            if error < 1e-6:
                steps.append(
                    SimpleSolutionStep(
                        description=f"Verification for λ_{i+1}",
                        mathematical_expression=f"Av_{i+1} = {self._format_vector(Av)}\n{λ:.4g}v_{i+1} = {self._format_vector(λv)}",
                        explanation=f"✓ Error: {error:.2e} (verified)",
                    )
                )

        # Summary
        result = {
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
        }

        steps.append(
            SimpleSolutionStep(
                description="Summary",
                mathematical_expression=self._format_eigen_summary(eigenvalues, eigenvectors),
                explanation="Eigenvalue decomposition complete",
            )
        )

        return SimpleSolution(
            operation="Eigenvalue Decomposition",
            final_answer=result,
            steps=steps,
        )

    def _solve_2x2_characteristic(self, A: np.ndarray) -> List[SimpleSolutionStep]:
        """Show characteristic polynomial for 2×2 matrix."""
        steps = []

        a, b = A[0, 0], A[0, 1]
        c, d = A[1, 0], A[1, 1]

        steps.append(
            SimpleSolutionStep(
                description="Characteristic Polynomial (2×2)",
                mathematical_expression=f"det(A - λI) = det([ {a}-λ    {b}   ]\n"
                                       f"                  [  {c}    {d}-λ ])",
                explanation="Subtract λ from diagonal elements",
            )
        )

        # Expand determinant
        # det = (a-λ)(d-λ) - bc
        trace = a + d
        det = a * d - b * c

        steps.append(
            SimpleSolutionStep(
                description="Expand determinant",
                mathematical_expression=f"({a}-λ)({d}-λ) - ({b})({c})\n"
                                       f"= λ² - {trace}λ + {det}",
                explanation="This gives us a quadratic in λ",
            )
        )

        # Quadratic formula
        discriminant = trace**2 - 4*det

        steps.append(
            SimpleSolutionStep(
                description="Quadratic formula",
                mathematical_expression=f"λ = ({trace} ± √({trace}² - 4({det}))) / 2\n"
                                       f"λ = ({trace} ± √{discriminant:.4g}) / 2",
                explanation="Solve using quadratic formula",
            )
        )

        if discriminant >= 0:
            λ1 = (trace + np.sqrt(discriminant)) / 2
            λ2 = (trace - np.sqrt(discriminant)) / 2
            steps.append(
                SimpleSolutionStep(
                    description="Real eigenvalues",
                    mathematical_expression=f"λ₁ = {λ1:.4g}\nλ₂ = {λ2:.4g}",
                    explanation="Two real eigenvalues (discriminant ≥ 0)",
                )
            )
        else:
            real_part = trace / 2
            imag_part = np.sqrt(-discriminant) / 2
            steps.append(
                SimpleSolutionStep(
                    description="Complex eigenvalues",
                    mathematical_expression=f"λ₁ = {real_part:.4g} + {imag_part:.4g}i\n"
                                           f"λ₂ = {real_part:.4g} - {imag_part:.4g}i",
                    explanation="Complex conjugate pair (discriminant < 0)",
                )
            )

        return steps

    def _format_matrix(self, matrix: np.ndarray) -> str:
        """Format matrix for display."""
        rows = []
        for row in matrix:
            formatted_row = "  ".join(f"{val:8.4g}" for val in row)
            rows.append(f"[ {formatted_row} ]")
        return "\n".join(rows)

    def _format_vector(self, vec: np.ndarray) -> str:
        """Format vector for display."""
        return "[" + ", ".join(f"{val:.4g}" for val in vec) + "]"

    def _format_eigenvalues(self, eigenvalues: np.ndarray) -> str:
        """Format eigenvalues for display."""
        result = []
        for i, λ in enumerate(eigenvalues):
            if np.isreal(λ):
                result.append(f"λ_{i+1} = {λ.real:.4g}")
            else:
                result.append(f"λ_{i+1} = {λ.real:.4g} + {λ.imag:.4g}i")
        return "\n".join(result)

    def _format_eigen_summary(self, eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> str:
        """Format eigenvalue/eigenvector summary."""
        lines = ["Eigenvalue-Eigenvector Pairs:"]
        for i, (λ, v) in enumerate(zip(eigenvalues, eigenvectors.T)):
            if np.isreal(λ):
                lines.append(f"λ_{i+1} = {λ.real:.4g}, v_{i+1} = {self._format_vector(v.real)}")
            else:
                lines.append(f"λ_{i+1} = {λ:.4g}, v_{i+1} = {self._format_vector(v)}")
        return "\n".join(lines)


class CharacteristicPolynomialSolver:
    """Solver for characteristic polynomial of a matrix."""

    def solve_with_steps(self, A: np.ndarray) -> SimpleSolution:
        """Find characteristic polynomial det(A - λI).

        Args:
            A: Square matrix

        Returns:
            Solution with polynomial coefficients
        """
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"Matrix must be square, got {A.shape}")

        n = A.shape[0]
        steps = []

        steps.append(
            SimpleSolutionStep(
                description="Characteristic Polynomial",
                mathematical_expression=f"p(λ) = det(A - λI)\n\nA =\n{self._format_matrix(A)}",
                explanation=f"Find polynomial whose roots are eigenvalues",
            )
        )

        # Compute characteristic polynomial using numpy
        # Note: numpy's poly uses eigenvalues, but we'll show the concept
        coeffs = np.poly(A)

        # Format polynomial
        poly_str = self._format_polynomial(coeffs)

        steps.append(
            SimpleSolutionStep(
                description="Polynomial coefficients",
                mathematical_expression=f"p(λ) = {poly_str}",
                explanation=f"Degree {n} polynomial",
            )
        )

        # Show properties
        trace = np.trace(A)
        det = np.linalg.det(A)

        steps.append(
            SimpleSolutionStep(
                description="Polynomial properties",
                mathematical_expression=f"Coefficient of λ^{n-1}: -{trace:.4g} (negative trace)\n"
                                       f"Constant term: {det:.4g} (determinant)",
                explanation="Trace = sum of eigenvalues, det = product of eigenvalues",
            )
        )

        return SimpleSolution(
            operation="Characteristic Polynomial",
            final_answer=coeffs,
            steps=steps,
        )

    def _format_matrix(self, matrix: np.ndarray) -> str:
        """Format matrix for display."""
        rows = []
        for row in matrix:
            formatted_row = "  ".join(f"{val:8.4g}" for val in row)
            rows.append(f"[ {formatted_row} ]")
        return "\n".join(rows)

    def _format_polynomial(self, coeffs: np.ndarray) -> str:
        """Format polynomial from coefficients."""
        n = len(coeffs) - 1
        terms = []

        for i, c in enumerate(coeffs):
            power = n - i
            if abs(c) < 1e-10:
                continue

            # Format coefficient
            if abs(c - 1) < 1e-10 and power > 0:
                coef_str = ""
            elif abs(c + 1) < 1e-10 and power > 0:
                coef_str = "-"
            else:
                coef_str = f"{c:.4g}"

            # Format power
            if power == 0:
                term = coef_str if coef_str else "1"
            elif power == 1:
                term = f"{coef_str}λ" if coef_str else "λ"
            else:
                term = f"{coef_str}λ^{power}" if coef_str else f"λ^{power}"

            terms.append(term)

        if not terms:
            return "0"

        # Join with + signs, handling negatives
        result = terms[0]
        for term in terms[1:]:
            if term.startswith("-"):
                result += f" - {term[1:]}"
            else:
                result += f" + {term}"

        return result
