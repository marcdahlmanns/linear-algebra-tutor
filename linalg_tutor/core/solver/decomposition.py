"""Matrix decomposition solvers (LU, QR, SVD)."""

import numpy as np
from typing import Tuple
from .simple_solution import SimpleSolution, SimpleSolutionStep


class LUDecompositionSolver:
    """Solver for LU decomposition A = LU."""

    def solve_with_steps(self, A: np.ndarray) -> SimpleSolution:
        """Compute LU decomposition with steps.

        Args:
            A: Square matrix to decompose

        Returns:
            Solution with L and U matrices
        """
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"Matrix must be square, got {A.shape}")

        n = A.shape[0]
        steps = []

        steps.append(
            SimpleSolutionStep(
                description="LU Decomposition: A = LU",
                mathematical_expression=f"Matrix A ({n}×{n}):\n{self._format_matrix(A)}",
                explanation="Decompose into lower (L) and upper (U) triangular matrices",
            )
        )

        # Initialize L and U
        L = np.eye(n)
        U = A.astype(float).copy()

        steps.append(
            SimpleSolutionStep(
                description="Initialize",
                mathematical_expression=f"L = I (identity)\nU = A (copy of A)",
                explanation="We'll perform Gaussian elimination while recording L",
            )
        )

        # Gaussian elimination with L recording
        for col in range(n - 1):
            pivot = U[col, col]

            if abs(pivot) < 1e-10:
                steps.append(
                    SimpleSolutionStep(
                        description=f"Warning: Small pivot at ({col},{col})",
                        mathematical_expression=f"pivot = {pivot:.2e}",
                        explanation="May need pivoting for numerical stability",
                    )
                )
                continue

            for row in range(col + 1, n):
                if abs(U[row, col]) < 1e-10:
                    continue

                # Compute multiplier
                multiplier = U[row, col] / pivot
                L[row, col] = multiplier

                # Eliminate
                U[row] = U[row] - multiplier * U[col]

                steps.append(
                    SimpleSolutionStep(
                        description=f"Eliminate U[{row},{col}]",
                        mathematical_expression=f"L[{row},{col}] = {multiplier:.4g}\n"
                                               f"U[{row},:] = U[{row},:] - {multiplier:.4g} × U[{col},:]",
                        explanation=f"Store multiplier in L, update U",
                        intermediate_result=f"Current U:\n{self._format_matrix(U)}",
                    )
                )

        # Clean up near-zero
        U[np.abs(U) < 1e-10] = 0
        L[np.abs(L) < 1e-10] = 0

        steps.append(
            SimpleSolutionStep(
                description="Final L (Lower Triangular)",
                mathematical_expression=self._format_matrix(L),
                explanation="L has 1s on diagonal, multipliers below",
            )
        )

        steps.append(
            SimpleSolutionStep(
                description="Final U (Upper Triangular)",
                mathematical_expression=self._format_matrix(U),
                explanation="U is in row echelon form",
            )
        )

        # Verification
        LU = L @ U
        error = np.linalg.norm(A - LU)

        steps.append(
            SimpleSolutionStep(
                description="Verification: L × U",
                mathematical_expression=self._format_matrix(LU),
                explanation=f"Error: {error:.2e} (should be ≈ 0)",
            )
        )

        result = {"L": L, "U": U}

        return SimpleSolution(
            operation="LU Decomposition",
            final_answer=result,
            steps=steps,
        )

    def _format_matrix(self, matrix: np.ndarray) -> str:
        """Format matrix for display."""
        rows = []
        for row in matrix:
            formatted_row = "  ".join(f"{val:8.4g}" if abs(val) > 1e-10 else "       0" for val in row)
            rows.append(f"[ {formatted_row} ]")
        return "\n".join(rows)


class QRDecompositionSolver:
    """Solver for QR decomposition A = QR (Gram-Schmidt process)."""

    def solve_with_steps(self, A: np.ndarray) -> SimpleSolution:
        """Compute QR decomposition using Gram-Schmidt.

        Args:
            A: Matrix to decompose (m × n, m ≥ n)

        Returns:
            Solution with orthogonal Q and upper triangular R
        """
        m, n = A.shape
        steps = []

        steps.append(
            SimpleSolutionStep(
                description="QR Decomposition: A = QR",
                mathematical_expression=f"Matrix A ({m}×{n}):\n{self._format_matrix(A)}",
                explanation="Decompose into orthogonal (Q) and upper triangular (R)",
            )
        )

        steps.append(
            SimpleSolutionStep(
                description="Method: Gram-Schmidt Process",
                mathematical_expression="Orthogonalize columns of A",
                explanation="Create orthonormal basis from column space of A",
            )
        )

        # Initialize
        Q = np.zeros((m, n))
        R = np.zeros((n, n))

        # Gram-Schmidt
        for j in range(n):
            # Start with j-th column of A
            v = A[:, j].copy()

            steps.append(
                SimpleSolutionStep(
                    description=f"Process column {j}",
                    mathematical_expression=f"v_{j} = A[:,{j}] = {self._format_vector(v)}",
                    explanation=f"Start with original column {j}",
                )
            )

            # Subtract projections onto previous orthonormal vectors
            for i in range(j):
                R[i, j] = np.dot(Q[:, i], A[:, j])
                v = v - R[i, j] * Q[:, i]

                if abs(R[i, j]) > 1e-10:
                    steps.append(
                        SimpleSolutionStep(
                            description=f"Orthogonalize against q_{i}",
                            mathematical_expression=f"R[{i},{j}] = q_{i}ᵀa_{j} = {R[i, j]:.4g}\n"
                                                   f"v_{j} -= {R[i, j]:.4g} × q_{i}",
                            explanation=f"Remove component in direction of q_{i}",
                        )
                    )

            # Normalize
            R[j, j] = np.linalg.norm(v)

            if R[j, j] < 1e-10:
                steps.append(
                    SimpleSolutionStep(
                        description=f"Warning: Column {j} is linearly dependent",
                        mathematical_expression=f"||v_{j}|| = {R[j, j]:.2e}",
                        explanation="Column is in span of previous columns",
                    )
                )
                Q[:, j] = 0
            else:
                Q[:, j] = v / R[j, j]

                steps.append(
                    SimpleSolutionStep(
                        description=f"Normalize to get q_{j}",
                        mathematical_expression=f"R[{j},{j}] = ||v_{j}|| = {R[j, j]:.4g}\n"
                                               f"q_{j} = v_{j} / {R[j, j]:.4g}",
                        explanation=f"Unit vector in direction of v_{j}",
                    )
                )

        # Clean up
        Q[np.abs(Q) < 1e-10] = 0
        R[np.abs(R) < 1e-10] = 0

        steps.append(
            SimpleSolutionStep(
                description="Final Q (Orthogonal Matrix)",
                mathematical_expression=self._format_matrix(Q),
                explanation="Q has orthonormal columns: QᵀQ = I",
            )
        )

        steps.append(
            SimpleSolutionStep(
                description="Final R (Upper Triangular)",
                mathematical_expression=self._format_matrix(R),
                explanation="R contains the projection coefficients",
            )
        )

        # Verification
        QR = Q @ R
        error = np.linalg.norm(A - QR)

        steps.append(
            SimpleSolutionStep(
                description="Verification: Q × R",
                mathematical_expression=f"Error: {error:.2e}",
                explanation="Should recover original matrix A",
            )
        )

        # Check orthogonality
        QTQ = Q.T @ Q
        I = np.eye(n)
        orth_error = np.linalg.norm(QTQ - I)

        steps.append(
            SimpleSolutionStep(
                description="Check orthogonality: QᵀQ",
                mathematical_expression=self._format_matrix(QTQ),
                explanation=f"Should be identity (error: {orth_error:.2e})",
            )
        )

        result = {"Q": Q, "R": R}

        return SimpleSolution(
            operation="QR Decomposition",
            final_answer=result,
            steps=steps,
        )

    def _format_matrix(self, matrix: np.ndarray) -> str:
        """Format matrix for display."""
        rows = []
        for row in matrix:
            formatted_row = "  ".join(f"{val:8.4g}" if abs(val) > 1e-10 else "       0" for val in row)
            rows.append(f"[ {formatted_row} ]")
        return "\n".join(rows)

    def _format_vector(self, vec: np.ndarray) -> str:
        """Format vector for display."""
        return "[" + ", ".join(f"{val:.4g}" for val in vec) + "]"


class SVDSolver:
    """Solver for Singular Value Decomposition A = UΣVᵀ."""

    def solve_with_steps(self, A: np.ndarray) -> SimpleSolution:
        """Compute SVD with explanation.

        Args:
            A: Matrix to decompose (m × n)

        Returns:
            Solution with U, Σ, and Vᵀ
        """
        m, n = A.shape
        steps = []

        steps.append(
            SimpleSolutionStep(
                description="Singular Value Decomposition: A = UΣVᵀ",
                mathematical_expression=f"Matrix A ({m}×{n}):\n{self._format_matrix(A)}",
                explanation="Decompose into U (left singular vectors), Σ (singular values), Vᵀ (right singular vectors)",
            )
        )

        # Compute SVD using NumPy
        U, S, VT = np.linalg.svd(A, full_matrices=True)

        steps.append(
            SimpleSolutionStep(
                description="Method",
                mathematical_expression="1. Compute eigenvalues of AᵀA → singular values\n"
                                       "2. Find right singular vectors from AᵀA\n"
                                       "3. Find left singular vectors from AAᵀ",
                explanation="Using numerical SVD algorithm",
            )
        )

        # Show singular values
        steps.append(
            SimpleSolutionStep(
                description="Singular Values",
                mathematical_expression="\n".join([f"σ_{i+1} = {s:.4g}" for i, s in enumerate(S)]),
                explanation=f"Found {len(S)} singular value(s), sorted in descending order",
            )
        )

        # Create full Σ matrix
        Sigma = np.zeros((m, n))
        Sigma[:len(S), :len(S)] = np.diag(S)

        steps.append(
            SimpleSolutionStep(
                description=f"Σ Matrix ({m}×{n})",
                mathematical_expression=self._format_matrix(Sigma),
                explanation="Diagonal matrix of singular values (padded with zeros)",
            )
        )

        steps.append(
            SimpleSolutionStep(
                description=f"U Matrix ({m}×{m})",
                mathematical_expression=self._format_matrix(U),
                explanation="Left singular vectors (orthonormal columns)",
            )
        )

        steps.append(
            SimpleSolutionStep(
                description=f"Vᵀ Matrix ({n}×{n})",
                mathematical_expression=self._format_matrix(VT),
                explanation="Right singular vectors transposed (orthonormal rows)",
            )
        )

        # Verification
        reconstructed = U @ Sigma @ VT
        error = np.linalg.norm(A - reconstructed)

        steps.append(
            SimpleSolutionStep(
                description="Verification: UΣVᵀ",
                mathematical_expression=self._format_matrix(reconstructed),
                explanation=f"Error: {error:.2e} (should recover A)",
            )
        )

        # Rank analysis
        rank = np.sum(S > 1e-10)
        steps.append(
            SimpleSolutionStep(
                description="Matrix Properties",
                mathematical_expression=f"Rank: {rank} (number of non-zero singular values)\n"
                                       f"Condition number: {S[0]/S[-1] if S[-1] > 1e-10 else np.inf:.4g}",
                explanation="Rank and condition number from singular values",
            )
        )

        result = {"U": U, "S": S, "Sigma": Sigma, "VT": VT}

        return SimpleSolution(
            operation="Singular Value Decomposition",
            final_answer=result,
            steps=steps,
        )

    def _format_matrix(self, matrix: np.ndarray) -> str:
        """Format matrix for display."""
        rows = []
        for row in matrix:
            formatted_row = "  ".join(f"{val:8.4g}" if abs(val) > 1e-10 else "       0" for val in row)
            rows.append(f"[ {formatted_row} ]")
        return "\n".join(rows)
