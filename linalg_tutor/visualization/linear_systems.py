"""Visualizations for linear systems and Gaussian elimination."""

import numpy as np
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from rich.console import Group


def visualize_augmented_matrix(A: np.ndarray, b: np.ndarray, title: str = "Augmented Matrix") -> Table:
    """Create a visual representation of an augmented matrix [A|b].

    Args:
        A: Coefficient matrix
        b: Right-hand side vector
        title: Title for the table

    Returns:
        Rich Table with the augmented matrix
    """
    m, n = A.shape

    table = Table(title=title, show_header=False, box=None, padding=(0, 1))

    # Add columns for A
    for _ in range(n):
        table.add_column(justify="right")

    # Add separator column
    table.add_column(justify="center", width=1)

    # Add column for b
    table.add_column(justify="right")

    # Add rows
    for i in range(m):
        row_data = []
        # Add A columns
        for j in range(n):
            val = A[i, j]
            row_data.append(f"{val:7.3g}" if abs(val) > 1e-10 else "      0")

        # Add separator
        row_data.append("|")

        # Add b column
        val = b[i]
        row_data.append(f"{val:7.3g}" if abs(val) > 1e-10 else "      0")

        table.add_row(*row_data)

    return table


def visualize_linear_system_2d(A: np.ndarray, b: np.ndarray) -> str:
    """Create ASCII visualization of 2D linear system (two lines intersecting).

    Args:
        A: 2x2 coefficient matrix
        b: 2-element vector

    Returns:
        String representation of the geometric interpretation
    """
    if A.shape != (2, 2) or len(b) != 2:
        return "Geometric visualization only available for 2×2 systems"

    # Extract coefficients for each equation
    # a1*x + b1*y = c1
    # a2*x + b2*y = c2
    a1, b1 = A[0, :]
    a2, b2 = A[1, :]
    c1, c2 = b

    output = []
    output.append("Geometric Interpretation (2D):")
    output.append("")
    output.append(f"Line 1: {a1:g}x + {b1:g}y = {c1:g}")
    output.append(f"Line 2: {a2:g}x + {b2:g}y = {c2:g}")
    output.append("")

    # Check if lines are parallel (determinant = 0)
    det = a1 * b2 - a2 * b1

    if abs(det) < 1e-10:
        # Check if same line or parallel
        if abs(a1 * c2 - a2 * c1) < 1e-10 and abs(b1 * c2 - b2 * c1) < 1e-10:
            output.append("⚠ Lines are IDENTICAL → Infinitely many solutions")
            output.append("   (Same line, all points satisfy both equations)")
        else:
            output.append("⚠ Lines are PARALLEL → No solution")
            output.append("   (Lines never intersect)")
    else:
        # Compute solution
        x = (c1 * b2 - c2 * b1) / det
        y = (a1 * c2 - a2 * c1) / det
        output.append(f"✓ Lines intersect at ONE point → Unique solution")
        output.append(f"   Solution: x = {x:.4g}, y = {y:.4g}")

    return "\n".join(output)


def visualize_row_operation(A_before: np.ndarray, b_before: np.ndarray,
                            A_after: np.ndarray, b_after: np.ndarray,
                            operation_desc: str) -> Group:
    """Visualize a row operation on an augmented matrix.

    Args:
        A_before: Matrix before operation
        b_before: RHS before operation
        A_after: Matrix after operation
        b_after: RHS after operation
        operation_desc: Description of the operation

    Returns:
        Rich Group with before/after comparison
    """
    before = visualize_augmented_matrix(A_before, b_before, "Before")
    after = visualize_augmented_matrix(A_after, b_after, "After")

    return Group(
        before,
        Text(f"\n{operation_desc}\n", style="yellow"),
        after
    )


def visualize_solution_verification(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> str:
    """Verify a solution by showing Ax = b calculation.

    Args:
        A: Coefficient matrix
        b: Right-hand side
        x: Solution vector

    Returns:
        String showing the verification
    """
    Ax = A @ x

    output = []
    output.append("Solution Verification: Ax = b")
    output.append("")
    output.append(f"x = {x}")
    output.append("")
    output.append(f"Ax = {Ax}")
    output.append(f"b  = {b}")
    output.append("")

    # Check if equal (within tolerance)
    if np.allclose(Ax, b):
        output.append("✓ Verification PASSED: Ax = b")
        max_error = np.max(np.abs(Ax - b))
        output.append(f"  (Maximum error: {max_error:.2e})")
    else:
        output.append("✗ Verification FAILED: Ax ≠ b")
        error = np.linalg.norm(Ax - b)
        output.append(f"  (Error: {error:.4g})")

    return "\n".join(output)


def visualize_system_classification(A: np.ndarray, b: np.ndarray) -> str:
    """Classify a linear system (consistent, inconsistent, underdetermined).

    Args:
        A: Coefficient matrix
        b: Right-hand side vector

    Returns:
        String describing the system classification
    """
    m, n = A.shape
    rank_A = np.linalg.matrix_rank(A)

    # Augmented matrix
    Ab = np.column_stack([A, b])
    rank_Ab = np.linalg.matrix_rank(Ab)

    output = []
    output.append("System Classification:")
    output.append("")
    output.append(f"Matrix dimensions: {m}×{n}")
    output.append(f"rank(A) = {rank_A}")
    output.append(f"rank([A|b]) = {rank_Ab}")
    output.append("")

    if rank_A < rank_Ab:
        output.append("⚠ INCONSISTENT: No solution")
        output.append("   (rank(A) < rank([A|b]))")
    elif rank_A == n:
        output.append("✓ UNIQUE SOLUTION")
        output.append("   (rank(A) = number of unknowns)")
    else:
        output.append("⚠ INFINITELY MANY SOLUTIONS")
        output.append(f"   ({n - rank_A} free variables)")

    return "\n".join(output)


def explain_gaussian_elimination() -> str:
    """Explain the Gaussian elimination process.

    Returns:
        String with explanation
    """
    return """Gaussian Elimination Process:

1. FORWARD ELIMINATION:
   • Use row operations to create zeros below each pivot
   • Operations: Swap rows, scale rows, add multiples of rows
   • Goal: Transform matrix to row echelon form (upper triangular)

2. BACK SUBSTITUTION:
   • Start from bottom row
   • Solve for each variable in terms of those already found
   • Work backwards to find all unknowns

3. ROW OPERATIONS (preserve solution set):
   • Swap two rows
   • Multiply a row by a non-zero constant
   • Add a multiple of one row to another row

Example:
  [2  1 | 5]     [2  1 | 5]     x = 2
  [1  3 | 8]  →  [0  5 | 11] →  y = 2.2
"""


def visualize_homogeneous_system(A: np.ndarray) -> str:
    """Visualize properties of homogeneous system Ax = 0.

    Args:
        A: Coefficient matrix

    Returns:
        String describing the homogeneous system
    """
    m, n = A.shape
    rank = np.linalg.matrix_rank(A)
    nullity = n - rank

    output = []
    output.append("Homogeneous System: Ax = 0")
    output.append("")
    output.append(f"Dimensions: {m}×{n}")
    output.append(f"Rank: {rank}")
    output.append(f"Nullity: {nullity}")
    output.append("")
    output.append("Properties:")
    output.append("  • Always has the trivial solution: x = 0")

    if nullity > 0:
        output.append(f"  • Has {nullity} free variable(s)")
        output.append("  • Infinitely many non-trivial solutions")
        output.append(f"  • Solution space is {nullity}-dimensional")
    else:
        output.append("  • Only the trivial solution exists")
        output.append("  • Solution space is just {0}")

    return "\n".join(output)
